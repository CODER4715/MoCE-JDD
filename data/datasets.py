import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from data.unprocessor import ImageUnprocessor, IspProcessor
from matplotlib import pyplot as plt
from lightning.pytorch import seed_everything
from abc import ABC, abstractmethod


class BaseVideoDataset(Dataset, ABC):
    """
    数据集的抽象基类，封装了视频数据加载和处理的通用逻辑。

    子类需要实现 _initialize_metadata 和 _map_index_to_paths 方法。
    """

    def __init__(self, root_dir, bundle_frame=5, patch_size=None, transform=None, get_srgb=False, args=None):
        """
        初始化基类
        Args:
            root_dir (str): 数据集根目录
            bundle_frame (int): 每个样本包含的帧数 (必须是奇数)
            patch_size (int, optional): 图像裁剪的大小。如果为 None，则不进行裁剪。默认为 None。
            transform (callable, optional): 应用于样本的可选变换。
            get_srgb (bool): 是否在返回数据中包含 sRGB 图像。
            args (Namespace, optional): 其他参数，例如 trans12bit。
        """
        super().__init__()
        assert bundle_frame % 2 == 1, 'Bundle_frame must be an odd number'

        self.root_dir = root_dir
        self.bundle_frame = bundle_frame
        self.patch_size = patch_size
        self.transform = transform
        self.get_srgb = get_srgb
        self.n = (bundle_frame - 1) // 2  # 前后帧的偏移量

        self.unprocessor = ImageUnprocessor()
        self.trans12bit = args.trans12bit if args else False

        # 由子类负责初始化以下属性
        self.video_dirs = []
        self.video_params = []
        self.video_frame_counts = []

        # 调用子类的实现来填充元数据
        self._initialize_metadata()
        print(f"Initialized {self.__class__.__name__} with {len(self.video_dirs)} videos.")

    @abstractmethod
    def _initialize_metadata(self):
        """
        抽象方法：初始化数据集元数据。
        子类必须实现此方法来填充:
        - self.video_dirs: 视频目录列表
        - self.video_frame_counts: 每个视频的帧数列表
        - self.video_params: 每个视频的ISP参数
        """
        raise NotImplementedError

    @abstractmethod
    def _map_index_to_paths(self, idx):
        """
        抽象方法：将全局索引映射到具体的视频和帧索引。
        子类必须实现此方法。
        Args:
            idx (int): 数据集的全局索引。
        Returns:
            tuple: (video_idx, frame_idx, total_frames_in_video)
        """
        raise NotImplementedError

    def _generate_isp_params(self):
        """为单个视频生成一套随机ISP参数"""
        rgb2cam = self.unprocessor.random_ccm()
        cam2rgb = torch.inverse(rgb2cam)
        rgb_gain, red_gain, blue_gain = self.unprocessor.random_gains()
        shot_noise, read_noise = self.unprocessor.random_noise_levels()
        return {
            'rgb2cam': rgb2cam.cpu(), 'cam2rgb': cam2rgb.cpu(),
            'rgb_gain': rgb_gain.cpu(), 'red_gain': red_gain.cpu(),
            'blue_gain': blue_gain.cpu(), 'shot_noise': shot_noise.cpu(),
            'read_noise': read_noise.cpu()
        }

    def __getitem__(self, idx):
        # 1. 将全局索引映射到视频和帧
        video_idx, frame_idx, total_frames = self._map_index_to_paths(idx)
        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])
        video_dir_name = self.video_dirs[video_idx]

        # 2. 计算需要加载的帧的索引，并处理边界情况
        frame_indices = list(range(max(0, frame_idx - self.n), min(total_frames, frame_idx + self.n + 1)))

        # 填充 (Padding)
        if frame_idx < self.n:
            pad_count = self.n - frame_idx
            frame_indices = [0] * pad_count + frame_indices
        elif frame_idx + self.n >= total_frames:
            pad_count = (frame_idx + self.n + 1) - total_frames
            frame_indices = frame_indices + [total_frames - 1] * pad_count

        # 3. 获取该视频共享的ISP参数
        params = self.video_params[video_idx]

        noise_frames = []
        top, left = None, None
        center_frame_index_in_bundle = self.n

        for i, frame_num in enumerate(frame_indices):
            img_path = os.path.join(video_dir, f"{frame_num:08d}.png")
            image = Image.open(img_path).convert('RGB')

            # 4. 随机裁剪 (如果 patch_size 被设置)
            if self.patch_size:
                if top is None or left is None:  # 保证一个 'bundle' 内的裁剪位置相同
                    w, h = image.size
                    top = torch.randint(0, h - self.patch_size, (1,)).item()
                    left = torch.randint(0, w - self.patch_size, (1,)).item()
                image = image.crop((left, top, left + self.patch_size, top + self.patch_size))

            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)

            # 5. 应用反ISP流程生成RAW数据
            _, noise_image, linear_RGB, _ = self.unprocessor.forward(
                image_tensor,
                add_noise=True,
                shot_noise=params['shot_noise'],
                read_noise=params['read_noise'],
                rgb2cam=params['rgb2cam'],
                rgb_gain=params['rgb_gain'],
                red_gain=params['red_gain'],
                blue_gain=params['blue_gain']
            )
            if self.trans12bit:
                noise_image = noise_image * (4095 - 240) + 240  # 240为黑电平

            if i == center_frame_index_in_bundle:
                linear_RGB_frame = linear_RGB.permute(2, 0, 1)
                if self.trans12bit:
                    linear_RGB_frame = linear_RGB_frame * (4095 - 240) + 240
                if self.get_srgb:
                    srgb_frame = image_tensor.permute(2, 0, 1)

            noise_frames.append(noise_image)

        # 6. 堆叠并整理Tensor维度
        noise_frames = torch.stack(noise_frames).permute(0, 3, 1, 2)

        # 7. 构建返回字典
        data = {
            'raw_noise': noise_frames.cpu(),
            'linear_RGB': linear_RGB_frame.cpu(),
            'metadata': params,
            'frame_idx': {'video': video_idx, 'frame': frame_idx},
            'video_name': video_dir_name
        }
        if self.get_srgb:
            data['srgb'] = srgb_frame.cpu()

        return data


class FixedFramesDataset(BaseVideoDataset):
    """
    用于每个视频片段帧数固定的数据集，例如 REDS。
    """

    def __init__(self, root_dir, frames_per_video=500, downsample_ratio=1, is_train=True, seed=42, **kwargs):
        self.frames_per_video = frames_per_video
        self.downsample_ratio = downsample_ratio
        self.is_train = is_train
        self.seed = seed
        # 训练时随机裁剪，验证时不裁剪
        if 'patch_size' not in kwargs and is_train:
            kwargs['patch_size'] = 256  # 默认值
        elif not is_train:
            kwargs['patch_size'] = None

        super().__init__(root_dir, **kwargs)

    def _initialize_metadata(self):
        self.video_dirs = sorted(
            [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.video_frame_counts = [self.frames_per_video] * len(self.video_dirs)

        # 验证集使用固定种子以保证ISP参数一致
        if not self.is_train:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.video_params = [self._generate_isp_params() for _ in self.video_dirs]

    def __len__(self):
        return len(self.video_dirs) * (self.frames_per_video // self.downsample_ratio)

    def _map_index_to_paths(self, idx):
        frames_per_video_downsampled = self.frames_per_video // self.downsample_ratio
        video_idx = idx // frames_per_video_downsampled
        frame_idx_in_video = (idx % frames_per_video_downsampled) * self.downsample_ratio
        return video_idx, frame_idx_in_video, self.frames_per_video


class VariableFramesDataset(BaseVideoDataset):
    """
    用于每个视频片段帧数不固定的数据集，例如 TVD, VideoMME。
    """

    def __init__(self, root_dir, is_train=True, seed=42, **kwargs):

        self.is_train = is_train
        self.seed = seed

        if 'patch_size' not in kwargs and is_train:
            kwargs['patch_size'] = 256
        elif not is_train:
            kwargs['patch_size'] = None
        super().__init__(root_dir, **kwargs)

    def _initialize_metadata(self):
        self.video_dirs = sorted(
            [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        for video_dir in self.video_dirs:
            path = os.path.join(self.root_dir, video_dir)
            self.video_frame_counts.append(len([f for f in os.listdir(path) if f.endswith('.png')]))

        # 验证集使用固定种子以保证ISP参数一致
        if not self.is_train:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.cumulative_lengths = np.cumsum([0] + self.video_frame_counts)
        self.video_params = [self._generate_isp_params() for _ in self.video_dirs]

    def __len__(self):
        return sum(self.video_frame_counts)

    def _map_index_to_paths(self, idx):
        # 找到idx属于哪个视频
        video_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        # 计算在该视频中的局部索引
        frame_idx_in_video = idx - self.cumulative_lengths[video_idx]
        total_frames = self.video_frame_counts[video_idx]
        return video_idx, frame_idx_in_video, total_frames


class CombinedDataset(Dataset):
    """
    将多个数据集合并为一个。
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = np.cumsum([0] + [len(d) for d in datasets])

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


# --- 主程序入口 ---
if __name__ == '__main__':
    seed_everything(42)

    # === 使用新的类来创建数据集 ===

    # 1. 创建 REDS 训练集
    # 原来的 REDS_Train_Dataset
    reds_train_dataset = FixedFramesDataset(
        root_dir='H:/datasets/JDD/REDS120/train/train_orig',
        bundle_frame=5,
        patch_size=256,
        is_train=True,
        get_srgb=True
    )
    print(f"REDS Train dataset length: {len(reds_train_dataset)}")

    # 2. 创建 REDS 验证集
    # 原来的 REDS_Val_Dataset
    reds_val_dataset = FixedFramesDataset(
        root_dir='H:/datasets/JDD/REDS120/val/val_orig',  # 请替换为你的验证集路径
        bundle_frame=5,
        is_train=False,  # 关键参数，设为False，则不进行patch crop
        seed=42,
        get_srgb=True
    )
    # print(f"REDS Val dataset length: {len(reds_val_dataset)}")

    # 3. 创建 TVD 数据集
    # 原来的 TVD
    tvd_dataset = VariableFramesDataset(
        root_dir=r'F:\datasets\Tencent_Video_Dataset\Video\frames',
        bundle_frame=5,
        patch_size=256,
        get_srgb=True
    )
    print(f"TVD dataset length: {len(tvd_dataset)}")

    # # 4. 创建 VideoMME 数据集
    # videomme_dataset = VariableFramesDataset(
    #     root_dir='path/to/VideoMME',  # 请替换为你的VideoMME路径
    #     bundle_frame=5,
    #     patch_size=256,
    #     get_srgb=True
    # )
    # print(f"VideoMME dataset length: {len(videomme_dataset)}")

    # 5. 合并数据集 (例如合并 REDS 和 TVD)
    combined_dataset = CombinedDataset([reds_train_dataset, tvd_dataset])
    print(f"Combined dataset length: {len(combined_dataset)}")

    # === 测试 DataLoader ===
    # 使用 TVD 数据集进行测试
    train_loader = torch.utils.data.DataLoader(tvd_dataset, batch_size=4, shuffle=True)
    batch = next(iter(train_loader))

    print("\n--- Batch Info ---")
    print(f"Raw Noise shape: {batch['raw_noise'].shape}")
    print(f"Linear RGB shape: {batch['linear_RGB'].shape}")
    print(f"sRGB shape: {batch['srgb'].shape}")
    print("Metadata keys:", list(batch['metadata'].keys()))
    for key, value in batch['metadata'].items():
        print(f"  - {key}: {value.shape}")

    # === 测试 ISP 流程 (与原代码一致) ===
    # 注意：IspProcessor 需要你的实现，这里假设它存在且可用
    # isp = IspProcessor()
    # linear2srgbs = isp.process(batch['linear_RGB'], batch['metadata']['red_gain'],
    #                                    batch['metadata']['blue_gain'],
    #                                    batch['metadata']['cam2rgb'],
    #                                     batch['metadata']['rgb_gain'], dem=False)

    # print(f"\nProcessed sRGB shape: {linear2srgbs.shape}")
    # # 可视化对比
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(linear2srgbs[0].cpu().permute(1, 2, 0)) # IspProcessor 输出可能是 (C,H,W)
    # plt.title("sRGB from Linear RGB")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(batch['srgb'][0].cpu().permute(1, 2, 0))
    # plt.title("Ground Truth sRGB")
    # plt.axis('off')
    # plt.show()