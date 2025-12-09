import os

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.unprocessor import ImageUnprocessor, IspProcessor
import numpy as np
from matplotlib import pyplot as plt
from lightning.pytorch import seed_everything


class REDS_Train_Dataset(Dataset):
    def __init__(self, root_dir, bundle_frame=5, patch_size=256,  transform=None, downsample_ratio=1, args=None):
        """
        初始化数据集
        Args:
            root_dir (str): 包含视频片段的根目录
            bundle_frame (int): 每个batch包含的帧数(奇数)
            patch_size (int): 图像裁剪的大小
            transform (callable, optional): 可选的图像变换
            downsample_ratio (int): 帧率降采样比率，默认为1（不进行降采样）
        """
        assert bundle_frame % 2 == 1, 'Bundle_frame must be Odd number'
        self.root_dir = root_dir
        self.transform = transform
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2  # 计算前后帧偏移量
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio

        self.unprocessor = ImageUnprocessor()

        self.get_srgb = False
        if args is not None:
            self.trans12bit = args.trans12bit
        else:
            self.trans12bit = False

        # 获取所有视频片段目录
        self.video_dirs = [d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))]

        self.video_params = []

        for _ in self.video_dirs:
            rgb2cam = self.unprocessor.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            # rgb2cam = unprocessor.rgb2cam
            # cam2rgb = unprocessor.cam2rgb

            rgb_gain, red_gain, blue_gain = self.unprocessor.random_gains()
            shot_noise, read_noise = self.unprocessor.random_noise_levels()
            self.video_params.append({
                'rgb2cam': rgb2cam.cpu(),
                'cam2rgb': cam2rgb.cpu(),
                'rgb_gain': rgb_gain.cpu(),
                'red_gain': red_gain.cpu(),
                'blue_gain': blue_gain.cpu(),
                'shot_noise': shot_noise.cpu(),
                'read_noise': read_noise.cpu()
            })


        print(len(self.video_params))


    def __len__(self):
        return len(self.video_dirs) * (500 // self.downsample_ratio)

    def __getitem__(self, idx):
        video_idx = idx // (500 // self.downsample_ratio)
        frame_idx = (idx % (500 // self.downsample_ratio)) * self.downsample_ratio

        # print(f'video idx:{video_idx} frame idx:{frame_idx}')

        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])

        # 获取当前帧及前后n帧
        frame_indices = range(max(0, frame_idx - self.n * self.downsample_ratio),
                            min(500, frame_idx + (self.n + 1) * self.downsample_ratio), self.downsample_ratio)

        # 处理边界情况(前后padding)
        if len(frame_indices) < self.bundle_frame:
            if frame_idx < self.n * self.downsample_ratio:  # 开头padding
                pad_count = self.n - frame_idx // self.downsample_ratio
                frame_indices = [0] * pad_count + list(frame_indices)
            else:  # 结尾padding
                pad_count = (frame_idx + self.n * self.downsample_ratio + self.downsample_ratio) // self.downsample_ratio - 500 // self.downsample_ratio
                frame_indices = list(frame_indices) + [499] * pad_count


        # 获取当前视频片段的共享参数
        params = self.video_params[video_idx]
        rgb2cam = params['rgb2cam']
        cam2rgb = params['cam2rgb']
        rgb_gain = params['rgb_gain']
        red_gain = params['red_gain']
        blue_gain = params['blue_gain']
        shot_noise = params['shot_noise']
        read_noise = params['read_noise']

        metadata = {
            'rgb2cam': rgb2cam,
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
            'shot_noise': shot_noise,
            'read_noise': read_noise,
        }

        # 读取所有帧并处理
        frames = []
        noise_frames = []
        rgb_frame = None

        top, left = None, None

        current_frame = frame_indices[self.n]

        for i in frame_indices:
            img_path = os.path.join(video_dir, f"{i:08d}.png")
            image = Image.open(img_path).convert('RGB')

            # 随机裁剪
            w, h = image.size
            if top is None or left is None:
                top = torch.randint(0, h - self.patch_size, (1,)).item()
                left = torch.randint(0, w - self.patch_size, (1,)).item()

            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            image = np.array(image)


            if self.get_srgb:
                if i == current_frame:
                    rgb_frame=(image)

            # 转换为tensor并处理
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)

            unprocessed_image, noise_image, linear_RGB, _ = self.unprocessor.forward(image_tensor, add_noise=True,
                                                                                shot_noise=shot_noise, read_noise=read_noise,
                                                                                rgb2cam=rgb2cam, rgb_gain=rgb_gain,
                                                                                red_gain=red_gain, blue_gain=blue_gain)

            if self.trans12bit:
                noise_image = noise_image * (4095 - 240) + 240 #240为黑电平

            if i == current_frame:
                linear_RGB_frame = linear_RGB.permute(2, 0, 1)
                if self.trans12bit:
                    linear_RGB_frame = linear_RGB_frame * (4095 - 240) + 240 #240为黑电平


            # frames.append(unprocessed_image)
            noise_frames.append(noise_image)

        # 转换为tensor
        # frames = torch.stack(frames)
        noise_frames = torch.stack(noise_frames)

        #permute
        # frames = frames.permute(0, 3, 1, 2)
        noise_frames = noise_frames.permute(0, 3, 1, 2)

        if self.get_srgb:
            rgb_frame = torch.from_numpy(rgb_frame.astype(np.float32) / 255.0)
            rgb_frame = rgb_frame.permute(2, 0, 1)

        if self.transform:
            frames = self.transform(frames)
            noise_frames = self.transform(noise_frames)
            linear_RGB_frame = self.transform(linear_RGB_frame)
            if self.get_srgb:
                rgb_frame = self.transform(rgb_frame)



        data = {
            # 'raw_clean': frames,
            'raw_noise': noise_frames,
            'linear_RGB': linear_RGB_frame,
            'metadata': metadata,
            'frame_idx': {'video': video_idx, 'frame': frame_idx},
        }
        if rgb_frame is not None:
            data['srgb'] = rgb_frame.cpu()

        return data

class REDS_Val_Dataset(Dataset):
    def __init__(self, root_dir, bundle_frame=5, transform=None, downsample_ratio=1, args=None, seed=42):
        """
        初始化数据集
        Args:
            root_dir (str): 包含视频片段的根目录
            bundle_frame (int): 每个batch包含的帧数(奇数)
            transform (callable, optional): 可选的图像变换
            downsample_ratio (int): 帧率降采样比率，默认为1（不进行降采样）
            seed (int): 随机种子，默认为42
        """
        assert bundle_frame % 2 == 1, 'Bundle_frame must be Odd number'
        self.root_dir = root_dir
        self.transform = transform
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2  # 计算前后帧偏移量
        self.downsample_ratio = downsample_ratio


        self.unprocessor = ImageUnprocessor()

        self.get_srgb = False

        if args is not None:
            self.trans12bit = args.trans12bit
        else:
            self.trans12bit = False

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 获取所有视频片段目录
        self.video_dirs = [d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))]

        # 为每个视频片段生成共享参数
        self.video_params = []
        for _ in self.video_dirs:
            rgb2cam = self.unprocessor.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            # rgb2cam = unprocessor.rgb2cam
            # cam2rgb = unprocessor.cam2rgb

            rgb_gain, red_gain, blue_gain = self.unprocessor.random_gains()
            shot_noise, read_noise = self.unprocessor.random_noise_levels()
            self.video_params.append({
                'rgb2cam': rgb2cam,
                'cam2rgb': cam2rgb,
                'rgb_gain': rgb_gain,
                'red_gain': red_gain,
                'blue_gain': blue_gain,
                'shot_noise': shot_noise,
                'read_noise': read_noise
            })

    def __len__(self):
        return len(self.video_dirs) * (500 // self.downsample_ratio)

    def __getitem__(self, idx):
        video_idx = idx // (500 // self.downsample_ratio)
        frame_idx = (idx % (500 // self.downsample_ratio)) * self.downsample_ratio

        # print(f'video idx:{video_idx} frame idx:{frame_idx}')

        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])

        # 获取当前帧及前后n帧
        frame_indices = range(max(0, frame_idx - self.n * self.downsample_ratio),
                            min(500, frame_idx + (self.n + 1) * self.downsample_ratio), self.downsample_ratio)

        # 处理边界情况(前后padding)
        if len(frame_indices) < self.bundle_frame:
            if frame_idx < self.n * self.downsample_ratio:  # 开头padding
                pad_count = self.n - frame_idx // self.downsample_ratio
                frame_indices = [0] * pad_count + list(frame_indices)
            else:  # 结尾padding
                pad_count = (frame_idx + self.n * self.downsample_ratio + self.downsample_ratio) // self.downsample_ratio - 500 // self.downsample_ratio
                frame_indices = list(frame_indices) + [499] * pad_count

        # 获取当前视频片段的共享参数
        params = self.video_params[video_idx]
        rgb2cam = params['rgb2cam']
        cam2rgb = params['cam2rgb']
        rgb_gain = params['rgb_gain']
        red_gain = params['red_gain']
        blue_gain = params['blue_gain']
        shot_noise = params['shot_noise']
        read_noise = params['read_noise']

        metadata = {
            'rgb2cam': rgb2cam.cpu(),
            'cam2rgb': cam2rgb.cpu(),
            'rgb_gain': rgb_gain.cpu(),
            'red_gain': red_gain.cpu(),
            'blue_gain': blue_gain.cpu(),
            'shot_noise': shot_noise.cpu(),
            'read_noise': read_noise.cpu(),
        }

        # 读取所有帧并处理
        frames = []
        noise_frames = []
        rgb_frame = None

        current_frame = frame_indices[self.n]

        for i in frame_indices:
            img_path = os.path.join(video_dir, f"{i:08d}.png")
            image = Image.open(img_path).convert('RGB')

            image = np.array(image)


            if self.get_srgb:
                if i == current_frame:
                    rgb_frame=(image)

            # 转换为tensor并处理
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)

            unprocessed_image, noise_image, linear_RGB, _ = self.unprocessor.forward(image_tensor, add_noise=True,
                                                                                shot_noise=shot_noise, read_noise=read_noise,
                                                                                rgb2cam=rgb2cam, rgb_gain=rgb_gain,
                                                                                red_gain=red_gain, blue_gain=blue_gain)
            if self.trans12bit:
                noise_image = noise_image * (4095 - 240) + 240 #240为黑电平

            if i == current_frame:
                linear_RGB_frame = linear_RGB.permute(2, 0, 1)
                if self.trans12bit:
                    linear_RGB_frame = linear_RGB_frame * (4095 - 240) + 240 #240为黑电平

            # frames.append(unprocessed_image)
            noise_frames.append(noise_image)

        # 转换为tensor
        # frames = torch.stack(frames)
        noise_frames = torch.stack(noise_frames)


        #permute
        # frames = frames.permute(0, 3, 1, 2)
        noise_frames = noise_frames.permute(0, 3, 1, 2)

        if self.get_srgb:
            rgb_frame = torch.from_numpy(rgb_frame.astype(np.float32) / 255.0)
            rgb_frame = rgb_frame.permute(2, 0, 1)

        if self.transform:
            frames = self.transform(frames)
            noise_frames = self.transform(noise_frames)
            linear_RGB_frame = self.transform(linear_RGB_frame)
            if self.get_srgb:
                rgb_frame = self.transform(rgb_frame)

        data = {
            # 'raw_clean': frames.cpu(),
            'raw_noise': noise_frames.cpu(),
            'linear_RGB': linear_RGB_frame.cpu(),
            'metadata': metadata,
            'frame_idx': {'video': video_idx, 'frame': frame_idx},
        }
        if rgb_frame is not None:
            data['srgb'] = rgb_frame.cpu()

        return data


class TVD(Dataset):
    def __init__(self, root_dir, bundle_frame=5, patch_size=256, transform=None, args=None):
        """
        初始化TVD数据集
        Args:
            root_dir (str): 包含视频片段的根目录，格式为TVD/{video_name}/{%08d.png}
            bundle_frame (int): 每个batch包含的帧数(奇数)
            patch_size (int): 图像裁剪的大小
            transform (callable, optional): 可选的图像变换
        """
        assert bundle_frame % 2 == 1, 'Bundle_frame must be Odd number'
        self.root_dir = root_dir
        self.transform = transform
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2  # 计算前后帧偏移量
        self.patch_size = patch_size

        self.unprocessor = ImageUnprocessor()

        self.get_srgb = False
        if args is not None:
            self.trans12bit = args.trans12bit
        else:
            self.trans12bit = False

        # 获取所有视频片段目录
        self.video_dirs = [d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))]

        self.video_params = []
        self.video_frame_counts = []

        for video_dir in self.video_dirs:
            # 获取每个视频的帧数
            frames_dir = os.path.join(self.root_dir, video_dir)
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
            self.video_frame_counts.append(len(frame_files))
            
            rgb2cam = self.unprocessor.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            rgb_gain, red_gain, blue_gain = self.unprocessor.random_gains()
            shot_noise, read_noise = self.unprocessor.random_noise_levels()
            self.video_params.append({
                'rgb2cam': rgb2cam.cpu(),
                'cam2rgb': cam2rgb.cpu(),
                'rgb_gain': rgb_gain.cpu(),
                'red_gain': red_gain.cpu(),
                'blue_gain': blue_gain.cpu(),
                'shot_noise': shot_noise.cpu(),
                'read_noise': read_noise.cpu()
            })

        print(f"TVD dataset loaded with {len(self.video_dirs)} videos")

    def __len__(self):
        # 返回所有视频帧数之和
        return sum(self.video_frame_counts)

    def __getitem__(self, idx):
        # 确定是哪个视频以及该视频中的哪一帧
        video_idx = 0
        frame_idx = idx
        for i, frame_count in enumerate(self.video_frame_counts):
            if frame_idx < frame_count:
                video_idx = i
                break
            frame_idx -= frame_count

        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])
        total_frames = self.video_frame_counts[video_idx]

        # 获取当前帧及前后n帧
        frame_indices = list(range(max(0, frame_idx - self.n),
                                   min(total_frames, frame_idx + self.n + 1)))

        # 处理边界情况(前后padding)
        if len(frame_indices) < self.bundle_frame:
            if frame_idx < self.n:  # 开头padding
                pad_count = self.n - frame_idx
                frame_indices = [0] * pad_count + list(frame_indices)
            else:  # 结尾padding
                pad_count = (frame_idx + self.n + 1) - total_frames
                frame_indices = list(frame_indices) + [total_frames - 1] * pad_count

        # 获取当前视频片段的共享参数
        params = self.video_params[video_idx]
        rgb2cam = params['rgb2cam']
        cam2rgb = params['cam2rgb']
        rgb_gain = params['rgb_gain']
        red_gain = params['red_gain']
        blue_gain = params['blue_gain']
        shot_noise = params['shot_noise']
        read_noise = params['read_noise']

        metadata = {
            'rgb2cam': rgb2cam,
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
            'shot_noise': shot_noise,
            'read_noise': read_noise,
        }

        # 读取所有帧并处理
        noise_frames = []
        rgb_frame = None
        top, left = None, None
        current_frame = frame_indices[self.n]

        for i in frame_indices:
            img_path = os.path.join(video_dir, f"{i:08d}.png")
            image = Image.open(img_path).convert('RGB')

            # 随机裁剪
            w, h = image.size
            if top is None or left is None:
                top = torch.randint(0, h - self.patch_size, (1,)).item()
                left = torch.randint(0, w - self.patch_size, (1,)).item()

            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            image = np.array(image)

            if self.get_srgb:
                if i == current_frame:
                    rgb_frame = image

            # 转换为tensor并处理
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)

            unprocessed_image, noise_image, linear_RGB, _ = self.unprocessor.forward(image_tensor, add_noise=True,
                                                                                shot_noise=shot_noise, read_noise=read_noise,
                                                                                rgb2cam=rgb2cam, rgb_gain=rgb_gain,
                                                                                red_gain=red_gain, blue_gain=blue_gain)

            if self.trans12bit:
                noise_image = noise_image * (4095 - 240) + 240 #240为黑电平

            if i == current_frame:
                linear_RGB_frame = linear_RGB.permute(2, 0, 1)
                if self.trans12bit:
                    linear_RGB_frame = linear_RGB_frame * (4095 - 240) + 240 #240为黑电平

            noise_frames.append(noise_image)

        # 转换为tensor
        noise_frames = torch.stack(noise_frames)

        #permute
        noise_frames = noise_frames.permute(0, 3, 1, 2)

        if self.get_srgb:
            rgb_frame = torch.from_numpy(rgb_frame.astype(np.float32) / 255.0)
            rgb_frame = rgb_frame.permute(2, 0, 1)

        if self.transform:
            noise_frames = self.transform(noise_frames)
            linear_RGB_frame = self.transform(linear_RGB_frame)
            if self.get_srgb:
                rgb_frame = self.transform(rgb_frame)

        data = {
            'raw_noise': noise_frames,
            'linear_RGB': linear_RGB_frame,
            'metadata': metadata,
            'frame_idx': {'video': video_idx, 'frame': frame_idx},
        }
        
        if rgb_frame is not None:
            data['srgb'] = rgb_frame.cpu()
            
        return data

# 添加VideoMME数据集类，仿照TVD实现
class VideoMME(Dataset):
    def __init__(self, root_dir, bundle_frame=5, patch_size=256, transform=None, args=None):
        """
        初始化VideoMME数据集
        Args:
            root_dir (str): 包含视频片段的根目录，格式为VideoMME/{video_name}/{%08d.png}
            bundle_frame (int): 每个batch包含的帧数(奇数)
            patch_size (int): 图像裁剪的大小
            transform (callable, optional): 可选的图像变换
            args: 其他参数
        """
        assert bundle_frame % 2 == 1, 'Bundle_frame must be Odd number'
        self.root_dir = root_dir
        self.transform = transform
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2  # 计算前后帧偏移量
        self.patch_size = patch_size

        self.unprocessor = ImageUnprocessor()

        self.get_srgb = False
        if args is not None:
            self.trans12bit = args.trans12bit
        else:
            self.trans12bit = False

        # 获取所有视频片段目录
        self.video_dirs = [d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))]

        self.video_params = []
        self.video_frame_counts = []

        for video_dir in self.video_dirs:
            # 获取每个视频的帧数
            frames_dir = os.path.join(self.root_dir, video_dir)
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
            self.video_frame_counts.append(len(frame_files))
            
            rgb2cam = self.unprocessor.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            rgb_gain, red_gain, blue_gain = self.unprocessor.random_gains()
            shot_noise, read_noise = self.unprocessor.random_noise_levels()
            self.video_params.append({
                'rgb2cam': rgb2cam.cpu(),
                'cam2rgb': cam2rgb.cpu(),
                'rgb_gain': rgb_gain.cpu(),
                'red_gain': red_gain.cpu(),
                'blue_gain': blue_gain.cpu(),
                'shot_noise': shot_noise.cpu(),
                'read_noise': read_noise.cpu()
            })

        print(f"VideoMME dataset loaded with {len(self.video_dirs)} videos")

    def __len__(self):
        # 返回所有视频帧数之和
        return sum(self.video_frame_counts)

    def __getitem__(self, idx):
        # 确定是哪个视频以及该视频中的哪一帧
        video_idx = 0
        frame_idx = idx
        for i, frame_count in enumerate(self.video_frame_counts):
            if frame_idx < frame_count:
                video_idx = i
                break
            frame_idx -= frame_count

        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])
        total_frames = self.video_frame_counts[video_idx]

        # 获取当前帧及前后n帧
        frame_indices = list(range(max(0, frame_idx - self.n),
                                   min(total_frames, frame_idx + self.n + 1)))

        # 处理边界情况(前后padding)
        if len(frame_indices) < self.bundle_frame:
            if frame_idx < self.n:  # 开头padding
                pad_count = self.n - frame_idx
                frame_indices = [0] * pad_count + list(frame_indices)
            else:  # 结尾padding
                pad_count = (frame_idx + self.n + 1) - total_frames
                frame_indices = list(frame_indices) + [total_frames - 1] * pad_count

        # 获取当前视频片段的共享参数
        params = self.video_params[video_idx]
        rgb2cam = params['rgb2cam']
        cam2rgb = params['cam2rgb']
        rgb_gain = params['rgb_gain']
        red_gain = params['red_gain']
        blue_gain = params['blue_gain']
        shot_noise = params['shot_noise']
        read_noise = params['read_noise']

        metadata = {
            'rgb2cam': rgb2cam,
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
            'shot_noise': shot_noise,
            'read_noise': read_noise,
        }

        # 读取所有帧并处理
        noise_frames = []
        rgb_frame = None
        top, left = None, None
        current_frame = frame_indices[self.n]

        for i in frame_indices:
            img_path = os.path.join(video_dir, f"{i:08d}.png")
            image = Image.open(img_path).convert('RGB')

            # 随机裁剪
            w, h = image.size
            if top is None or left is None:
                top = torch.randint(0, h - self.patch_size, (1,)).item()
                left = torch.randint(0, w - self.patch_size, (1,)).item()

            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            image = np.array(image)

            if self.get_srgb:
                if i == current_frame:
                    rgb_frame = image

            # 转换为tensor并处理
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)

            unprocessed_image, noise_image, linear_RGB, _ = self.unprocessor.forward(image_tensor, add_noise=True,
                                                                                shot_noise=shot_noise, read_noise=read_noise,
                                                                                rgb2cam=rgb2cam, rgb_gain=rgb_gain,
                                                                                red_gain=red_gain, blue_gain=blue_gain)

            if self.trans12bit:
                noise_image = noise_image * (4095 - 240) + 240 #240为黑电平

            if i == current_frame:
                linear_RGB_frame = linear_RGB.permute(2, 0, 1)
                if self.trans12bit:
                    linear_RGB_frame = linear_RGB_frame * (4095 - 240) + 240 #240为黑电平

            noise_frames.append(noise_image)

        # 转换为tensor
        noise_frames = torch.stack(noise_frames)

        #permute
        noise_frames = noise_frames.permute(0, 3, 1, 2)

        if self.get_srgb:
            rgb_frame = torch.from_numpy(rgb_frame.astype(np.float32) / 255.0)
            rgb_frame = rgb_frame.permute(2, 0, 1)

        if self.transform:
            noise_frames = self.transform(noise_frames)
            linear_RGB_frame = self.transform(linear_RGB_frame)
            if self.get_srgb:
                rgb_frame = self.transform(rgb_frame)

        data = {
            'raw_noise': noise_frames,
            'linear_RGB': linear_RGB_frame,
            'metadata': metadata,
            'frame_idx': {'video': video_idx, 'frame': frame_idx},
        }
        
        if rgb_frame is not None:
            data['srgb'] = rgb_frame.cpu()
            
        return data

class CombinedDataset(Dataset):
    def __init__(self, datasets):
        """
        初始化融合数据集
        Args:
            datasets (list): 数据集列表
        """
        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        
    def __len__(self):
        return sum(self.dataset_lengths)
    
    def __getitem__(self, idx):
        # 确定是哪个数据集
        dataset_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i+1]:
                dataset_idx = i
                break
        
        # 获取在该数据集中的索引
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

if __name__ == '__main__':

    seed_everything(42)
    # dataset = REDS_Train_Dataset(root_dir='H:/datasets/JDD/REDS120/train/train_orig',
    #                                   bundle_frame=5, patch_size=256, downsample_ratio=2)

    red_dataset = REDS_Train_Dataset(root_dir='H:/datasets/JDD/REDS120/train/train_orig', bundle_frame=5, patch_size=256)
    tvd_dataset = TVD(root_dir=r'F:\datasets\Tencent_Video_Dataset\Video\frames', bundle_frame=5, patch_size=256)
    red_dataset.get_srgb = True
    tvd_dataset.get_srgb = True


    # 使用CombinedDataset合并数据集
    dataset = CombinedDataset([ tvd_dataset])

    dataset.get_srgb = True
    # isp = BayerImageProcessor()
    # print(len(dataset))
    # idx = 2
    # print(dataset[idx]['raw_clean'].shape)
    # raw = dataset[idx]['raw_clean'].cpu()[0].unsqueeze(0)
    # processed_img = isp.process(raw, dataset[idx]['metadata']['red_gain'].cpu().unsqueeze(0),
    #                                       dataset[idx]['metadata']['blue_gain'].cpu().unsqueeze(0),
    #                                       dataset[idx]['metadata']['cam2rgb'].cpu().unsqueeze(0))
    # print(processed_img.shape)
    # #Noise
    # print(dataset[idx]['raw_noise'].shape)
    # raw = dataset[idx]['raw_noise'].cpu()[0].unsqueeze(0)
    # noise_img = isp.process(raw, dataset[idx]['metadata']['red_gain'].cpu().unsqueeze(0),
    #                             dataset[idx]['metadata']['blue_gain'].cpu().unsqueeze(0),
    #                             dataset[idx]['metadata']['cam2rgb'].cpu().unsqueeze(0))
    # print(noise_img.shape)
    #
    # cv2.imwrite('isp.png', cv2.cvtColor(np.uint8(processed_img[0] * 255.0), cv2.COLOR_RGB2BGR))
    # cv2.imwrite('isp_noise.png', cv2.cvtColor(np.uint8(noise_img[0] * 255.0), cv2.COLOR_RGB2BGR))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    batch = next(iter(train_loader))
    print(batch['raw_noise'].shape)
    print(batch['linear_RGB'].shape)
    # print(batch['srgb'].shape)
    print(batch['metadata'].keys())
    for key in batch['metadata'].keys():
        print(key, batch['metadata'][key].shape)

    isp = IspProcessor()
    linear2srgbs = isp.process(batch['linear_RGB'], batch['metadata']['red_gain'],
                                       batch['metadata']['blue_gain'],
                                       batch['metadata']['cam2rgb'],
                                        batch['metadata']['rgb_gain'], dem=False)

    print(linear2srgbs.shape)
    plt.imshow(linear2srgbs[0].cpu())
    plt.show()
    cv2.imwrite('linear2srgbs.png', cv2.cvtColor(np.uint8(linear2srgbs[0].cpu() * 255.0), cv2.COLOR_RGB2BGR))
    plt.imshow(batch['srgb'][0].cpu().permute(1, 2, 0))
    plt.show()
    cv2.imwrite('srgb.png', cv2.cvtColor(np.uint8(batch['srgb'][0].cpu().permute(1, 2, 0) * 255.0), cv2.COLOR_RGB2BGR))

