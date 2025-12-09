import os

import cv2
import torch
from torch.utils.data import Dataset
import skimage
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


from data.unprocessor import ImageUnprocessor, IspProcessor
from data.Hamilton_Adam_demo import HamiltonAdam



def raw_to_bayer(raw_image):
    black_level = 240
    white_level = 2**12-1
    raw_image = np.maximum(raw_image - black_level, 0) / (white_level-black_level)

    H, W = raw_image.shape
    # 确保尺寸是偶数
    H = H // 2 * 2
    W = W // 2 * 2
    raw_image = raw_image[:H, :W]

    chan0 = raw_image[0::2, 0::2]  # g
    chan1 = raw_image[0::2, 1::2]  # b
    chan2 = raw_image[1::2, 0::2]  # r
    chan3 = raw_image[1::2, 1::2]  # g

    # 堆叠成4通道图像 (4, H/2, W/2)
    # Pytorch的 Tensor 格式通常是 (C, H, W)
    rggb_image = np.stack([chan0, chan1, chan2, chan3], axis=0)
    rggb_image = torch.tensor(rggb_image, dtype=torch.float32)

    return rggb_image

def pack_rgbg_raw(raw):
    #pack GBRG Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 1:W:2, :],  # Red at (0,1)
                          im[1:H:2, 1:W:2, :],  # Green_r at (1,1)
                          im[1:H:2, 0:W:2, :],  # Blue at (1,0)
                          im[0:H:2, 0:W:2, :]), axis=2)  # Green_b at (0,0)
    out = torch.tensor(out, dtype=torch.float32).permute(2,0,1)
    return out

def pack_rggb_raw(raw):

    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],  #
                          im[0:H:2, 1:W:2, :],  #
                          im[1:H:2, 0:W:2, :],  #
                          im[1:H:2, 1:W:2, :]), axis=2)  #
    out = torch.tensor(out, dtype=torch.float32).permute(2,0,1)
    return out

def srgb_to_rgbg(srgb_image):
    # 确保输入图像的尺寸是偶数，以便完美镶嵌
    h, w, _ = srgb_image.shape
    h, w = h // 2 * 2, w // 2 * 2
    srgb_image = srgb_image[:h, :w]

    # G R
    # B G
    R  = srgb_image[1::2, 0::2, 0]  # R 通道位于 sRGB 图像的 (0,1) 位置
    Gr = srgb_image[1::2, 1::2, 1]  # G 通道位于 sRGB 图像的 (1,1) 位置 (R行)
    B  = srgb_image[0::2, 1::2, 2]  # B 通道位于 sRGB 图像的 (1,0) 位置
    Gb = srgb_image[0::2, 0::2, 1]  # G 通道位于 sRGB 图像的 (0,0) 位置 (B行)

    # 按照 ISP 模型期望的 [R, Gr, B, Gb] 顺序堆叠通道
    packed_bayer = torch.stack([B, Gr, R, Gb], dim=0)

    return packed_bayer

class CRVDataset(Dataset):
    def __init__(self, root_dir, bundle_frame=5, args=None):
        """
        初始化数据集加载类。

        参数:
        root_dir (str): 数据集根目录的路径。
        bundle_frame (int): 每个batch包含的帧数(奇数)，默认为5。
        transform (callable, 可选): 应用于图像的转换操作。
        args (object, 可选): 包含额外参数的对象，例如 trans12bit。
        """
        assert bundle_frame % 2 == 1, 'bundle_frame 必须是奇数'
        self.root_dir = root_dir
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2  # 计算前后帧偏移量

        self.image_groups = []  # 存储帧序列分组信息

        self.inv_ccm = torch.tensor([[1.07955733, -0.40125771, 0.32170038], [-0.15390743, 1.35677921, -0.20287178],
                                [-0.00235972, -0.55155296, 1.55391268]], dtype=torch.float32).unsqueeze(0).cuda()

        # self.unprocessor = ImageUnprocessor()

        gt_dir = os.path.join(root_dir, 'indoor_raw_gt')
        noisy_base_dir = os.path.join(root_dir, 'indoor_raw_noisy')

        self.video_dirs = []

        for scene in os.listdir(gt_dir):
            scene_path_gt = os.path.join(gt_dir, scene)
            if not os.path.isdir(scene_path_gt):
                continue

            for iso_dir in os.listdir(scene_path_gt):
                iso_path_gt = os.path.join(scene_path_gt, iso_dir)
                if not os.path.isdir(iso_path_gt):
                    continue

                iso_path_noisy = os.path.join(noisy_base_dir, scene, iso_dir)
                if not os.path.isdir(iso_path_noisy):
                    # print(f"警告: 未找到对应的噪声数据目录 {iso_path_noisy}，已跳过。")
                    continue

                video_key = f"{scene}/{iso_dir}"
                self.video_dirs.append(video_key)

                clean_img_names = sorted([f for f in os.listdir(iso_path_gt) if f.endswith('.tiff')])

                # 为每个视频片段构建帧序列信息
                for current_filename in clean_img_names:

                    if not current_filename.endswith('_clean_and_slightly_denoised.tiff'):
                        continue

                    # 1. 获取文件名主干，例如 "frame1"
                    stem = current_filename.replace('_clean_and_slightly_denoised.tiff', '')

                    # 2. 检查格式是否正确并解析
                    if stem.startswith('frame'):
                        base_name = 'frame'  # 基础名称现在是固定的 "frame"
                        frame_number_str = stem[len(base_name):]  # 提取数字部分，例如 "1"

                        try:
                            start_frame_num = int(frame_number_str)
                        except ValueError:
                            # 如果 "frame" 后面不是纯数字，则跳过
                            continue

                        # 将解析后的信息存入
                        self.image_groups.append({
                            'video_key': video_key,
                            'base_name': base_name,
                            'start_frame_num': start_frame_num,
                            'gt_iso_path': iso_path_gt,
                            'noisy_iso_path': iso_path_noisy,
                            'scene': scene,
                            'iso_dir': iso_dir
                        })


    def __len__(self):
        """返回数据集中样本的数量。"""
        return len(self.image_groups)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        此版本修正了文件名构造、路径使用和帧填充逻辑。
        """
        group_info = self.image_groups[idx]
        base_name = group_info['base_name']  # e.g., 'frame'
        center_frame_num = group_info['start_frame_num']
        gt_iso_path = group_info['gt_iso_path']
        noisy_iso_path = group_info['noisy_iso_path']  # 直接使用__init__中准备好的路径

        clean_frames = []
        noisy_frames = []

        # 定义一个内部函数，用于加载和预处理单个帧，避免代码重复
        def _load_and_process_frame(frame_number):
            # 1. --- 核心修正：构建正确的文件名 ---
            clean_img_name = f"{base_name}{frame_number}_clean_and_slightly_denoised.tiff"
            noisy_img_name = f"{base_name}{frame_number}_noisy0.tiff"

            clean_img_path = os.path.join(gt_iso_path, clean_img_name)
            noisy_img_path = os.path.join(noisy_iso_path, noisy_img_name)

            # 检查文件是否存在，如果任一文件不存在，则返回None
            if not os.path.exists(clean_img_path) or not os.path.exists(noisy_img_path):
                return None, None

            # 读取图像
            # clean_image = skimage.io.imread(clean_img_path).astype(np.float32)
            # noisy_image = skimage.io.imread(noisy_img_path).astype(np.float32)

            clean_image = cv2.imread(clean_img_path, -1)
            noisy_image = cv2.imread(noisy_img_path, -1)

            # 转换为Bayer格式
            # clean_raw = raw_to_bayer(clean_image)
            # noisy_raw = raw_to_bayer(noisy_image)

            clean_raw = pack_rgbg_raw(clean_image)
            noisy_raw = pack_rggb_raw(noisy_image)


            return clean_raw, noisy_raw

        # 2. --- 核心修正：简化帧加载与填充逻辑 ---
        # 首先，加载中心帧。根据__init__的逻辑，中心帧保证存在。
        # 它将作为缺失帧的填充源。
        center_clean_raw, center_noisy_raw = _load_and_process_frame(center_frame_num)
        if center_clean_raw is None:
            # 这种情况理论上不应发生，但作为保险措施
            raise FileNotFoundError(f"中心帧 {center_frame_num} 未找到，请检查数据集！")

        # 循环获取整个帧束
        for i in range(self.bundle_frame):
            frame_num = center_frame_num + (i - self.n) * 2  # 计算需要的帧号，间隔为2

            clean_raw, noisy_raw = _load_and_process_frame(frame_num)

            # 如果加载失败 (说明是边界帧，文件不存在)，则使用中心帧进行填充
            if clean_raw is None:
                clean_frames.append(center_clean_raw.clone())  # 使用clone()确保tensor独立
                noisy_frames.append(center_noisy_raw.clone())
            else:
                clean_frames.append(clean_raw)
                noisy_frames.append(noisy_raw)

        # 转换为tensor
        clean_frames = torch.stack(clean_frames)
        noisy_frames = torch.stack(noisy_frames)


        data = {
            'clean': clean_frames,
            'noisy': noisy_frames,
            'video_key': group_info['video_key'],
            'center_frame_num': center_frame_num
        }

        return data


# 使用示例
if __name__ == "__main__":
    # 请确保将此路径替换为您的 CRVD 数据集所在的实际路径
    root_dir = r'G:\datasets\CRVD_dataset'

    # 检查路径是否存在
    if not os.path.exists(root_dir):
        print(f"Error: Dataset directory not found at {root_dir}")
        print("Please update the 'root_dir' variable to the correct path.")
    else:
        dataset = CRVDataset(root_dir, bundle_frame=5)
        print(f"数据集大小 (样本组数): {len(dataset)}")

        if len(dataset) > 0:
            # 获取第一个样本
            sample = dataset[0]
            clean_bundle = sample['clean']
            noisy_bundle = sample['noisy']
            print(f"干净图像bundle形状: {clean_bundle.shape}")  # (bundle_frame, 4, H/2, W/2)
            print(f"噪声图像bundle形状: {noisy_bundle.shape}")  # (bundle_frame, 4, H/2, W/2)

            dem = HamiltonAdam(pattern='gbrg')
            # isp = IspProcessor()

            # 取中间帧进行处理
            center_frame_bayer = clean_bundle[2].unsqueeze(0)  # 添加batch维度 (1, 4, H/2, W/2)

            # Demosaic: Bayer -> RGB
            demosaiced_image = dem(center_frame_bayer)  # (1, 3, H, W)

            # ISP
            # isp_image = isp.process(demosaiced_image, torch.tensor(2).unsqueeze(0),
            #                         torch.tensor(2).unsqueeze(0), dataset.inv_ccm,
            #                         torch.tensor(1.2).unsqueeze(0),dem=False)

            isp = torch.load('H:/My-JDD/CRVD_ISP/ISP_CNN.pth', weights_only=False).cuda()
            for k, v in isp.named_parameters():
                v.requires_grad = False
            isp_image = isp(center_frame_bayer.cuda()).permute(0,2,3,1)[0]

            remosaic = srgb_to_rgbg(isp_image).unsqueeze(0) # (1, 4, H, W)
            reisp = isp(remosaic).permute(0,2,3,1)[0]

            # isp_image = demosaiced_image.permute(0,2,3,1).clamp(0, 1)

            # 从Tensor转换为Numpy数组以便显示
            # 将 (1, 3, H, W) 转换为 (H, W, 3)
            img_to_show = isp_image.squeeze(0).cpu().numpy()
            reisp = reisp.squeeze(0).cpu().numpy()

            # 裁剪到[0, 1]范围并显示
            img_to_show = np.clip(img_to_show, 0, 1)
            reisp = np.clip(reisp, 0, 1)

            print(f"最终用于显示的图像形状: {img_to_show.shape}")

            plt.imshow(img_to_show)
            plt.show()

            plt.imshow(reisp)
            plt.show()

        else:
            print("数据集中没有找到任何有效的图像组，请检查数据集路径和文件结构。")