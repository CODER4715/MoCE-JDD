"""
计算输入噪声的PSNR
"""

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

def calculate_psnr_ssim(input_folder, gt_folder):
    video_results = {}
    # 获取视频文件夹列表
    # video_folders = os.listdir(input_folder)
    video_folders = sorted([f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))])
    # 使用 tqdm 包装视频文件夹列表，添加进度条
    for video_folder in tqdm(video_folders, desc="处理视频片段", unit="个"):
        video_input_path = os.path.join(input_folder, video_folder)
        video_gt_path = os.path.join(gt_folder, video_folder)

        if os.path.isdir(video_input_path) and os.path.isdir(video_gt_path):
            total_psnr = 0
            num_images = 0
            # 遍历当前视频文件夹下的所有图像文件
            for image_name in os.listdir(video_input_path):
                input_image_path = os.path.join(video_input_path, image_name)
                gt_image_path = os.path.join(video_gt_path, image_name)

                if os.path.isfile(input_image_path) and os.path.isfile(gt_image_path):
                    # 读取图像
                    input_image = cv2.imread(input_image_path)
                    gt_image = cv2.imread(gt_image_path)

                    # 转换为 RGB 格式
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

                    # 计算 PSNR
                    psnr_value = psnr(gt_image, input_image, data_range=255)

                    total_psnr += psnr_value
                    num_images += 1

            if num_images > 0:
                avg_psnr = total_psnr / num_images
                video_results[video_folder] = avg_psnr

    return video_results

if __name__ == "__main__":
    # input_folder = r"H:\My-JDD\results\MoCE_JDD\input"
    # gt_folder = r"H:\datasets\JDD\REDS120\val\val_orig"
    # video_results = calculate_psnr_ssim(input_folder, gt_folder)

    # input_folder = r"H:\My-JDD\result_tvd\MoCE_JDD\input" # input
    # input_folder = r"H:\My-JDD\result_tvd\MoCE_JDD\exp16_netdepcv_L1_fft" # ours
    input_folder = r"H:\My-JDD\result_tvd\MoCE_JDD\exp_fastDVDnetMini"  # fastdvd-mini
    gt_folder = r"F:\datasets\Tencent_Video_Dataset\part"
    video_results = calculate_psnr_ssim(input_folder, gt_folder)

    if video_results:
        with open('../utils/results.txt', 'w', encoding='utf-8') as f:
            for video_folder, avg_psnr in video_results.items():
                f.write(f"Video {video_folder}: sRGB PSNR: {avg_psnr:.4f}\n")
        print("结果已保存到 results.txt")
    else:
        print("未找到有效的图像对。")