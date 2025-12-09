import os
import cv2
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from unprocessor import ImageUnprocessor
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

class REDSValDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.video_dirs.sort()
        self.frame_paths = []
        
        # 收集所有视频帧路径
        for video_dir in self.video_dirs:
            video_path = os.path.join(root_dir, video_dir)
            frames = [f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            frames.sort()
            for frame in frames:
                self.frame_paths.append((video_dir, frame))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        video_dir, frame_name = self.frame_paths[idx]
        frame_path = os.path.join(self.root_dir, video_dir, frame_name)
        
        # 读取图像并转换为RGB
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"无法读取图像: {frame_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)
        
        return {
            'image': img_tensor,
            'video_dir': video_dir,
            'frame_name': frame_name
        }

def save_tensor(tensor, save_path):
    """保存tensor为.npy文件"""
    np.save(save_path, tensor.cpu().numpy())

def save_metadata(metadata, save_path):
    """保存metadata为JSON文件"""
    # 将tensor转换为可序列化的类型
    serializable_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            serializable_metadata[key] = value.cpu().numpy().tolist()
        else:
            serializable_metadata[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metadata, f, indent=4)

def main():    
    parser = argparse.ArgumentParser(description='生成REDS验证集的噪声数据')
    parser.add_argument('--input_dir', type=str, default=r'H:\datasets\JDD\REDS120\val\val_orig',
                        help='原始图像目录')
    parser.add_argument('--output_dir', type=str, default=r'H:\datasets\JDD\REDS120\val\val_noisy',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='使用的设备')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    args = parser.parse_args()

    # 创建输出目录及三个主文件夹
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'raw_noise'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'linear_rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metadata'), exist_ok=True)

    # 初始化数据集和数据加载器
    dataset = REDSValDataset(args.input_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)

    # 创建ImageUnprocessor实例
    unprocessor = ImageUnprocessor(device=args.device)

    # 处理每个批次
    for batch in tqdm(dataloader, desc='处理图像'):
        images = batch['image'].to(args.device)
        video_dirs = batch['video_dir']
        frame_names = batch['frame_name']

        for i in range(images.shape[0]):
            # 获取单张图像数据
            image = images[i]
            video_dir = video_dirs[i]
            frame_name = frame_names[i]
            frame_id = os.path.splitext(frame_name)[0]

            # 为每种数据类型创建视频输出目录
            raw_noise_dir = os.path.join(args.output_dir, 'raw_noise', video_dir)
            linear_rgb_dir = os.path.join(args.output_dir, 'linear_rgb', video_dir)
            metadata_dir = os.path.join(args.output_dir, 'metadata', video_dir)
            
            os.makedirs(raw_noise_dir, exist_ok=True)
            os.makedirs(linear_rgb_dir, exist_ok=True)
            os.makedirs(metadata_dir, exist_ok=True)

            # 使用ImageUnprocessor处理图像
            try:
                raw_clean, raw_noise, linear_RGB, metadata = unprocessor.forward(image, add_noise=True)
            except Exception as e:
                print(f"处理图像 {video_dir}/{frame_name} 时出错: {str(e)}")
                continue

            # 保存raw_noise
            raw_noise_path = os.path.join(raw_noise_dir, f'{frame_id}.npy')
            save_tensor(raw_noise, raw_noise_path)

            # 保存linear_RGB
            linear_rgb_path = os.path.join(linear_rgb_dir, f'{frame_id}.npy')
            save_tensor(linear_RGB, linear_rgb_path)

            # 保存metadata
            metadata_path = os.path.join(metadata_dir, f'{frame_id}.json')
            save_metadata(metadata, metadata_path)

    print(f"处理完成，结果保存在: {args.output_dir}")

if __name__ == '__main__':
    main()