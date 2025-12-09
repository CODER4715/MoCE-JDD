import os
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_single_video(file_path, frames_folder):
    """处理单个视频文件，提取帧并保存为图片"""
    # 获取视频文件名（不含扩展名）
    video_name = file_path.stem
    
    # 创建以视频命名的文件夹
    output_folder = frames_folder / video_name
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(str(file_path))
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {file_path}")
        return 0
    
    frame_count = 0
    success = True
    
    # 逐帧读取视频并保存为图片
    while success:
        success, frame = cap.read()
        
        if success:
            # 生成帧图片文件名
            frame_filename = output_folder / f"{frame_count:08d}.png"
            
            # 保存帧为图片
            cv2.imwrite(str(frame_filename), frame)
            frame_count += 1
    
    # 释放视频捕获对象
    cap.release()
    
    return frame_count

def extract_frames_from_videos(tvd_path):
    # 定义视频文件夹路径
    video_folder = Path(tvd_path)
    
    # 检查视频文件夹是否存在
    if not video_folder.exists():
        print(f"视频文件夹 {video_folder} 不存在")
        return
    
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    # 获取所有视频文件
    video_files = [f for f in video_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        print("未找到任何视频文件")
        return
    
    # 创建frames文件夹（如果不存在）
    frames_folder = video_folder / "frames"
    frames_folder.mkdir(exist_ok=True, parents=True)
    
    # 使用线程池并行处理视频
    with ThreadPoolExecutor(max_workers=12) as executor:
        # 提交所有任务
        future_to_video = {executor.submit(process_single_video, video_file, frames_folder): video_file 
                          for video_file in video_files}
        
        # 使用tqdm显示进度条
        with tqdm(total=len(video_files), desc="处理视频") as pbar:
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                try:
                    frame_count = future.result()
                    pbar.set_postfix_str(f"{video_file.name}: {frame_count} 帧")
                    pbar.update(1)
                except Exception as exc:
                    print(f'{video_file} 处理时发生异常: {exc}')
                    pbar.update(1)

if __name__ == "__main__":
    tvd_path = r'F:\datasets\Tencent_Video_Dataset\Video'
    extract_frames_from_videos(tvd_path)