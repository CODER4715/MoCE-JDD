import os
import cv2
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 硬编码输出文件夹路径，可修改此变量来指定不同的输出位置
OUTPUT_FOLDER_PATH = Path(r"F:\datasets\video-mme")
# 可配置的截取时长（秒），可修改此变量来指定不同的截取时长
CLIP_DURATION = 5


def process_single_video(file_path, output_folder, seed=42):
    """处理单个视频文件，提取帧并保存为图片"""
    # 设置随机种子以确保确定性
    random.seed(seed)

    # 获取视频文件名（不含扩展名）
    video_name = file_path.stem

    # 创建以视频命名的文件夹
    output_video_folder = output_folder / video_name
    output_video_folder.mkdir(exist_ok=True, parents=True)

    # 打开视频文件
    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        print(f"无法打开视频文件: {file_path}")
        return False

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # 计算指定秒数需要的帧数
    target_frames = int(CLIP_DURATION * fps)

    # 如果视频时长小于等于指定秒数，提取所有帧
    if duration <= CLIP_DURATION:
        start_frame = 0
        end_frame = total_frames
    else:
        # 随机选择中间的指定秒数起始点
        max_start_frame = total_frames - target_frames
        start_frame = random.randint(0, max_start_frame)
        end_frame = start_frame + target_frames

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 读取并保存指定范围内的帧
    frame_count = 0
    frames_written = 0
    success = True

    while success and frames_written < (end_frame - start_frame):
        success, frame = cap.read()
        if success:
            # 生成帧图片文件名
            frame_filename = output_video_folder / f"{frame_count:08d}.png"

            # 保存帧为图片
            cv2.imwrite(str(frame_filename), frame)
            frame_count += 1
            frames_written += 1

    # 释放资源
    cap.release()

    return True


def process_video_mme_dataset(dataset_path):
    # 定义视频文件夹路径
    video_folder = Path(dataset_path)

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

    # 创建输出文件夹（如果不存在）
    output_folder = Path(OUTPUT_FOLDER_PATH)
    output_folder.mkdir(exist_ok=True, parents=True)

    # 使用线程池并行处理视频
    with ThreadPoolExecutor(max_workers=12) as executor:
        # 提交所有任务
        future_to_video = {executor.submit(process_single_video, video_file, output_folder): video_file
                           for video_file in video_files}

        # 使用tqdm显示进度条
        with tqdm(total=len(video_files), desc="处理视频") as pbar:
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                try:
                    success = future.result()
                    if success:
                        pbar.set_postfix_str(f"{video_file.name}: 处理完成")
                    else:
                        pbar.set_postfix_str(f"{video_file.name}: 处理失败")
                    pbar.update(1)
                except Exception as exc:
                    print(f'{video_file} 处理时发生异常: {exc}')
                    pbar.update(1)


if __name__ == "__main__":
    dataset_path = r'G:\datasets\JDD\Video-MME'
    process_video_mme_dataset(dataset_path)