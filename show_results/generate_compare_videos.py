import os
import re
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array
import concurrent.futures
from tqdm import tqdm  # 引入tqdm用于显示进度条


# --- 这是被每个子进程独立执行的函数 ---
def process_single_video(video_name, base_path, psnr_values_dict, output_dir, model_folders, model_labels):
    """
    处理单个视频文件：加载、添加标签、拼接成2x2网格并保存。
    这个函数是多进程任务的执行单元。

    返回:
        str: 描述处理结果的消息。
    """
    try:
        print(f"开始处理视频: {video_name} (在子进程中)")
        clips_for_grid = {}

        current_psnr_info = psnr_values_dict.get(video_name, {})

        for folder_name in model_folders:
            video_path = os.path.join(base_path, folder_name, video_name)
            if os.path.exists(video_path):
                clip = VideoFileClip(video_path)

                # 根据视频分辨率计算字体大小
                font_size = max(20, int(min(clip.size) * 0.04))

                # 模型名称标签
                text_clip_model = (TextClip(model_labels[folder_name], fontsize=font_size, color='white', bg_color='black')
                                   .set_position(("left", "top"))
                                   .set_duration(clip.duration)
                                   .set_opacity(0.8))

                final_clip_elements = [clip, text_clip_model]

                # PSNR标签 (GT不需要)
                if folder_name != "gt":
                    psnr_value = current_psnr_info.get(folder_name)
                    if psnr_value is not None:
                        text_clip_psnr = (
                            TextClip(f"PSNR: {psnr_value:.2f}", fontsize=font_size, color='yellow', bg_color='black')
                            .set_position(("right", "top"))
                            .set_duration(clip.duration)
                            .set_opacity(0.8))
                        final_clip_elements.append(text_clip_psnr)
                    else:
                        print(f"  - 警告：视频 '{video_name}' 的 '{folder_name}' 模型未找到PSNR值。")

                final_clip = CompositeVideoClip(final_clip_elements)
                clips_for_grid[folder_name] = final_clip
            else:
                clips_for_grid[folder_name] = None
                return f"处理失败: {video_name} - 找不到 '{folder_name}' 文件夹中的视频。"

        # 检查是否所有必要的视频都已加载
        if all(clips_for_grid.get(f) is not None for f in model_folders):
            # 按照指定的顺序排列视频，以形成 2x2 网格
            grid_clips = [
                [clips_for_grid["gt"], clips_for_grid["input"]],
                [clips_for_grid["FastDVDnet-Mini"], clips_for_grid["MoCE-JDD"]]
            ]

            final_video = clips_array(grid_clips)
            output_filepath = os.path.join(output_dir, f"stitched_{video_name}")

            # 写入视频文件。可以设置logger='bar'来禁用moviepy自己的进度条，避免与tqdm冲突
            final_video.write_videofile(output_filepath, codec="libx264", audio_codec="aac",
                                        fps=clips_for_grid["gt"].fps, logger=None)

            # 释放资源
            final_video.close()
            for clip in clips_for_grid.values():
                if clip:
                    clip.close()

            return f"处理完成: {video_name}"
        else:
            return f"处理跳过: {video_name} - 未能找到所有必需的视频文件。"

    except Exception as e:
        return f"处理出错: {video_name} - 错误: {e}"


def stitch_videos_multiprocess(base_path, psnr_values_dict, output_dir="H:\\My-JDD\\stitched_videos"):
    """
    使用多进程并行拼接指定文件夹中的同名视频，并根据预定义字典添加标签。
    以 2x2 网格形式同时展示四个视频。
    """
    model_folders = ["gt", "input", "FastDVDnet-Mini", "MoCE-JDD"]
    model_labels = {
        "gt": "GT",
        "input": "Noisy",
        "FastDVDnet-Mini": "FastDVDnet-Mini",
        "MoCE-JDD": "MoCE-JDD"
    }

    # 1. 查找所有需要处理的视频
    all_video_names = []
    # 以gt文件夹为基准，因为它必须存在
    gt_folder_path = os.path.join(base_path, "gt")
    if os.path.exists(gt_folder_path):
        all_video_names = sorted([f for f in os.listdir(gt_folder_path) if f.endswith('.mp4')])

    if not all_video_names:
        print(f"在 'gt' 文件夹中没有找到视频文件，无法继续。")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"找到 {len(all_video_names)} 个视频待处理。将在 '{base_path}' 中查找并处理...")

    # 2. 创建进程池并提交任务
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        # 准备提交给每个进程的任务和参数
        futures = [
            executor.submit(process_single_video, video_name, base_path, psnr_values_dict, output_dir, model_folders,
                            model_labels)
            for video_name in all_video_names
        ]

        # 3. 处理任务结果并显示进度条
        # tqdm会根据任务完成情况自动更新进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_video_names)):
            result_message = future.result()
            # 可以选择性地打印每个任务的结果，或者只在出错时打印
            if "失败" in result_message or "出错" in result_message:
                print(result_message)

    print("\n所有视频处理任务已完成！")


def read_psnr_from_files(base_video_path):
    """
    从 input、FastDVDnet-Mini、MoCE-JDD 文件夹下的 result.txt 文件中读取 sRGB PSNR 值。
    (此函数保持不变)
    """
    model_folders = ["input", "FastDVDnet-Mini", "MoCE-JDD"]
    psnr_data = {}

    for folder in model_folders:
        file_path = os.path.join(base_video_path, folder, "results.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.startswith('Video'):
                        continue
                    parts = line.split(':')
                    video_name = parts[0].split(' ')[1] + '.mp4'
                    match = re.search(r'sRGB PSNR:\s*([\d\.]+)', line)
                    if match:
                        srgb_psnr = float(match.group(1))
                        if video_name not in psnr_data:
                            psnr_data[video_name] = {}
                        psnr_data[video_name][folder] = srgb_psnr
                    else:
                        print(f"警告：未能从行 '{line.strip()}' 中提取 sRGB PSNR 值。")
        else:
            print(f"警告：未找到 {file_path} 文件件。")
    return psnr_data


if __name__ == "__main__":
    base_video_path = r"H:\My-JDD\result_tvd\videos_result"
    output_path = r"H:\My-JDD\result_tvd\stitched_videos"

    print("正在从文件中读取PSNR数据...")
    psnr_data = read_psnr_from_files(base_video_path)

    print("开始多进程处理视频...")
    stitch_videos_multiprocess(base_video_path, psnr_data, output_dir=output_path)