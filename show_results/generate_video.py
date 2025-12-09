"""
读取推理得到的视频帧，输出mp4视频
"""
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def get_image_files(image_folder):
    """获取指定文件夹下所有以 .png 结尾的图片文件，并按文件名排序。"""
    return sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

def create_video_from_images(image_folder, output_folder, video_name, fps=30):
    """根据指定文件夹下的图片生成视频。"""
    image_files = get_image_files(image_folder)
    if not image_files:
        print(f"No images found in {image_folder}.")
        return

    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = os.path.join(output_folder, video_name + '.mp4')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    for image_file in tqdm(image_files, desc=f"Generating video {video_name}"):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved to {video_output_path}")

def create_video_from_gt(image_folder, output_folder, video_name, fps=30):
    """根据指定文件夹下的图片生成视频。"""
    image_files = get_image_files(image_folder)
    if not image_files:
        print(f"No images found in {image_folder}.")
        return

    # image_folder = os.path.join(r'H:\datasets\JDD\REDS120\val\val_orig', os.path.basename(image_folder))

    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = os.path.join(output_folder, video_name + '.mp4')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    for image_file in tqdm(image_files, desc=f"Generating video {video_name}"):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved to {video_output_path}")

def process_experiment(exp_folder, output_folder, fps=30):
    """处理单个实验文件夹，将每个视频文件夹的图片生成视频。"""
    os.makedirs(output_folder, exist_ok=True)
    # orig
    # video_folders = sorted([f for f in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, f)) and f.isdigit()])

    # new
    video_folders = sorted([f for f in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, f))])

    for video_folder in tqdm(video_folders, desc="Processing videos"):
        video_path = os.path.join(exp_folder, video_folder)
        create_video_from_images(video_path, output_folder, video_folder, fps)


def process_gt(exp_folder, output_folder, fps=30):
    os.makedirs(output_folder, exist_ok=True)
    video_folders = sorted([f for f in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, f))])

    for video_folder in tqdm(video_folders, desc="Processing videos"):
        video_path = os.path.join(exp_folder, video_folder)
        create_video_from_gt(video_path, output_folder, video_folder, fps)

def stitch_images(input_folder, output_folder, fps=30):
    """拼接图片并生成视频。"""
    # 创建临时文件夹
    os.makedirs('temp', exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有视频文件夹
    video_folders = sorted([f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()])


    for video_folder in tqdm(video_folders, desc="Processing videos"):
        # 存储拼接后的图片路径
        stitched_image_paths = []

        video_path = os.path.join(input_folder, video_folder)
        input_image_folder = os.path.join(input_folder, 'input', video_folder)
        result_image_folder = video_path

        # 获取所有图片文件
        input_image_files = sorted([f for f in os.listdir(input_image_folder) if f.endswith('.png')])
        result_image_files = sorted([f for f in os.listdir(result_image_folder) if f.endswith('.png')])

        for input_file, result_file in zip(input_image_files, result_image_files):
            input_image_path = os.path.join(input_image_folder, input_file)
            result_image_path = os.path.join(result_image_folder, result_file)

            # 读取图片
            input_image = cv2.imread(input_image_path)
            result_image = cv2.imread(result_image_path)

            # 拼接图片
            stitched_image = np.hstack((input_image, result_image))

            # 保存拼接后的图片，加入视频序号信息
            stitched_image_name = f"{video_folder}_{os.path.splitext(input_file)[0]}_stitched.png"
            stitched_image_path = os.path.join('temp', stitched_image_name)
            cv2.imwrite(stitched_image_path, stitched_image)
            stitched_image_paths.append(stitched_image_path)

        # 生成视频
        if stitched_image_paths:
            first_image = cv2.imread(stitched_image_paths[0])
            height, width, _ = first_image.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_output_path = os.path.join(output_folder, video_folder+'.mp4')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

            for image_path in tqdm(stitched_image_paths, desc="Generating video"):
                image = cv2.imread(image_path)
                video_writer.write(image)

            video_writer.release()
            print(f"Video saved to {video_output_path}")
        else:
            print("No images found to stitch.")

        #清空temp
        for file in os.listdir('temp'):
            file_path = os.path.join('temp', file)
            if os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    #将输入拼在一起对比，输出视频
    # exp_name = 'exp15_netdepcv_L1'
    # input_folder = 'results/MoCE_JDD/' + exp_name
    # output_folder = 'videos_result/' + exp_name
    # fps = 30
    
    # stitch_images(input_folder, output_folder, fps)

    ################################################################################
    #单独输出视频，根据exp_name选择
    # exp_name = 'exp_fastDVDnetMini'
    # exp_name = 'exp002_netdepcv_L1_fft_vgg_192'
    # input_folder = 'results/MoCE_JDD/' + exp_name
    # output_folder = 'videos_result/' + exp_name + '/clips/'
    # fps = 30
    # process_experiment(input_folder, output_folder, fps)

    #输出GT视频
    # output_folder = 'videos_result/gt'
    # process_gt(input_folder, output_folder, fps)


    #################TVD##########################################
    # # 单独输出视频，根据exp_name选择
    # exp_name = 'exp_fastDVDnetMini'
    # exp_name = 'exp16_netdepcv_L1_fft'
    # exp_name = 'input'

    # input_folder = os.path.join('H:/My-JDD/result_tvd/MoCE_JDD/',  exp_name)
    # output_folder = os.path.join('H:/My-JDD/result_tvd/videos_result/', exp_name)
    # fps = 25
    # process_experiment(input_folder, output_folder, fps)

    # 输出GT视频
    input_folder = 'F:/datasets/Tencent_Video_Dataset/part'
    output_folder = 'H:/My-JDD/result_tvd/videos_result/gt'
    fps = 25
    process_gt(input_folder, output_folder, fps)