# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import pathlib
import argparse
import sys
import glob
import json
import multiprocessing
import concurrent.futures
import math
import threading
import queue
import subprocess

# --- 核心依赖 ---
import numpy as np
import cv2
import tifffile
from tqdm import tqdm
import onnxruntime as ort
from PIL import Image

# --- GUI 依赖 ---
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog

# --- 动态导入核心逻辑模块 ---
try:
    from unprocessor_np import IspProcessor, ImageUnprocessor

    MODULES_IMPORTED_SUCCESSFULLY = True
except ImportError:
    print("警告: 无法从 'unprocessor_np' 导入核心模块。")
    IspProcessor = None
    ImageUnprocessor = None
    MODULES_IMPORTED_SUCCESSFULLY = False


# =====================================================================
#          ONNX去马赛克处理器
# =====================================================================
class DemosaicProcessor:
    def __init__(self, model_path="models/hamilton_adam_demosaic.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"去马赛克ONNX模型未找到: '{model_path}'.")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX Demosaic Processor 已初始化。输入名: '{self.input_name}'")

    def process(self, bayer_input_np: np.ndarray) -> np.ndarray:
        ort_outputs = self.session.run(None, {self.input_name: bayer_input_np})
        return ort_outputs[0]


# =====================================================================
#                          辅助函数
# =====================================================================
def np_to_pil(img_np):
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        if ar.shape[0] < ar.shape[-1]:
            ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)


def save_image(name, image_np, output_path="output/", return_path=False):
    try:
        os.makedirs(output_path, exist_ok=True)
        p = np_to_pil(image_np)
        full_path = os.path.join(output_path, f"{name}.png")
        p.save(full_path)
        if return_path:
            return full_path
    except Exception as e:
        print(f"Error saving image {name} to {output_path}: {e}")
    return None


# =====================================================================
#              多进程工作函数
# =====================================================================
def _process_single_frame_worker(args_tuple):
    frame_file, raw_output_folder, shared_metadata = args_tuple
    if ImageUnprocessor is None: return False, frame_file
    unprocessor = ImageUnprocessor()
    img = cv2.imread(frame_file)
    if img is None: return False, frame_file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = (img / 255.0).astype(np.float32)
    _, raw_noise, _, _ = unprocessor.forward(
        image, add_noise=True, rgb2cam=shared_metadata['rgb2cam'],
        rgb_gain=shared_metadata['rgb_gain'], red_gain=shared_metadata['red_gain'],
        blue_gain=shared_metadata['blue_gain'], shot_noise=shared_metadata['shot_noise'],
        read_noise=shared_metadata['read_noise']
    )
    frame_name = os.path.splitext(os.path.basename(frame_file))[0]
    raw_path = os.path.join(raw_output_folder, f"{frame_name}.tiff")
    isp_params = {
        'red_gain': float(shared_metadata['red_gain']), 'blue_gain': float(shared_metadata['blue_gain']),
        'rgb_gain': float(shared_metadata['rgb_gain']), 'rgb2cam': shared_metadata['rgb2cam'].tolist()
    }
    metadata_json = json.dumps(isp_params, indent=4)
    try:
        tifffile.imwrite(raw_path, raw_noise.astype(np.float32), description=metadata_json)
        return True, frame_file
    except Exception as e:
        print(f"保存TIFF {raw_path} 时出错: {e}")
        return False, frame_file


def _extract_frame_range_worker(args_tuple):
    video_path, start_frame, num_frames_to_extract, output_folder = args_tuple
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    extracted_count = 0
    for i in range(num_frames_to_extract):
        ret, frame = cap.read()
        if not ret: break
        frame_path = os.path.join(output_folder, f"{start_frame + i:08d}.png")
        cv2.imwrite(frame_path, frame)
        extracted_count += 1
    cap.release()
    return extracted_count


# =====================================================================
#                          数据预处理
# =====================================================================
class DataProcess:
    def __init__(self, args):
        self.args = args

    def extract_and_process_video(self, video_path, frames_output_dir):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder_for_video = os.path.join(frames_output_dir, video_name)
        pathlib.Path(output_folder_for_video).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError(f"无法打开视频: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames == 0: return None
        print(f"开始从 {video_path} 并行提取 {total_frames} 帧...")
        num_workers = max(1, os.cpu_count() - 2)
        frames_per_worker = math.ceil(total_frames / num_workers)
        tasks = []
        for i in range(num_workers):
            start_frame = i * frames_per_worker
            if start_frame >= total_frames: continue
            num_to_extract = min(frames_per_worker, total_frames - start_frame)
            tasks.append((video_path, start_frame, num_to_extract, output_folder_for_video))
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(executor.map(_extract_frame_range_worker, tasks), total=len(tasks), desc=f"提取帧 ({video_name})"))
        if self.args.stop_event.is_set(): return None
        self.process_frames_to_raw(output_folder_for_video)
        return os.path.join(self.args.raw_output_dir, video_name)

    def process_frames_to_raw(self, frames_folder):
        video_name = os.path.basename(frames_folder)
        raw_output_folder = os.path.join(self.args.raw_output_dir, video_name)
        pathlib.Path(raw_output_folder).mkdir(parents=True, exist_ok=True)
        frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
        if not frame_files: return
        print("为视频序列生成统一的ISP元数据...")
        unprocessor = ImageUnprocessor()
        rgb2cam = unprocessor.random_ccm()
        rgb_gain, red_gain, blue_gain = unprocessor.random_gains()
        shot_noise, read_noise = unprocessor.random_noise_levels()
        shared_metadata = {
            'rgb2cam': rgb2cam, 'rgb_gain': rgb_gain, 'red_gain': red_gain, 'blue_gain': blue_gain,
            'shot_noise': shot_noise, 'read_noise': read_noise
        }
        print("元数据生成完毕，将应用于所有帧。")
        print(f"开始将 {len(frame_files)} 帧图像处理为TIFF...")
        num_workers = max(1, os.cpu_count() - 2)
        tasks_with_metadata = [(f, raw_output_folder, shared_metadata) for f in frame_files]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(_process_single_frame_worker, tasks_with_metadata), total=len(frame_files),
                                desc=f"处理为RAW ({video_name})"))
        print(f"处理完成，浮点型RAW图像已保存到 {raw_output_folder}")


# =====================================================================
#                          数据加载
# =====================================================================
def read_tiff_with_isp_metadata(tiff_path):
    if not os.path.exists(tiff_path): raise FileNotFoundError(f"文件未找到: {tiff_path}")
    with tifffile.TiffFile(tiff_path) as tif:
        image_data = tif.asarray()
        metadata_json = tif.pages[0].tags.get('ImageDescription', None)
        if metadata_json is None: return image_data, None
        isp_params = json.loads(metadata_json.value)
        if 'rgb2cam' in isp_params:
            isp_params['rgb2cam'] = np.array(isp_params['rgb2cam'])
        return image_data, isp_params


class ONNXRawInferenceDataset:
    def __init__(self, root_dir, bundle_frame=5, frame_sampling_rate=1):
        assert bundle_frame % 2 == 1, 'Bundle_frame 必须是奇数'
        self.root_dir = root_dir
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2
        self.frame_sampling_rate = frame_sampling_rate
        self.sequences = self._find_sequences()
        if not self.sequences:
            raise FileNotFoundError(f"在 '{root_dir}' 目录下没有找到任何TIFF序列或包含TIFF序列的子目录。")
        self.bundle_definitions = self._create_bundle_definitions()

    def _generate_random_metadata(self):
        if ImageUnprocessor is None:
            return None
        print("警告: 未找到元数据，正在生成随机ISP元数据...")
        unprocessor = ImageUnprocessor()
        rgb2cam = unprocessor.random_ccm()
        rgb_gain, red_gain, blue_gain = unprocessor.random_gains()
        return {
            'red_gain': float(red_gain), 'blue_gain': float(blue_gain),
            'rgb_gain': float(rgb_gain), 'rgb2cam': rgb2cam.tolist()
        }

    def _find_sequences(self):
        sequences = []
        if any(f.lower().endswith(('.tiff', '.tif')) for f in os.listdir(self.root_dir)):
            sequences.append({
                'name': os.path.basename(self.root_dir),
                'path': self.root_dir
            })
        else:
            for d in os.listdir(self.root_dir):
                path = os.path.join(self.root_dir, d)
                if os.path.isdir(path):
                    if any(f.lower().endswith(('.tiff', '.tif')) for f in os.listdir(path)):
                        sequences.append({'name': d, 'path': path})
        return sorted(sequences, key=lambda x: x['name'])

    def _create_bundle_definitions(self):
        bundle_defs = []
        print("正在扫描数据集并创建捆绑包定义...")
        for seq_idx, seq_info in enumerate(tqdm(self.sequences, desc="扫描视频序列")):
            frame_files = sorted(glob.glob(os.path.join(seq_info['path'], "*.tiff")))
            if not frame_files: continue
            _, metadata = read_tiff_with_isp_metadata(frame_files[0])
            if metadata is None:
                metadata = self._generate_random_metadata()
            if metadata and 'rgb2cam' in metadata and isinstance(metadata['rgb2cam'], np.ndarray):
                metadata['rgb2cam'] = metadata['rgb2cam'].tolist()
            seq_info['metadata'] = metadata
            seq_info['frame_files'] = frame_files
            num_frames = len(frame_files)
            for center_frame_local_idx in range(0, num_frames, self.frame_sampling_rate):
                bundle_defs.append({
                    'seq_idx': seq_idx,
                    'center_frame_local_idx': center_frame_local_idx
                })
        return bundle_defs

    def __len__(self):
        return len(self.bundle_definitions)

    def __getitem__(self, idx):
        bundle_info = self.bundle_definitions[idx]
        seq_idx = bundle_info['seq_idx']
        center_idx = bundle_info['center_frame_local_idx']
        sequence = self.sequences[seq_idx]
        video_frame_paths = sequence['frame_files']
        num_frames_in_video = len(video_frame_paths)
        indices = list(range(center_idx - self.n, center_idx + self.n + 1))
        padded_indices = np.clip(indices, 0, num_frames_in_video - 1).tolist()
        raw_frames = [read_tiff_with_isp_metadata(video_frame_paths[frame_idx])[0] for frame_idx in padded_indices]
        raw_bundle = np.stack(raw_frames, axis=0)
        raw_bundle = np.transpose(raw_bundle, (0, 3, 1, 2))
        metadata = sequence['metadata'].copy()
        if metadata and 'rgb2cam' in metadata:
            metadata['rgb2cam'] = np.array(metadata['rgb2cam'])
        return {
            'raw_bundle': raw_bundle.astype(np.float32),
            'metadata': metadata,
            'video_name': sequence['name'],
            'center_frame_filename': os.path.basename(video_frame_paths[center_idx])
        }


# =====================================================================
#                          核心处理逻辑
# =====================================================================
def infer_with_tiling(session, input_name, input_batch, num_tiles,
                      isp_processor=None, gains=None, cam2rgb=None, perform_isp=True):
    try:
        B, T, C, H, W = input_batch.shape
    except ValueError:
        print(f"错误: 输入张量形状无法解包为5个维度。实际形状: {input_batch.shape}")
        return None

    if num_tiles == 1:
        output = session.run(None, {input_name: input_batch})[0]
        if perform_isp:
            output = np.transpose(output, (0, 2, 3, 1))
            processed_img = isp_processor.process(output, gains['red'], gains['blue'], cam2rgb, gains['rgb'], dem=False)
            return processed_img[0]
        else:
            return output[0]

    OVERLAP = 32
    if H <= OVERLAP * 2 or W <= OVERLAP * 2:
        return infer_with_tiling(session, input_name, input_batch, 1, isp_processor, gains, cam2rgb, perform_isp)

    rows, cols = (2, 2) if num_tiles == 4 else (2, 4) if num_tiles == 8 else (1, 1)
    stitched_output = np.zeros((H, W, 3), dtype=np.float32) if perform_isp else np.zeros((3, H, W), dtype=np.float32)
    tile_h, tile_w = H // rows, W // cols

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * tile_h, ((i + 1) * tile_h if i < rows - 1 else H)
            x1, x2 = j * tile_w, ((j + 1) * tile_w if j < cols - 1 else W)
            y1_pad, y2_pad = np.clip(y1 - OVERLAP, 0, H), np.clip(y2 + OVERLAP, 0, H)
            x1_pad, x2_pad = np.clip(x1 - OVERLAP, 0, W), np.clip(x2 + OVERLAP, 0, W)
            tile_input = input_batch[:, :, :, y1_pad:y2_pad, x1_pad:x2_pad]
            tile_output = session.run(None, {input_name: tile_input})[0]
            crop_y1, crop_y2 = y1 - y1_pad, (y1 - y1_pad) + (y2 - y1)
            crop_x1, crop_x2 = x1 - x1_pad, (x1 - x1_pad) + (x2 - x1)

            if perform_isp:
                processed_tile = \
                    isp_processor.process(np.transpose(tile_output, (0, 2, 3, 1)), gains['red'], gains['blue'], cam2rgb,
                                          gains['rgb'], dem=False)[0]
                stitched_output[y1:y2, x1:x2, :] = processed_tile[crop_y1:crop_y2, crop_x1:crop_x2, :]
            else:
                stitched_output[:, y1:y2, x1:x2] = tile_output[0][:, crop_y1:crop_y2, crop_x1:crop_x2]

    return stitched_output


def run_processing_logic(opt):
    if opt.random_seed and opt.random_seed.isdigit():
        np.random.seed(int(opt.random_seed))
        print(f"已设置Numpy随机数种子为: {opt.random_seed}")
    else:
        print("未设置或种子格式无效，将使用随机初始化。")

    processed_sequences_fps = {}
    model_name = "models/model_isp.onnx" if opt.input_mode == 'tiffs_with_isp_model' else "models/model.onnx"

    if opt.input_mode == 'video':
        root_dir_for_dataset = opt.raw_output_dir
        if not opt.skip_generation:
            processor = DataProcess(opt)
            video_paths = sorted(glob.glob(os.path.join(opt.video_input_dir, '*.mp4')))
            if not video_paths:
                print(f"错误: 在文件夹 '{opt.video_input_dir}' 中未找到任何 .mp4 文件。")
                return
            for video_path in video_paths:
                if opt.stop_event.is_set(): return
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                cap = cv2.VideoCapture(video_path)
                processed_sequences_fps[video_name] = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
                cap.release()
                processor.extract_and_process_video(video_path, opt.frames_output_dir)
    else:  # tiffs or tiffs_with_isp_model
        root_dir_for_dataset = opt.tiff_input_dir

    if opt.stop_event.is_set(): return
    if not os.path.exists(model_name):
        print(f"错误: ONNX模型未找到: {model_name}")
        return

    ort_session = ort.InferenceSession(model_name, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    try:
        dataset = ONNXRawInferenceDataset(root_dir=root_dir_for_dataset, bundle_frame=5,
                                          frame_sampling_rate=opt.frame_sampling_rate)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    perform_isp = (opt.input_mode != 'tiffs_with_isp_model')
    isp = IspProcessor() if perform_isp and IspProcessor else None
    demosaic_processor = DemosaicProcessor()

    processed_sequences = {}
    ### 修改开始: 初始化一个新的字典来存储带噪图像的可视化结果 ###
    noisy_sequences = {}
    ### 修改结束 ###

    for i in tqdm(range(len(dataset)), desc="推理全部帧"):
        if opt.stop_event.is_set(): break
        data_bundle = dataset[i]
        video_name = data_bundle['video_name']

        result_dir = os.path.join('results', video_name)
        pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
        if video_name not in processed_sequences:
            processed_sequences[video_name] = {'path': result_dir, 'frames': []}

        output_filename_base = f"{len(processed_sequences[video_name]['frames']):08d}"

        # --- 主模型推理流程 ---
        input_batch = np.expand_dims(demosaic_processor.process(data_bundle['raw_bundle']), axis=0)

        if perform_isp and isp:
            metadata = data_bundle['metadata']
            gains = {k: np.expand_dims(np.array(metadata[v]), 0) for k, v in
                     [('rgb', 'rgb_gain'), ('red', 'red_gain'), ('blue', 'blue_gain')]}
            cam2rgb = np.expand_dims(np.linalg.inv(np.array(metadata['rgb2cam'])), 0)

            ### 修改开始: 增加对带噪输入的 可视化处理 ###
            # 1. 准备目录和数据结构
            noisy_result_dir = os.path.join('results', f"{video_name}_noisy")
            pathlib.Path(noisy_result_dir).mkdir(parents=True, exist_ok=True)
            if video_name not in noisy_sequences:
                noisy_sequences[video_name] = {'path': noisy_result_dir, 'frames': []}

            # 2. 提取输入帧束的中心帧 (这就是带噪的RAW输入)
            # raw_bundle shape: (T, C, H, W), T=5, C=4. 我们需要中心帧 T//2
            center_noisy_raw = data_bundle['raw_bundle'][data_bundle['raw_bundle'].shape[0] // 2]

            # 3. 对中心帧进行去马赛克
            # demosaic需要(T,C,H,W)输入, 所以加一维
            demosaiced_noisy_raw = demosaic_processor.process(np.expand_dims(center_noisy_raw, axis=0))

            # 4. 对去马赛克后的图像应用ISP流程,使其可视化
            # isp.process需要(B, H, W, C)输入, B=1
            demosaiced_noisy_raw_transposed = np.transpose(demosaiced_noisy_raw, (0, 2, 3, 1))
            noisy_rgb_image = isp.process(demosaiced_noisy_raw_transposed, gains['red'], gains['blue'], cam2rgb,
                                          gains['rgb'], dem=False)

            # 5. 裁剪并保存
            noisy_rgb_image_clipped = np.clip(noisy_rgb_image[0], 0, 1)  # 从(B,H,W,C)中取出
            noisy_output_image_for_save = np.transpose(noisy_rgb_image_clipped, (2, 0, 1))  # 转为(C,H,W)以供保存

            if noisy_png_path := save_image(output_filename_base, noisy_output_image_for_save,
                                            output_path=noisy_result_dir, return_path=True):
                noisy_sequences[video_name]['frames'].append(noisy_png_path)
            ### 修改结束 ###

            # --- 继续原有的去噪流程 ---
            processed_img = infer_with_tiling(ort_session, input_name, input_batch, opt.num_tiles, isp, gains, cam2rgb,
                                              True)
            output_image = np.transpose(np.clip(processed_img, 0, 1), (2, 0, 1)) if processed_img is not None else None
        else:
            # 对于模型内置ISP的情况，不单独进行带噪可视化
            processed_img = infer_with_tiling(ort_session, input_name, input_batch, opt.num_tiles, perform_isp=False)
            output_image = np.clip(processed_img, 0, 1) if processed_img is not None else None

        if output_image is not None:
            if png_path := save_image(output_filename_base, output_image, output_path=result_dir, return_path=True):
                processed_sequences[video_name]['frames'].append(png_path)

    if opt.stop_event.is_set(): return
    if opt.export_to_mp4:
        # --- 导出处理后的干净视频 ---
        for video_name, info in processed_sequences.items():
            if not info['frames']: continue
            output_video_path = os.path.join('results', f"{video_name}.mp4")
            first_frame = cv2.imread(info['frames'][0])
            if first_frame is None: continue
            height, width, _ = first_frame.shape

            original_fps = processed_sequences_fps.get(video_name, 25.0)
            new_fps = original_fps / opt.frame_sampling_rate

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, new_fps, (width, height))
            for frame_path in tqdm(info['frames'], desc=f"导出 {video_name}.mp4 (去噪后)"):
                video_writer.write(cv2.imread(frame_path))
            video_writer.release()
            print(f"视频已保存至: {output_video_path}")

            if opt.delete_pngs:
                for frame_path in info['frames']:
                    os.remove(frame_path)
                try:
                    os.rmdir(info['path'])
                except OSError:
                    pass

        ### 修改开始: 增加导出带噪可视化视频的逻辑 ###
        for video_name, info in noisy_sequences.items():
            if not info['frames']: continue
            output_video_path = os.path.join('results', f"{video_name}_noisy_visualization.mp4")
            first_frame = cv2.imread(info['frames'][0])
            if first_frame is None: continue
            height, width, _ = first_frame.shape

            original_fps = processed_sequences_fps.get(video_name, 25.0)
            new_fps = original_fps / opt.frame_sampling_rate

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, new_fps, (width, height))
            for frame_path in tqdm(info['frames'], desc=f"导出 {video_name}.mp4 (带噪可视化)"):
                video_writer.write(cv2.imread(frame_path))
            video_writer.release()
            print(f"带噪可视化视频已保存至: {output_video_path}")

            if opt.delete_pngs:
                for frame_path in info['frames']:
                    os.remove(frame_path)
                try:
                    os.rmdir(info['path'])
                except OSError:
                    pass
        ### 修改结束 ###

    print("处理完成。")


# =====================================================================
#                               GUI 部分
# =====================================================================
class QueueHandler(queue.Queue):
    def write(self, msg):
        self.put(msg)

    def flush(self):
        if sys.__stdout__ is not None:
            try:
                sys.__stdout__.flush()
            except Exception:
                pass


class App(ttk.Window):
    def __init__(self):
        super().__init__(themename="cosmo")
        self.title("MoCE-JDD：视频联合去噪去马赛克算法")
        self.after(100, self._setup_icon)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width_ratio, height_ratio = 0.35, 0.7
        window_width = int(screen_width * width_ratio)
        window_height = int(screen_height * height_ratio)
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 2 - 20
        self.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        self.minsize(850, 800)

        self.vars = {
            'input_mode': tk.StringVar(value="video"),
            'video_input_dir': tk.StringVar(),
            'tiff_input_dir': tk.StringVar(),
            'frames_output_dir': tk.StringVar(value='video_frames'),
            'raw_output_dir': tk.StringVar(value='raw_frames'),
            'skip_generation': tk.BooleanVar(value=False),
            'tiling_option': tk.StringVar(value="不分块"),
            'frame_sampling_rate': tk.StringVar(value="每帧都处理"),
            'export_to_mp4': tk.BooleanVar(value=False),
            'delete_pngs': tk.BooleanVar(value=False),
            'random_seed': tk.StringVar(value="42"),
        }
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._create_widgets()

    def _setup_icon(self):
        try:
            base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(
                os.path.abspath(__file__))
            icon_path = os.path.join(base_path, 'asset', 'logo.png')
            if os.path.exists(icon_path):
                self.icon_image = tk.PhotoImage(file=icon_path)
                self.iconphoto(False, self.icon_image)
        except Exception as e:
            print(f"【Logo状态】加载图标时发生错误: {e}")

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)

        mode_frame = ttk.LabelFrame(main_frame, text="1. 输入模式", padding="15")
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))

        ttk.Radiobutton(mode_frame, text="处理视频文件夹(MP4)", variable=self.vars['input_mode'], value="video",
                        command=self._update_input_mode_ui, bootstyle="info").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="处理TIFF(RAW)文件夹(包含ISP参数,显式ISP)", variable=self.vars['input_mode'],
                        value="tiffs", command=self._update_input_mode_ui, bootstyle="info").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="处理TIFF(RAW)文件夹(模型完成ISP)", variable=self.vars['input_mode'],
                        value="tiffs_with_isp_model", command=self._update_input_mode_ui, bootstyle="info").pack(
            side=tk.LEFT, padx=10)

        paths_frame = ttk.LabelFrame(main_frame, text="2. 路径设置", padding="15")
        paths_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        paths_frame.grid_columnconfigure(1, weight=1)

        self.video_input_frame = ttk.Frame(paths_frame)
        self.video_input_frame.grid(row=0, column=0, columnspan=3, sticky='ew')
        self.video_input_frame.grid_columnconfigure(1, weight=1)

        self._create_path_entry(self.video_input_frame, "视频文件夹:", "video_input_dir", 0,
                                lambda: self._browse_directory_for_var('video_input_dir', "选择视频文件夹"))
        self._create_path_entry(self.video_input_frame, "帧输出目录:", "frames_output_dir", 1,
                                lambda: self._browse_directory_for_var('frames_output_dir', "选择帧输出目录"))
        self._create_path_entry(self.video_input_frame, "RAW输出目录:", "raw_output_dir", 2,
                                lambda: self._browse_directory_for_var('raw_output_dir', "选择RAW输出目录"))
        self.tiff_input_frame = ttk.Frame(paths_frame)
        self.tiff_input_frame.grid(row=0, column=0, columnspan=3, sticky='ew')
        self.tiff_input_frame.grid_columnconfigure(1, weight=1)
        self._create_path_entry(self.tiff_input_frame, "TIFF父文件夹:", "tiff_input_dir", 0,
                                lambda: self._browse_directory_for_var('tiff_input_dir', "选择TIFF父文件夹"))

        params_frame = ttk.LabelFrame(main_frame, text="3. 参数配置", padding="15")
        params_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        params_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(params_frame, text="推理分块:").grid(row=0, column=0, sticky="w", padx=5, pady=8)
        ttk.Combobox(params_frame, textvariable=self.vars['tiling_option'], values=["不分块", "4块 (2x2)", "8块 (2x4)"],
                     state="readonly").grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(params_frame, text="帧采样率:").grid(row=1, column=0, sticky="w", padx=5, pady=8)
        ttk.Combobox(params_frame, textvariable=self.vars['frame_sampling_rate'],
                     values=["每帧都处理", "每2帧处理1帧", "每3帧处理1帧", "每5帧处理1帧"], state="readonly").grid(
            row=1, column=1, sticky="w", padx=5)

        ttk.Label(params_frame, text="随机数种子:").grid(row=2, column=0, sticky="w", padx=5, pady=8)
        ttk.Entry(params_frame, textvariable=self.vars['random_seed'], width=20).grid(row=2, column=1, sticky="w",
                                                                                      padx=5)

        self.skip_check = ttk.Checkbutton(params_frame, text="跳过数据生成步骤 (如果RAW文件已存在)",
                                          variable=self.vars['skip_generation'], bootstyle="primary")
        self.skip_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=8)

        output_frame = ttk.LabelFrame(main_frame, text="4. 输出选项", padding="15")
        output_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))

        export_check = ttk.Checkbutton(output_frame, text="导出为MP4", variable=self.vars['export_to_mp4'],
                                       command=self._toggle_delete_png_checkbox, bootstyle="success")
        export_check.pack(side=tk.LEFT, padx=10)

        self.delete_png_check = ttk.Checkbutton(output_frame, text="完成后删除PNG序列",
                                                variable=self.vars['delete_pngs'], state=tk.DISABLED,
                                                bootstyle="danger")
        self.delete_png_check.pack(side=tk.LEFT, padx=10)

        control_frame = ttk.LabelFrame(main_frame, text="5. 执行与监控", padding="15")
        control_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 15))
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_rowconfigure(1, weight=1)

        button_bar = ttk.Frame(control_frame)
        button_bar.grid(row=0, column=0, sticky="ew", pady=5)

        self.run_button = ttk.Button(button_bar, text="▶ 开始处理", command=self.start_processing, bootstyle="success")
        self.run_button.pack(side=tk.LEFT, padx=10, ipady=5)

        self.interrupt_button = ttk.Button(button_bar, text="⏹ 中断", command=self.interrupt_processing,
                                           state=tk.DISABLED, bootstyle="danger-outline")
        self.interrupt_button.pack(side=tk.LEFT, padx=10, ipady=5)

        self.open_folder_button = ttk.Button(button_bar, text="📁 打开结果文件夹", command=self._open_result_folder,
                                             bootstyle="info")
        self.open_folder_button.pack(side=tk.RIGHT, padx=10, ipady=5)

        log_frame = ttk.Frame(control_frame)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, height=10,
                                relief="solid", bd=1)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview, bootstyle="round")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        self._update_input_mode_ui()

    def _update_input_mode_ui(self):
        mode = self.vars['input_mode'].get()
        if mode == "video":
            self.video_input_frame.grid()
            self.tiff_input_frame.grid_remove()
            self.skip_check.config(state=tk.NORMAL)
        else:
            self.video_input_frame.grid_remove()
            self.tiff_input_frame.grid()
            self.skip_check.config(state=tk.DISABLED)

    def _toggle_delete_png_checkbox(self):
        self.delete_png_check.config(state=tk.NORMAL if self.vars['export_to_mp4'].get() else tk.DISABLED)

    def _open_result_folder(self):
        path = "results"
        os.makedirs(path, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
        except Exception as e:
            ttk.dialogs.Messagebox.show_error(f"无法打开文件夹: {e}", title="操作失败")

    def _create_path_entry(self, parent, label_text, var_name, row, command):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=8)
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(parent, text="浏览...", command=command, bootstyle="secondary-outline").grid(row=row, column=2,
                                                                                                sticky="e", padx=5)

    def _browse_directory_for_var(self, var_name, title):
        path = filedialog.askdirectory(title=title)
        if path: self.vars[var_name].set(path)

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.update_idletasks()

    def interrupt_processing(self):
        self.log("收到中断信号，等待当前步骤完成...\n")
        self.stop_event.set()
        self.interrupt_button.config(state=tk.DISABLED)

    def start_processing(self):
        # Validation checks
        mode = self.vars['input_mode'].get()
        if mode == 'video' and not self.vars['video_input_dir'].get():
            ttk.dialogs.Messagebox.show_error("请选择视频文件夹！", "参数错误")
            return
        if 'tiffs' in mode and not self.vars['tiff_input_dir'].get():
            ttk.dialogs.Messagebox.show_error("请选择TIFF父文件夹！", "参数错误")
            return

        self.stop_event.clear()
        self.run_button.config(state=tk.DISABLED)
        self.interrupt_button.config(state=tk.NORMAL)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        args_dict = {k: v.get() for k, v in self.vars.items()}
        tiling_map = {"不分块": 1, "4块 (2x2)": 4, "8块 (2x4)": 8}
        sampling_map = {"每帧都处理": 1, "每2帧处理1帧": 2, "每3帧处理1帧": 3, "每5帧处理1帧": 5}
        args_dict['num_tiles'] = tiling_map.get(args_dict['tiling_option'], 1)
        args_dict['frame_sampling_rate'] = sampling_map.get(args_dict['frame_sampling_rate'], 1)
        args_dict['bundle_frame'] = 5  # Fixed value
        args_dict['stop_event'] = self.stop_event

        self.log_queue = QueueHandler()
        self.processing_thread = threading.Thread(
            target=self.processing_worker,
            args=(argparse.Namespace(**args_dict), self.log_queue),
            daemon=True)
        self.processing_thread.start()
        self.after(100, self.check_log_queue)

    def check_log_queue(self):
        while not self.log_queue.empty():
            self.log(self.log_queue.get_nowait())
        if self.processing_thread and self.processing_thread.is_alive():
            self.after(100, self.check_log_queue)
        else:
            self.run_button.config(state=tk.NORMAL)
            self.interrupt_button.config(state=tk.DISABLED)

    def processing_worker(self, args, log_queue):
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = log_queue, log_queue
        tqdm.pandas(file=log_queue)
        try:
            print("=" * 20 + " 开始处理 " + "=" * 20 + "\n")
            run_processing_logic(args)
            print("\n" + "=" * 20 + " 处理结束 " + "=" * 20)
        except Exception as e:
            import traceback
            print("\n" + "!" * 20 + " 发生错误 " + "!" * 20)
            print(f"错误类型: {type(e).__name__}\n错误信息: {e}\n详细追溯信息:\n{traceback.format_exc()}")
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            tqdm.pandas(file=sys.__stdout__)


# =====================================================================
#                          程序主入口
# =====================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()

    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)

    app = App()
    app.mainloop()