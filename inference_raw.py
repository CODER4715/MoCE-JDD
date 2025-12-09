import os
import numpy as np
import tifffile

from tqdm import tqdm
from skimage.util import img_as_ubyte

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net.moce_jdd_isp import MoCEJDD_ISP
from utils.test_utils import save_img

import cv2
import argparse
from matplotlib import pyplot as plt
import random
import yaml
import glob
from data.Hamilton_Adam_demo import HamiltonAdam


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class RawInferenceDataset(torch.utils.data.Dataset):
    # ... (这部分代码没有变化，保持原样) ...
    def __init__(self, root_dir, bundle_frame=5, frame_sampling_rate=1):
        """
        初始化数据集。

        参数:
        - root_dir (str): 包含一个或多个TIFF序列子目录的根目录。
        - bundle_frame (int): 每个数据捆绑包中的帧数，必须为奇数。
        - frame_sampling_rate (int): 采样中心帧的步长。
        """
        assert bundle_frame % 2 == 1, 'bundle_frame 必须是奇数'
        self.root_dir = root_dir
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2  # 中心帧单侧的邻近帧数量
        self.frame_sampling_rate = frame_sampling_rate

        # 查找包含TIFF文件的序列目录
        self.sequences = self._find_sequences()
        if not self.sequences:
            raise FileNotFoundError(f"在 '{root_dir}' 目录下没有找到任何包含TIFF序列的子目录。")

        # 创建捆绑包的定义
        self.bundle_definitions = self._create_bundle_definitions()

    def _find_sequences(self):
        """查找所有包含TIFF文件的子目录作为视频序列。"""
        sequences = []
        # 检查根目录本身是否是一个序列
        if any(f.lower().endswith(('.tiff', '.tif')) for f in os.listdir(self.root_dir)):
            sequences.append({
                'name': os.path.basename(self.root_dir),
                'path': self.root_dir
            })
        # 否则，检查其子目录
        else:
            for d in os.listdir(self.root_dir):
                path = os.path.join(self.root_dir, d)
                if os.path.isdir(path):
                    if any(f.lower().endswith(('.tiff', '.tif')) for f in os.listdir(path)):
                        sequences.append({'name': d, 'path': path})
        return sorted(sequences, key=lambda x: x['name'])

    def _create_bundle_definitions(self):
        """为数据集中的每个可能的捆绑包创建定义。"""
        bundle_defs = []
        print("正在扫描数据集并创建捆绑包定义...")
        for seq_idx, seq_info in enumerate(tqdm(self.sequences, desc="扫描视频序列")):
            # 查找所有tiff和tif文件并排序
            frame_files = sorted(
                glob.glob(os.path.join(seq_info['path'], "*.tiff")) +
                glob.glob(os.path.join(seq_info['path'], "*.tif"))
            )
            if not frame_files:
                continue

            # 存储文件列表
            seq_info['frame_files'] = frame_files
            num_frames = len(frame_files)

            # 根据采样率确定每个捆绑包的中心帧
            for center_frame_local_idx in range(0, num_frames, self.frame_sampling_rate):
                bundle_defs.append({
                    'seq_idx': seq_idx,
                    'center_frame_local_idx': center_frame_local_idx
                })
        return bundle_defs

    def __len__(self):
        """返回数据集中捆绑包的总数。"""
        return len(self.bundle_definitions)

    def __getitem__(self, idx):
        """
        获取一个数据捆绑包。

        参数:
        - idx (int): 捆绑包的索引。

        返回:
        - dict: 包含图像数据张量和元信息的字典。
        """
        bundle_info = self.bundle_definitions[idx]
        seq_idx = bundle_info['seq_idx']
        center_idx = bundle_info['center_frame_local_idx']

        sequence = self.sequences[seq_idx]
        video_frame_paths = sequence['frame_files']
        num_frames_in_video = len(video_frame_paths)

        # 计算捆绑包中所有帧的索引
        indices = list(range(center_idx - self.n, center_idx + self.n + 1))
        # 对于超出边界的索引，使用边缘帧进行填充（padding）
        padded_indices = np.clip(indices, 0, num_frames_in_video - 1).tolist()

        # 读取所有帧的图像数据
        raw_frames = []
        for frame_idx in padded_indices:
            path = video_frame_paths[frame_idx]
            with tifffile.TiffFile(path) as tif:
                image = tif.asarray()
            if image is None:
                raise IOError(f"无法读取文件: {path}")
            # 确保图像有通道维度，以便后续堆叠和转置 (H, W) -> (H, W, 1)
            if image.ndim == 2:
                image = image[..., np.newaxis]
            raw_frames.append(image)

        # 将帧列表堆叠成一个Numpy数组，形状为 (Bundle, Height, Width, Channels)
        raw_bundle = np.stack(raw_frames, axis=0)

        # 转置维度以匹配常见的深度学习输入格式 (B, C, H, W)
        raw_bundle = np.transpose(raw_bundle, (0, 3, 1, 2))

        raw_bundle_tensor = torch.from_numpy(raw_bundle.astype(np.float32))

        return {
            'raw_noise': raw_bundle_tensor,
            'video_name': sequence['name'],
            'center_frame_filename': os.path.basename(video_frame_paths[center_idx].split('.')[0])
        }


def tiled_inference(net, demosaic_raw, tile_grid_dims, overlap, device):
    """
    对输入的图像张量进行分块推理，并自动处理尺寸和5D填充问题。
    """
    grid_h, grid_w = tile_grid_dims
    if grid_h <= 1 and grid_w <= 1:
        return net(demosaic_raw)

    B, T, C, H, W = demosaic_raw.shape
    output_shape = (B, 3, H, W)

    final_output = torch.zeros(output_shape, device=device)
    weight_map = torch.zeros((B, 1, H, W), device=device)

    tile_h = H // grid_h
    tile_w = W // grid_w

    for i in range(grid_h):
        for j in range(grid_w):
            y_start = i * tile_h
            y_end = H if (i == grid_h - 1) else (i + 1) * tile_h
            x_start = j * tile_w
            x_end = W if (j == grid_w - 1) else (j + 1) * tile_w

            y_start_pad = max(0, y_start - overlap)
            y_end_pad = min(H, y_end + overlap)
            x_start_pad = max(0, x_start - overlap)
            x_end_pad = min(W, x_end + overlap)

            input_tile = demosaic_raw[:, :, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad]

            # --- 核心修复点: Reshape 5D -> 4D -> 5D 来解决 F.pad 的问题 ---
            h_tile, w_tile = input_tile.shape[-2:]

            MIN_DIVISOR = 16
            pad_h = (MIN_DIVISOR - h_tile % MIN_DIVISOR) % MIN_DIVISOR
            pad_w = (MIN_DIVISOR - w_tile % MIN_DIVISOR) % MIN_DIVISOR

            # 1. 获取原始的B, T, C维度信息
            B_tile, T_tile, C_tile, _, _ = input_tile.shape

            # 2. "压平"：将5D张量重塑为4D张量 (B*T, C, H, W)
            input_tile_4d = input_tile.view(B_tile * T_tile, C_tile, h_tile, w_tile)

            # 3. 对4D张量进行填充，这不会报错
            padded_input_tile_4d = F.pad(input_tile_4d,
                                         (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                                         mode='reflect')

            # 4. "恢复"：获取填充后的新尺寸，并将4D张量恢复为5D
            h_padded, w_padded = padded_input_tile_4d.shape[-2:]
            padded_input_tile = padded_input_tile_4d.view(B_tile, T_tile, C_tile, h_padded, w_padded)
            # --- 修复结束 ---

            # 对填充后的图块进行推理
            output_padded_tile = net(padded_input_tile)

            # 裁剪掉填充区域，得到与原始图块对应大小的输出
            output_tile = output_padded_tile[..., pad_h // 2: (pad_h // 2) + h_tile, pad_w // 2: (pad_w // 2) + w_tile]

            # 拼接逻辑：使用汉宁窗进行平滑融合
            hann_y = torch.hann_window(h_tile, periodic=False, device=device)
            hann_x = torch.hann_window(w_tile, periodic=False, device=device)
            hann_2d = (hann_y.unsqueeze(1) * hann_x.unsqueeze(0)).view(1, 1, h_tile, w_tile)

            final_output[:, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad] += output_tile * hann_2d
            weight_map[:, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad] += hann_2d

    final_output = final_output / (weight_map + 1e-8)

    return final_output


def run_inference(opts, net, dataloader, tile_grid_dims):  # <-- 修改了参数名
    testloader = dataloader
    save_folder = os.path.join(os.getcwd(), f"inference_raw_result")
    os.makedirs(save_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ham_demosaic = HamiltonAdam(pattern='rggb').to(device)

    with torch.no_grad():
        for batch in tqdm(testloader):
            raw_noise = batch['raw_noise'].to(device)
            video_name = batch['video_name'][0]
            frame_idx = batch['center_frame_filename'][0]

            demosaic_raw = torch.zeros((raw_noise.shape[0], raw_noise.shape[1],
                                        3, raw_noise.shape[3] * 2, raw_noise.shape[4] * 2)).to(device)

            for bundle in range(raw_noise.shape[0]):
                demosaic_raw[bundle] = ham_demosaic(raw_noise[bundle])

            restored = tiled_inference(net, demosaic_raw, tile_grid_dims, opts.tile_overlap, device)

            srgb_restored = restored
            srgb_restored = srgb_restored.permute(0, 2, 3, 1)

            video_folder = os.path.join(save_folder, video_name)
            os.makedirs(video_folder, exist_ok=True)
            srgb_restored = srgb_restored.clamp(0, 1)

            save_img(os.path.join(video_folder, f"{frame_idx}.png"),
                     img_as_ubyte(srgb_restored[0].cpu()))


def main(opt):
    seed_everything(42)


    if opt.tiling not in [1, 4, 8, 16]:
        raise ValueError(f"--tiling 的值必须是 1, 4, 8, 或 16, 但收到了 {opt.tiling}")

    # 将 tiling 值映射到 (height_grid, width_grid)
    grid_map = {1: (1, 1), 4: (2, 2), 8: (2, 4), 16: (4, 4)}
    tile_grid_dims = grid_map[opt.tiling]

    if opt.tiling > 1:
        print(f"启用分块推理: {tile_grid_dims[0]}x{tile_grid_dims[1]} 网格, 重叠 {opt.tile_overlap} 像素。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MoCEJDD_ISP(
        dim=opt.dim, num_blocks=opt.num_blocks, num_dec_blocks=opt.num_dec_blocks,
        levels=len(opt.num_blocks), heads=opt.heads, num_refinement_blocks=opt.num_refinement_blocks,
        topk=opt.topk, num_experts=opt.num_exp_blocks, rank=opt.latent_dim,
        with_complexity=opt.with_complexity, depth_type=opt.depth_type,
        stage_depth=opt.stage_depth, rank_type=opt.rank_type,
        complexity_scale=opt.complexity_scale,
    ).to(device)

    checkpoint = torch.load(os.path.join("checkpoints/MoCE_JDD/last_isp.ckpt"))
    model_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                        if k.startswith('net.')}
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    valset = RawInferenceDataset(root_dir=opt.video_path, bundle_frame=opt.num_frames)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)


    run_inference(opt, model, valloader, tile_grid_dims)


def load_hparams_to_opt(opt):
    hparams_path = os.path.join("checkpoints/MoCE_JDD", "hparams_isp.yaml")
    if not os.path.exists(hparams_path): return opt
    fields = ['heads', 'dim', 'num_blocks', 'num_dec_blocks',
              'num_exp_blocks', 'num_frames', 'num_refinement_blocks',
              'stage_depth', 'topk', 'latent_dim', 'with_complexity',
              'depth_type', 'rank_type', 'complexity_scale']
    try:
        with open(hparams_path, 'r', encoding='utf-8') as f:
            hparams = yaml.safe_load(f)
        for key in fields:
            if key in hparams:
                setattr(opt, key, hparams[key])
        print(f"Successfully loaded and updated hparams from {hparams_path}")
    except Exception as e:
        print(f"Error loading hparams.yaml: {e}")
    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="视频预处理与ONNX推理工具")
    parser.add_argument('--video_path', type=str, required=True, help='输入视频文件路径。')
    parser.add_argument('--num_frames', type=int, default=5, help='用于模型输入的连续帧数量。')

    # --- 修改点: 更新 --tiling 参数的帮助信息 ---
    parser.add_argument('--tiling', type=int, default=1, choices=[1, 4, 8, 16],
                        help='分块数量。1:不分块, 4:2x2, 8:2x4, 16:4x4。')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='分块推理时的重叠像素数。推荐值为16, 32或64。')

    opt = parser.parse_args()
    opt = load_hparams_to_opt(opt)
    main(opt)