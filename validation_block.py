# =====================================================================================
#   并行推理版本 (支持可选分块)
# =====================================================================================
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from skimage.util import img_as_ubyte

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from net.moce_jdd import MoCEJDD
from options import train_options
from utils.test_utils import save_img
from data.Hamilton_Adam_demo import HamiltonAdam
from data.unprocessor import IspProcessor, ImageUnprocessor
from PIL import Image

from data.datasets import VariableFramesDataset
from net.fastDVDnet_mini import FastDVDnet


class REDS_Val_Dataset(Dataset):
    def __init__(self, root_dir, bundle_frame=5, downsample_ratio=1, args=None):
        assert bundle_frame % 2 == 1, 'Bundle_frame must be Odd number'
        self.root_dir = root_dir
        self.bundle_frame = bundle_frame
        self.n = (bundle_frame - 1) // 2
        self.downsample_ratio = downsample_ratio
        self.unprocessor = ImageUnprocessor()
        self.get_srgb = False
        self.trans12bit = args.trans12bit if args is not None else False

        self.video_dirs = sorted([d for d in os.listdir(root_dir)
                                  if os.path.isdir(os.path.join(root_dir, d))])

        self.video_params = []
        for _ in self.video_dirs:
            rgb2cam = self.unprocessor.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            rgb_gain, red_gain, blue_gain = self.unprocessor.random_gains()
            shot_noise, read_noise = self.unprocessor.random_noise_levels()
            self.video_params.append({
                'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain,
                'red_gain': red_gain, 'blue_gain': blue_gain,
                'shot_noise': shot_noise, 'read_noise': read_noise
            })

        self.frame_paths = self._get_all_frame_paths()

    def _get_all_frame_paths(self):
        paths = []
        num_frames_per_video = 500 // self.downsample_ratio
        for video_idx, video_dir_name in enumerate(self.video_dirs):
            for i in range(num_frames_per_video):
                frame_idx = i * self.downsample_ratio
                paths.append({'video_idx': video_idx, 'frame_idx': frame_idx})
        return paths

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path_info = self.frame_paths[idx]
        video_idx = path_info['video_idx']
        frame_idx = path_info['frame_idx']
        video_dir_name = self.video_dirs[video_idx]
        video_dir = os.path.join(self.root_dir, video_dir_name)
        frame_indices = list(range(max(0, frame_idx - self.n * self.downsample_ratio),
                                   min(500, frame_idx + (self.n + 1) * self.downsample_ratio), self.downsample_ratio))
        if len(frame_indices) < self.bundle_frame:
            if frame_idx < self.n * self.downsample_ratio:
                pad_count = self.bundle_frame - len(frame_indices)
                frame_indices = [frame_indices[0]] * pad_count + frame_indices
            else:
                pad_count = self.bundle_frame - len(frame_indices)
                frame_indices = frame_indices + [frame_indices[-1]] * pad_count
        frame_indices = frame_indices[:self.bundle_frame]
        params = self.video_params[video_idx]
        metadata = {k: v.clone() for k, v in params.items()}
        noise_frames, linear_RGB_frame, rgb_frame = [], None, None
        for i, frame_original_idx in enumerate(frame_indices):
            img_path = os.path.join(video_dir, f"{frame_original_idx:08d}.png")
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            forward_params = {
                'shot_noise': params['shot_noise'], 'read_noise': params['read_noise'],
                'rgb2cam': params['rgb2cam'], 'rgb_gain': params['rgb_gain'],
                'red_gain': params['red_gain'], 'blue_gain': params['blue_gain']
            }
            _, noise_image, linear_RGB, _ = self.unprocessor.forward(image_tensor, add_noise=True, **forward_params)
            if self.trans12bit:
                noise_image = noise_image * (4095 - 240) + 240
            if frame_original_idx == frame_idx:
                linear_RGB_frame = linear_RGB.permute(2, 0, 1)
                if self.trans12bit:
                    linear_RGB_frame = linear_RGB_frame * (4095 - 240) + 240
                if self.get_srgb:
                    rgb_frame = image_tensor.permute(2, 0, 1)
            noise_frames.append(noise_image)
        noise_frames = torch.stack(noise_frames).permute(0, 3, 1, 2)
        return {
            'raw_noise': noise_frames, 'linear_RGB': linear_RGB_frame, 'srgb': rgb_frame,
            'metadata': metadata, 'video_name': video_dir_name, 'frame_idx': frame_idx,
        }


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_hparams_to_opt(opt, exp_name):
    hparams_path = os.path.join("logs", exp_name, "lightning_logs", "version_0", "hparams.yaml")
    if not os.path.exists(hparams_path): return opt
    # 指定需要加载的字段
    fields = ['heads', 'dim', 'num_blocks', 'num_dec_blocks',
              'num_exp_blocks', 'num_frames', 'num_refinement_blocks',
              'stage_depth']
    try:
        with open(hparams_path, 'r', encoding='utf-8') as f:
            hparams = yaml.safe_load(f)
        # 只更新指定的字段
        for key in fields:
            if key in hparams:
                setattr(opt, key, hparams[key])
        print(f"Successfully loaded and updated hparams from {hparams_path}")
    except Exception as e:
        print(f"Error loading hparams.yaml: {e}")
    return opt


def run_inference_direct(opts, net, dataloader, device):
    """
    直接推理函数：对整个图像进行一次性推理，不分块。
    适用于显存充足或图像尺寸较小的场景。
    """
    save_folder = os.path.join(os.getcwd(), f"{opts.output_path}/{opts.checkpoint_id}/{opts.exp_name}")
    ham_demosaic = HamiltonAdam(pattern='rggb').to(device)
    isp = IspProcessor()

    is_srgb_mode = hasattr(opts, 'gt_type') and opts.gt_type == 'srgb'

    # 按需初始化指标计算器
    if not is_srgb_mode:
        psnr_linear_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_linear_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_srgb_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_srgb_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    all_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Direct Inference on {device}"):
            raw_noise = batch['raw_noise'].to(device, non_blocking=True)
            srgb_gt = batch['srgb'].to(device, non_blocking=True)
            if not is_srgb_mode:
                linear_gt = batch['linear_RGB'].to(device, non_blocking=True)

            metadata = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in
                        batch['metadata'].items()}
            video_names = batch['video_name']
            frame_indices = batch['frame_idx']
            current_batch_size = raw_noise.shape[0]

            demosaic_raw_list = [ham_demosaic(raw_noise[i]) for i in range(current_batch_size)]
            demosaic_raw_batch = torch.stack(demosaic_raw_list)

            if opts.save_results:
                center_frame_idx = demosaic_raw_batch.shape[1] // 2
                noisy_center_frame = demosaic_raw_batch[:, center_frame_idx, :, :, :].clone()
                srgb_noisy_input_batch = isp.process(
                    noisy_center_frame,
                    metadata['red_gain'], metadata['blue_gain'],
                    metadata['cam2rgb'], metadata['rgb_gain'], dem=False
                ).clamp(0, 1)

            # --- 直接对整个图像进行推理 ---
            restored_batch = net(demosaic_raw_batch, metadata).clamp(0, 1)

            srgb_restored_batch = isp.process(
                restored_batch,
                metadata['red_gain'], metadata['blue_gain'],
                metadata['cam2rgb'], metadata['rgb_gain'], dem=False
            ).clamp(0, 1)

            for i in range(current_batch_size):
                video_name, frame_idx = video_names[i], frame_indices[i].item()
                srgb_restored_single = srgb_restored_batch[i].unsqueeze(0).permute(0, 3, 1, 2)
                srgb_gt_single = srgb_gt[i].unsqueeze(0)

                srgb_psnr = psnr_srgb_metric(srgb_restored_single, srgb_gt_single).item()
                srgb_ssim = ssim_srgb_metric(srgb_restored_single, srgb_gt_single).item()

                if not np.isnan(srgb_psnr):
                    result_dict = {
                        'video': video_name, 'frame': frame_idx,
                        'srgb_psnr': srgb_psnr, 'srgb_ssim': srgb_ssim
                    }
                    if not is_srgb_mode:
                        restored_single = restored_batch[i].unsqueeze(0)
                        linear_gt_single = linear_gt[i].unsqueeze(0)
                        linear_psnr = psnr_linear_metric(restored_single, linear_gt_single).item()
                        linear_ssim = ssim_linear_metric(restored_single, linear_gt_single).item()
                        result_dict['linear_psnr'] = linear_psnr
                        result_dict['linear_ssim'] = linear_ssim

                    all_results.append(result_dict)

                if opts.save_results:
                    video_folder = os.path.join(save_folder, video_name)
                    input_video_folder = os.path.join(f"{opts.output_path}/{opts.checkpoint_id}", 'input', video_name)
                    os.makedirs(video_folder, exist_ok=True)
                    os.makedirs(input_video_folder, exist_ok=True)

                    srgb_restored_np = srgb_restored_single.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    save_img(os.path.join(video_folder, f"{frame_idx:08d}.png"), img_as_ubyte(srgb_restored_np))

                    srgb_noisy_np = srgb_noisy_input_batch[i].cpu().numpy()
                    noisy_input_save_path = os.path.join(input_video_folder, f"{frame_idx:08d}.png")
                    save_img(noisy_input_save_path, img_as_ubyte(srgb_noisy_np))
    return all_results


def run_inference_with_tiling(opts, net, dataloader, device):
    """
    带有瓦片化推理的核心函数。
    此版本使用重叠瓦片和线性融合算法，以实现更平滑的拼接效果。
    """
    save_folder = os.path.join(os.getcwd(), f"{opts.output_path}/{opts.checkpoint_id}/{opts.exp_name}")
    ham_demosaic = HamiltonAdam(pattern='rggb').to(device)
    isp = IspProcessor()

    is_srgb_mode = hasattr(opts, 'gt_type') and opts.gt_type == 'srgb'

    if not is_srgb_mode:
        psnr_linear_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_linear_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_srgb_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_srgb_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    all_results = []

    overlap = 32

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Blended Tiling Inference on {device}"):
            raw_noise = batch['raw_noise'].to(device, non_blocking=True)
            srgb_gt = batch['srgb'].to(device, non_blocking=True)
            if not is_srgb_mode:
                linear_gt = batch['linear_RGB'].to(device, non_blocking=True)

            metadata = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in
                        batch['metadata'].items()}
            video_names = batch['video_name']
            frame_indices = batch['frame_idx']

            current_batch_size = raw_noise.shape[0]

            demosaic_raw_list = [ham_demosaic(raw_noise[i]) for i in range(current_batch_size)]
            demosaic_raw_batch = torch.stack(demosaic_raw_list)

            if opts.save_results:
                center_frame_idx = demosaic_raw_batch.shape[1] // 2
                noisy_center_frame = demosaic_raw_batch[:, center_frame_idx, :, :, :].clone()
                srgb_noisy_input_batch = isp.process(
                    noisy_center_frame,
                    metadata['red_gain'], metadata['blue_gain'],
                    metadata['cam2rgb'], metadata['rgb_gain'], dem=False
                ).clamp(0, 1)

            B, F, C, H, W = demosaic_raw_batch.shape
            h_split, w_split = H // 2, W // 2
            overlap_h, overlap_w = overlap // 2, overlap // 2

            tile_coords = [
                (0, h_split + overlap_h, 0, w_split + overlap_w), (0, h_split + overlap_h, w_split - overlap_w, W),
                (h_split - overlap_h, H, 0, w_split + overlap_w), (h_split - overlap_h, H, w_split - overlap_w, W)
            ]

            output_tiles = []
            for h_start, h_end, w_start, w_end in tile_coords:
                h_start, w_start = max(0, h_start), max(0, w_start)
                h_end, w_end = min(H, h_end), min(W, w_end)
                input_tile = demosaic_raw_batch[:, :, :, h_start:h_end, w_start:w_end]
                restored_tile = net(input_tile, metadata).clamp(0, 1)
                output_tiles.append(restored_tile)

            tl, tr, bl, br = output_tiles[0], output_tiles[1], output_tiles[2], output_tiles[3]

            weight_w = torch.linspace(1, 0, overlap, device=device).view(1, 1, 1, overlap)
            weight_h = torch.linspace(1, 0, overlap, device=device).view(1, 1, overlap, 1)

            core_tl = tl[:, :, :, :-overlap];
            core_tr = tr[:, :, :, overlap:]
            overlap_l = tl[:, :, :, -overlap:];
            overlap_r = tr[:, :, :, :overlap]
            blended_top_seam = overlap_l * weight_w + overlap_r * (1 - weight_w)
            top_row = torch.cat([core_tl, blended_top_seam, core_tr], dim=3)

            core_bl = bl[:, :, :, :-overlap];
            core_br = br[:, :, :, overlap:]
            overlap_l = bl[:, :, :, -overlap:];
            overlap_r = br[:, :, :, :overlap]
            blended_bottom_seam = overlap_l * weight_w + overlap_r * (1 - weight_w)
            bottom_row = torch.cat([core_bl, blended_bottom_seam, core_br], dim=3)

            core_top = top_row[:, :, :-overlap, :];
            core_bottom = bottom_row[:, :, overlap:, :]
            overlap_t = top_row[:, :, -overlap:, :];
            overlap_b = bottom_row[:, :, :overlap, :]
            blended_final_seam = overlap_t * weight_h + overlap_b * (1 - weight_h)
            restored_batch = torch.cat([core_top, blended_final_seam, core_bottom], dim=2)

            srgb_restored_batch = isp.process(
                restored_batch,
                metadata['red_gain'], metadata['blue_gain'],
                metadata['cam2rgb'], metadata['rgb_gain'], dem=False
            ).clamp(0, 1)

            for i in range(current_batch_size):
                video_name, frame_idx = video_names[i], frame_indices['frame'].item()
                srgb_restored_single = srgb_restored_batch[i].unsqueeze(0).permute(0, 3, 1, 2)
                srgb_gt_single = srgb_gt[i].unsqueeze(0)

                srgb_psnr = psnr_srgb_metric(srgb_restored_single, srgb_gt_single).item()
                srgb_ssim = ssim_srgb_metric(srgb_restored_single, srgb_gt_single).item()

                if not np.isnan(srgb_psnr):
                    result_dict = {
                        'video': video_name, 'frame': frame_idx,
                        'srgb_psnr': srgb_psnr, 'srgb_ssim': srgb_ssim
                    }
                    if not is_srgb_mode:
                        restored_single = restored_batch[i].unsqueeze(0)
                        linear_gt_single = linear_gt[i].unsqueeze(0)
                        linear_psnr = psnr_linear_metric(restored_single, linear_gt_single).item()
                        linear_ssim = ssim_linear_metric(restored_single, linear_gt_single).item()
                        result_dict['linear_psnr'] = linear_psnr
                        result_dict['linear_ssim'] = linear_ssim

                    all_results.append(result_dict)

                if opts.save_results:
                    video_folder = os.path.join(save_folder, video_name)
                    input_video_folder = os.path.join(save_folder, 'input', video_name)
                    os.makedirs(video_folder, exist_ok=True)
                    os.makedirs(input_video_folder, exist_ok=True)

                    srgb_restored_np = srgb_restored_single.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    save_img(os.path.join(video_folder, f"{frame_idx:08d}.png"), img_as_ubyte(srgb_restored_np))

                    srgb_noisy_np = srgb_noisy_input_batch[i].cpu().numpy()
                    noisy_input_save_path = os.path.join(input_video_folder, f"{frame_idx:08d}.png")
                    save_img(noisy_input_save_path, img_as_ubyte(srgb_noisy_np))
    return all_results


def main(opt):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查 gt_type 模式
    is_srgb_mode = hasattr(opt, 'gt_type') and opt.gt_type == 'srgb'

    BATCH_SIZE = 1
    NUM_WORKERS = 2

    model = MoCEJDD(
        dim=opt.dim, num_blocks=opt.num_blocks, num_dec_blocks=opt.num_dec_blocks,
        levels=len(opt.num_blocks), heads=opt.heads, num_refinement_blocks=opt.num_refinement_blocks,
        topk=opt.topk, num_experts=opt.num_exp_blocks, rank=opt.latent_dim, with_complexity=opt.with_complexity,
        depth_type=opt.depth_type, stage_depth=opt.stage_depth, rank_type=opt.rank_type,
        complexity_scale=opt.complexity_scale, need_isp=opt.isp_bwd,
    ).to(device)

    # model = FastDVDnet(need_isp=opt.isp_bwd,).to(device)

    checkpoint_path = os.path.join(opt.ckpt_dir, opt.exp_name, "last.ckpt")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('net.')}
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()

    print(f"Configuring DataLoader: Batch Size = {BATCH_SIZE}, Num Workers = {NUM_WORKERS}")

    valset = VariableFramesDataset(
        root_dir=opt.val_dir,
        bundle_frame=opt.num_frames,
        is_train=False,
        seed=42,
        get_srgb=True
    )

    valloader = DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    # --- 根据参数选择推理模式 ---
    if hasattr(opt, 'use_tiling') and opt.use_tiling:
        print(">>> Mode: Tiled inference with blending enabled.")
        results = run_inference_with_tiling(opt, model, valloader, device)
    else:
        print(">>> Mode: Direct inference (whole image at once).")
        print(">>> If memory issues occur, try running with the --use_tiling flag.")
        results = run_inference_direct(opt, model, valloader, device)

    print("\n--- Inference finished. Aggregating results. ---")

    video_metrics = {}
    for res in results:
        video = res['video']
        if video not in video_metrics:
            video_metrics[video] = {'srgb_psnr': [], 'srgb_ssim': []}
            if not is_srgb_mode:
                video_metrics[video].update({'linear_psnr': [], 'linear_ssim': []})

        video_metrics[video]['srgb_psnr'].append(res['srgb_psnr'])
        video_metrics[video]['srgb_ssim'].append(res['srgb_ssim'])
        if not is_srgb_mode:
            video_metrics[video]['linear_psnr'].append(res['linear_psnr'])
            video_metrics[video]['linear_ssim'].append(res['linear_ssim'])

    video_avg_results_str = []
    for video_name in sorted(video_metrics.keys()):
        avg_srgb_psnr = np.mean(video_metrics[video_name]['srgb_psnr'])
        avg_srgb_ssim = np.mean(video_metrics[video_name]['srgb_ssim'])
        srgb_part = f"sRGB PSNR: {avg_srgb_psnr:.4f}, SSIM: {avg_srgb_ssim:.4f}"

        if not is_srgb_mode:
            avg_linear_psnr = np.mean(video_metrics[video_name]['linear_psnr'])
            avg_linear_ssim = np.mean(video_metrics[video_name]['linear_ssim'])
            linear_part = f"Linear RGB PSNR: {avg_linear_psnr:.4f}, SSIM: {avg_linear_ssim:.4f}, "
            line = f"Video {video_name}: {linear_part}{srgb_part}\n"
        else:
            line = f"Video {video_name}: {srgb_part}\n"

        print(line.strip())
        video_avg_results_str.append(line)

    total_srgb_psnr = np.mean([res['srgb_psnr'] for res in results])
    total_srgb_ssim = np.mean([res['srgb_ssim'] for res in results])
    srgb_total_part = f"sRGB PSNR: {total_srgb_psnr:.4f}, SSIM: {total_srgb_ssim:.4f}"

    if not is_srgb_mode:
        total_linear_psnr = np.mean([res['linear_psnr'] for res in results])
        total_linear_ssim = np.mean([res['linear_ssim'] for res in results])
        linear_total_part = f"Linear RGB PSNR: {total_linear_psnr:.4f}, SSIM: {total_linear_ssim:.4f}, "
        total_avg_results_str = f"Total Average: {linear_total_part}{srgb_total_part}\n"
    else:
        total_avg_results_str = f"Total Average: {srgb_total_part}\n"

    print(total_avg_results_str)

    save_folder = os.path.join(os.getcwd(), f"{opt.output_path}/{opt.checkpoint_id}/{opt.exp_name}")
    result_file_path = os.path.join(save_folder, "results.txt")
    with open(result_file_path, "w") as f:
        f.writelines(video_avg_results_str)
        f.write("\n")
        f.write(total_avg_results_str)
    print(f"Results successfully saved to {result_file_path}")


# =====================================================================================
# 5. 脚本入口 (Script Entry Point)
# =====================================================================================
if __name__ == '__main__':
    train_opt = train_options()
    if hasattr(train_opt, 'exp_name') and train_opt.exp_name:
        train_opt = load_hparams_to_opt(train_opt, train_opt.exp_name)
    main(train_opt)