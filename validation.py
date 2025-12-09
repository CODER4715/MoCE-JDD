import os
import pathlib
import numpy as np

from tqdm import tqdm
from skimage.util import img_as_ubyte

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure,PeakSignalNoiseRatio

from net.moce_jdd import MoCEJDD
from net.fastDVDnet_mini import FastDVDnet
from options import train_options
from utils.test_utils import save_img
from data.datasets_v0 import REDS_Val_Dataset
import cv2

from data.Hamilton_Adam_demo import HamiltonAdam
from data.unprocessor import IspProcessor
# from CRVD_ISP.ISP import ISP
from matplotlib import pyplot as plt
import random
import yaml

def seed_everything(seed: int = 42):
    """
    设置所有随机数生成器的种子，确保实验的可重复性。

    Args:
        seed (int): 随机数种子，默认为 42。
    """
    # 设置 Python 的随机数种子
    random.seed(seed)
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    # 设置 PyTorch 的 CPU 随机数种子
    torch.manual_seed(seed)
    # 设置 PyTorch 的 CUDA 随机数种子
    torch.cuda.manual_seed(seed)
    # 为所有 CUDA 设备设置相同的随机数种子
    torch.cuda.manual_seed_all(seed)
    # 禁用 CUDA 的非确定性算法，保证结果可复现
    torch.backends.cudnn.deterministic = True
    # 禁用 cuDNN 的自动寻找最优卷积算法功能，保证结果可复现
    torch.backends.cudnn.benchmark = False
    # 设置系统环境变量的随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_inference(opts, net, dataloader):
    testloader = dataloader

    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"{opts.output_path}/{opts.checkpoint_id}")).mkdir(parents=True,
                                                                                                  exist_ok=True)
    save_folder = os.path.join(os.getcwd(), f"{opts.output_path}/{opts.checkpoint_id}/{opts.exp_name}")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ham_demosaic = HamiltonAdam(pattern='rggb').to(device)

    isp = IspProcessor()
    # isp = ISP().cuda()

    # 根据 gt_type 判断是否为 srgb 模式
    is_srgb_mode = (opts.gt_type == 'srgb')

    # sRGB 指标总是需要计算
    video_psnr_srgb = {}
    video_ssim_srgb = {}
    total_psnr_srgb = []
    total_ssim_srgb = []
    psnr_srgb = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_srgb = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 仅在需要时初始化 linear RGB 相关变量
    if not is_srgb_mode:
        video_psnr_linear = {}
        video_ssim_linear = {}
        total_psnr_linear = []
        total_ssim_linear = []
        psnr_linear = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_linear = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    prev_video_idx = None  # 用于跟踪上一个视频索引
    nan_frame_info = []

    with torch.no_grad():
        for batch in tqdm(testloader):
            # --- 数据加载 ---
            raw_noise = batch['raw_noise'].to(device)
            # 总是加载 srgb_gt，并用它的维度信息来初始化张量，更安全
            srgb_gt = batch['srgb'].to(device)
            metadata = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['metadata'].items()}
            data = batch['frame_idx']
            video_idx = data['video'][0].item()
            frame_idx = data['frame'][0].item()

            # 按需加载 linear_RGB
            if not is_srgb_mode:
                linear_RGB = batch['linear_RGB'].to(device)
                linear_RGB = linear_RGB.clamp(0, 1)

            # --- 当视频索引变化时，输出上一个视频的平均指标 ---
            if prev_video_idx is not None and video_idx != prev_video_idx:
                log_str = f"Video {prev_video_idx:03d}: "
                if not is_srgb_mode:
                    avg_linear_psnr = np.mean(video_psnr_linear[prev_video_idx]) if prev_video_idx in video_psnr_linear else np.nan
                    log_str += f"Linear RGB PSNR: {avg_linear_psnr:.4f}, "

                avg_srgb_psnr = np.mean(video_psnr_srgb[prev_video_idx]) if prev_video_idx in video_psnr_srgb else np.nan
                log_str += f"sRGB PSNR: {avg_srgb_psnr:.4f}"
                print(log_str)

            prev_video_idx = video_idx

            # --- 前向传播 ---
            # 使用 srgb_gt 的维度信息初始化 demosaic_raw，避免依赖 linear_RGB
            demosaic_raw = torch.zeros((raw_noise.shape[0], raw_noise.shape[1],
                                            srgb_gt.shape[1], srgb_gt.shape[2], srgb_gt.shape[3])).to(device)

            for bundle in range(raw_noise.shape[0]):
                demosaic_raw[bundle] = ham_demosaic(raw_noise[bundle])

            restored = net(demosaic_raw, metadata)

            if opts.gt_type == 'linear_rgb':
                srgb_restored = isp.process(restored, metadata['red_gain'],
                                                metadata['blue_gain'],
                                                metadata['cam2rgb'],
                                                metadata['rgb_gain'], dem=False)

                # restored_mosaic = rgb_to_rggb_packed_pytorch(restored)
                #
                # srgb_restored = isp(restored_mosaic)
                # plt.imshow(srgb_restored[0].clip(0,1).permute(1,2,0).cpu()*255)
                # plt.show()
                # print('1')
            elif opts.gt_type == 'srgb':
                srgb_restored = restored
                srgb_restored = srgb_restored.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"不支持的gt_type: {opts.gt_type}")

            # --- 指标计算 ---
            # 计算 sRGB 的 PSNR 和 SSIM (总是执行)
            srgb_psnr = psnr_srgb(srgb_restored.permute(0,3,1,2), srgb_gt).item()
            srgb_ssim = ssim_srgb(srgb_restored.permute(0,3,1,2), srgb_gt).item()
            print(f"Video {video_idx:03d}, Frame {frame_idx:08d}: sRGB PSNR: {srgb_psnr:.4f}")

            if not np.isnan(srgb_psnr) and not np.isnan(srgb_ssim):
                if video_idx not in video_psnr_srgb:
                    video_psnr_srgb[video_idx] = []
                    video_ssim_srgb[video_idx] = []
                video_psnr_srgb[video_idx].append(srgb_psnr)
                video_ssim_srgb[video_idx].append(srgb_ssim)
                total_psnr_srgb.append(srgb_psnr)
                total_ssim_srgb.append(srgb_ssim)

            # 计算 linearRGB 的 PSNR 和 SSIM (按需执行)
            if not is_srgb_mode:
                linear_psnr = psnr_linear(restored, linear_RGB).item()
                linear_ssim = ssim_linear(restored, linear_RGB).item()
                if not np.isnan(linear_psnr) and not np.isnan(linear_ssim):
                    if video_idx not in video_psnr_linear:
                        video_psnr_linear[video_idx] = []
                        video_ssim_linear[video_idx] = []
                    video_psnr_linear[video_idx].append(linear_psnr)
                    video_ssim_linear[video_idx].append(linear_ssim)
                    total_psnr_linear.append(linear_psnr)
                    total_ssim_linear.append(linear_ssim)

            # --- 保存结果 ---
            video_idx_str = f"{video_idx:03d}"
            frame_idx_str = f"{frame_idx:08d}"
            video_folder = os.path.join(save_folder, video_idx_str)
            rawnoise_folder = os.path.join(save_folder, 'input', video_idx_str)
            os.makedirs(video_folder, exist_ok=True)
            os.makedirs(rawnoise_folder, exist_ok=True)

            srgb_restored = srgb_restored.clamp(0, 1)
            save_img(os.path.join(video_folder, f"{frame_idx_str}.png"),
                     img_as_ubyte(srgb_restored[0].cpu()))

            demosaic_raw_center = demosaic_raw[0][2].unsqueeze(0)
            demosaic_raw_center = isp.process(demosaic_raw_center, metadata['red_gain'],
                                           metadata['blue_gain'],
                                           metadata['cam2rgb'],
                                           metadata['rgb_gain'], dem=False)
            demosaic_raw_center = demosaic_raw_center.clamp(0, 1)
            save_img(os.path.join(rawnoise_folder, f"{frame_idx_str}.png"),
                     img_as_ubyte(demosaic_raw_center[0].cpu()))

        # --- 处理最后一个视频的平均指标输出 ---
        if prev_video_idx is not None:
            log_str = f"Video {prev_video_idx:03d}: "
            if not is_srgb_mode:
                avg_linear_psnr = np.mean(video_psnr_linear[prev_video_idx]) if prev_video_idx in video_psnr_linear else np.nan
                log_str += f"Linear RGB PSNR: {avg_linear_psnr:.4f}, "

            avg_srgb_psnr = np.mean(video_psnr_srgb[prev_video_idx]) if prev_video_idx in video_psnr_srgb else np.nan
            log_str += f"sRGB PSNR: {avg_srgb_psnr:.4f}"
            print(log_str)

    # --- 汇总并输出最终结果 ---
    video_avg_results = []
    # 使用 srgb 的键作为已处理视频的权威列表
    all_video_indices = sorted(video_psnr_srgb.keys())

    for video_idx in all_video_indices:
        # sRGB 部分
        avg_srgb_psnr = np.mean(video_psnr_srgb[video_idx])
        avg_srgb_ssim = np.mean(video_ssim_srgb[video_idx])
        srgb_part = f"sRGB PSNR: {avg_srgb_psnr:.4f}, SSIM: {avg_srgb_ssim:.4f}"

        # linear 部分 (按需)
        if not is_srgb_mode:
            avg_linear_psnr = np.mean(video_psnr_linear[video_idx])
            avg_linear_ssim = np.mean(video_ssim_linear[video_idx])
            linear_part = f"Linear RGB PSNR: {avg_linear_psnr:.4f}, SSIM: {avg_linear_ssim:.4f}, "
            video_avg_results.append(f"Video {video_idx:03d}: {linear_part}{srgb_part}\n")
        else:
            video_avg_results.append(f"Video {video_idx:03d}: {srgb_part}\n")

    # --- 计算总平均值 ---
    total_avg_srgb_psnr = np.mean(total_psnr_srgb) if total_psnr_srgb else np.nan
    total_avg_srgb_ssim = np.mean(total_ssim_srgb) if total_ssim_srgb else np.nan
    srgb_total_part = f"sRGB PSNR: {total_avg_srgb_psnr:.4f}, SSIM: {total_avg_srgb_ssim:.4f}"

    if not is_srgb_mode:
        total_avg_linear_psnr = np.mean(total_psnr_linear) if total_psnr_linear else np.nan
        total_avg_linear_ssim = np.mean(total_ssim_linear) if total_ssim_linear else np.nan
        linear_total_part = f"Linear RGB PSNR: {total_avg_linear_psnr:.4f}, SSIM: {total_avg_linear_ssim:.4f}, "
        total_avg_results = f"Total Average: {linear_total_part}{srgb_total_part}\n"
    else:
        total_avg_results = f"Total Average: {srgb_total_part}\n"

    # --- 输出结果到文件 ---
    result_file_path = os.path.join(save_folder, "results.txt")
    with open(result_file_path, "w") as f:
        f.writelines(video_avg_results)
        f.write("\n")
        f.write(total_avg_results)
    print(f"Results saved to {result_file_path}")

    nan_log_file_path = os.path.join(save_folder, "nan_frames.log")
    with open(nan_log_file_path, "w") as f:
        f.writelines(nan_frame_info)
    print(f"Nan frame information saved to {nan_log_file_path}")


def main(opt):
    # 使用 seed_everything 函数设置随机数种子
    seed_everything(42)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MoCEJDD(
        dim=opt.dim,
        num_blocks=opt.num_blocks,
        num_dec_blocks=opt.num_dec_blocks,
        levels=len(opt.num_blocks),
        heads=opt.heads,
        num_refinement_blocks=opt.num_refinement_blocks,
        topk=opt.topk,
        num_experts=opt.num_exp_blocks,
        rank=opt.latent_dim,
        with_complexity=opt.with_complexity,
        depth_type=opt.depth_type,
        stage_depth=opt.stage_depth,
        rank_type=opt.rank_type,
        complexity_scale=opt.complexity_scale,
        need_isp=opt.isp_bwd,
    ).to(device)

    # model = FastDVDnet(need_isp=opt.isp_bwd,).to(device)

    checkpoint = torch.load(os.path.join(opt.ckpt_dir, opt.exp_name, "last.ckpt"))
    model_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                        if k.startswith('net.')}
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    valset = REDS_Val_Dataset(root_dir=opt.val_dir, bundle_frame=opt.num_frames, downsample_ratio=4, args=opt)
    valset.get_srgb = True
    valset.get_srgb = True
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    print("--------> Inference on testset.")
    print("\n")

    run_inference(opt, model, valloader)


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

if __name__ == '__main__':
    train_opt = train_options()

    if hasattr(train_opt, 'exp_name') and train_opt.exp_name:
        train_opt = load_hparams_to_opt(train_opt, train_opt.exp_name)

    main(train_opt)