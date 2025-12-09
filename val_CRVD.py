import os
import pathlib
import numpy as np

from tqdm import tqdm
from skimage.util import img_as_ubyte

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from net.moce_jdd import MoCEJDD
from net.fastDVDnet_mini import FastDVDnet
from options import train_options
from utils.test_utils import save_img
from data.datasets_v0 import REDS_Val_Dataset
from data.crvd_dataset import CRVDataset
import cv2

from data.Hamilton_Adam_demo import HamiltonAdam
from matplotlib import pyplot as plt
import random, yaml
from CRVD_ISP.ISP import ISP

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

def srgb_to_rggb(srgb_image_tensor):
    # 确保输入图像的尺寸是偶数
    h, w, _ = srgb_image_tensor.shape
    h, w = h // 2 * 2, w // 2 * 2
    srgb_image_tensor = srgb_image_tensor[:h, :w, :]


    R  = srgb_image_tensor[1::2, 0::2, 0]  # R 通道
    Gr = srgb_image_tensor[1::2, 1::2, 1]  # G 通道 (在 R 行)
    B  = srgb_image_tensor[0::2, 1::2, 2]  # B 通道
    Gb = srgb_image_tensor[0::2, 0::2, 1]  # G 通道 (在 B 行)

    packed_bayer = torch.stack([B, Gr, R, Gb], dim=0)


    return packed_bayer
def run_inference(opts, net,net_isp, dataloader):
    testloader = dataloader

    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"CRVD/{opts.checkpoint_id}")).mkdir(parents=True,
                                                                                    exist_ok=True)
    save_folder = os.path.join(os.getcwd(), f"CRVD/{opts.checkpoint_id}/{opts.exp_name}")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # print("save to ", save_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ham_demosaic = HamiltonAdam(pattern='gbrg').to(device)

    isp = ISP().cuda()

    isp.load_state_dict(torch.load('H:/My-JDD/CRVD_ISP/ISP_CNN_weights_only.pth'))
    for k, v in isp.named_parameters():
        v.requires_grad = False

    total_psnr_linear = []
    total_ssim_linear = []
    total_psnr_srgb = []
    total_ssim_srgb = []

    # 用于存储每个ISO级别的结果
    iso_results = {}

    psnr_linear = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_linear = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


    with torch.no_grad():
        for batch in tqdm(testloader):

            raw_noise = batch['noisy'].to(device)
            raw_clean = batch['clean'].to(device)



            # srgb_clean = srgb_clean[0].permute(1,2,0)
            # plt.imshow(srgb_clean.cpu())
            # plt.show()

            # Forward pass
            demosaic_raw = torch.zeros((raw_noise.shape[0], raw_noise.shape[1],
                                        3, raw_noise.shape[3] * 2, raw_noise.shape[4] * 2)).to(device)

            for bundle in range(raw_noise.shape[0]):
                demosaic_raw[bundle] = ham_demosaic(raw_noise[bundle])

            demosaic_clean = ham_demosaic(raw_clean[0][2].unsqueeze(0)).clamp(0, 1)

            restored = net(demosaic_raw)

            restored = restored.clamp(0, 1)

            # srgb_remosaic = srgb_to_rggb(restored[0].permute(1,2,0))
            # srgb_resored = isp(srgb_remosaic.unsqueeze(0)).clamp(0, 1).squeeze(0).permute(1,2,0)

            # 把顺序调整成isp模型的rgbg
            new_order_indices = [1, 3, 2, 0]
            noisy_rgbg = raw_noise[:,:,new_order_indices,...]

            srgb_clean = isp(raw_clean[0][2].unsqueeze(0)).clamp(0, 1).squeeze(0).permute(1,2,0)
            srgb_noisy = isp(noisy_rgbg[0][2].unsqueeze(0)).clamp(0, 1).squeeze(0).permute(1,2,0)

            # plt.imshow(srgb_noisy.cpu())
            # plt.show()

            # plt.imshow(srgb_clean.cpu())
            # plt.show()

            srgb_resored = net_isp(demosaic_raw).clip(0, 1)

            # plt.imshow(srgb_resored[0].permute(1,2,0).cpu())
            # plt.show()

            # 计算 linearRGB 的 PSNR 和 SSIM
            linear_psnr = psnr_linear(restored, demosaic_clean).item()
            linear_ssim = ssim_linear(restored, demosaic_clean).item()
            srgb_psnr = psnr_linear(srgb_resored, srgb_clean.permute(2,0,1).unsqueeze(0)).item()
            srgb_ssim = ssim_linear(srgb_resored, srgb_clean.permute(2,0,1).unsqueeze(0)).item()

            total_psnr_linear.append(linear_psnr)
            total_ssim_linear.append(linear_ssim)
            total_psnr_srgb.append(srgb_psnr)
            total_ssim_srgb.append(srgb_ssim)

            # Save results
            video_idx_str = batch['video_key'][0]
            frame_idx_str = int(batch['center_frame_num'][0].cpu())

            # 从 video_idx_str (例如 'scene1_iso1600') 中提取 ISO 级别
            iso_level = video_idx_str.upper().split('ISO')[-1]

            # 记录每个ISO级别的结果
            if iso_level not in iso_results:
                iso_results[iso_level] = {'psnr': [], 'ssim': [], 'srgb_psnr': [], 'srgb_ssim': []}
            iso_results[iso_level]['psnr'].append(linear_psnr)
            iso_results[iso_level]['ssim'].append(linear_ssim)
            iso_results[iso_level]['srgb_psnr'].append(srgb_psnr)
            iso_results[iso_level]['srgb_ssim'].append(srgb_ssim)

            noisy_folder = os.path.join(save_folder, 'noisy', video_idx_str)
            video_folder = os.path.join(save_folder, 'restored', video_idx_str)
            raw_clean_folder = os.path.join(save_folder, 'gt', video_idx_str)

            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            if not os.path.exists(raw_clean_folder):
                os.makedirs(raw_clean_folder)
            if not os.path.exists(noisy_folder):
                os.makedirs(noisy_folder)

            demosaic_raw = demosaic_raw.clamp(0, 1)
            # save_img(os.path.join(noisy_folder, f"{frame_idx_str}.png"),
            #          img_as_ubyte(demosaic_raw[0][2].permute(1, 2, 0).cpu()))
            #
            # save_img(os.path.join(video_folder, f"{frame_idx_str}.png"),
            #          img_as_ubyte(restored[0].permute(1, 2, 0).cpu()))
            #
            # save_img(os.path.join(raw_clean_folder, f"{frame_idx_str}.png"),
            #          img_as_ubyte(demosaic_clean[0].permute(1, 2, 0).cpu()))


            save_img(os.path.join(noisy_folder, f"{frame_idx_str}.png"),
                     img_as_ubyte(srgb_noisy.cpu()))

            save_img(os.path.join(video_folder, f"{frame_idx_str}.png"),
                     img_as_ubyte(srgb_resored[0].permute(1, 2, 0).cpu()))

            save_img(os.path.join(raw_clean_folder, f"{frame_idx_str}.png"),
                     img_as_ubyte(srgb_clean.cpu()))



    # --- 修改部分开始 ---
    # 计算并保存每个ISO级别的平均PSNR和SSIM
    results_file = os.path.join(os.getcwd(), f"CRVD/MoCE_JDD/{opts.exp_name}", "iso_results.tsv")

    print("\n" + "=" * 50)
    print("           Average Metrics per ISO")
    print("=" * 50)

    with open(results_file, 'w') as f:
        header = "ISO Level\tLinear PSNR\tLinear SSIM\tSRGB PSNR\tSRGB SSIM\n"
        f.write(header)
        print(header.strip())

        # 为了输出顺序一致，对ISO级别进行排序 (按数值大小)
        sorted_iso_levels = sorted(iso_results.keys(), key=int)

        for iso_level in sorted_iso_levels:
            values = iso_results[iso_level]
            avg_psnr = np.mean(values['psnr'])
            avg_ssim = np.mean(values['ssim'])
            avg_srgb_psnr = np.mean(values['srgb_psnr'])
            avg_srgb_ssim = np.mean(values['srgb_ssim'])
            result_line = f"ISO {iso_level}\t{avg_psnr:.4f}\t\t{avg_ssim:.4f}\t\t{avg_srgb_psnr:.4f}\t\t{avg_srgb_ssim:.4f}"
            f.write(f"{iso_level}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_srgb_psnr:.4f}\t{avg_srgb_ssim:.4f}\n")
            print(result_line)

        print("-" * 50)

        # 计算并保存总体平均值
        overall_linear_psnr = np.mean(total_psnr_linear)
        overall_linear_ssim = np.mean(total_ssim_linear)
        overall_srgb_psnr = np.mean(total_psnr_srgb)
        overall_srgb_ssim = np.mean(total_ssim_srgb)
        overall_line = f"Overall\t\t{overall_linear_psnr:.4f}\t\t{overall_linear_ssim:.4f}\t\t{overall_srgb_psnr:.4f}\t\t{overall_srgb_ssim:.4f}"

        f.write(f"\nOverall\t{overall_linear_psnr:.4f}\t{overall_linear_ssim:.4f}\t{overall_srgb_psnr:.4f}\t{overall_srgb_ssim:.4f}\n")
        print(overall_line)
        print("=" * 50 + "\n")

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


    model2 = MoCEJDD(
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

    checkpoint = torch.load(os.path.join(opt.ckpt_dir, 'MoCE_JDD', "last_isp.ckpt"))
    model_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                        if k.startswith('net.')}
    model2.load_state_dict(model_state_dict, strict=False)
    model2.eval()

    root_dir = r'G:\datasets\CRVD_dataset'
    valset = CRVDataset(root_dir)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    print("--------> Inference on testset.")
    print("\n")

    run_inference(opt, model,model2, valloader)


if __name__ == '__main__':
    train_opt = train_options()

    if hasattr(train_opt, 'exp_name') and train_opt.exp_name:
        train_opt = load_hparams_to_opt(train_opt, train_opt.exp_name)


    main(train_opt)