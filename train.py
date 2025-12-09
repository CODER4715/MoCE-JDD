
import os
import pathlib
import numpy as np

from datetime import datetime

import random
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from net.moce_jdd import MoCEJDD
from net.moce_jdd_isp import MoCEJDD_ISP

from data.datasets import FixedFramesDataset, VariableFramesDataset, CombinedDataset
from data.datasets_v0 import REDS_Val_Dataset
from data.Hamilton_Adam_demo import HamiltonAdam
from data.Batch_Hamilton_Adam import BatchHamiltonAdam
from data.unprocessor import IspProcessor

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.loss_utils import VGG19Loss, MSSSIMLoss, PSNRLoss, MultiLayerVGGLoss, FFTLoss, CosineSimilarityLoss

from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio as psnr
from lightning.pytorch import seed_everything
import signal

from matplotlib import pyplot as plt


class InterruptCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.original_handler = None

    def on_train_start(self, trainer, pl_module):
        def handle_interrupt(signum, frame):
            print("\n捕获到中断信号，正在保存模型...")
            trainer.save_checkpoint(os.path.join(self.save_path, "last.ckpt"))
            print(f"模型已保存到: {os.path.join(self.save_path, 'last.ckpt')}")
            if self.original_handler is not None:
                self.original_handler(signum, frame)
            exit(0)

        self.original_handler = signal.signal(signal.SIGINT, handle_interrupt)

    def on_train_end(self, trainer, pl_module):
        if self.original_handler is not None:
            signal.signal(signal.SIGINT, self.original_handler)

class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.temp_files = []  # 存储临时文件路径
        self.save_hyperparameters(opt)
        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 定义临时文件夹路径
        self.temp_dir = os.path.join(current_dir, "temp")
        # 创建临时文件夹，如果不存在
        pathlib.Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # self.ham_demosaic = HamiltonAdam(pattern='rggb')
        self.ham_demosaic = BatchHamiltonAdam(pattern='rggb')

        if opt.gt_type == 'srgb':
            self.net = MoCEJDD_ISP(
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
            )

        else:
            self.net = MoCEJDD(
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
            )

        self.isp = IspProcessor()

        if opt.loss_fn == "L1":
            self.loss_fn = nn.L1Loss()
            print("Using L1 loss")
        elif opt.loss_fn == "L2":
            self.loss_fn = nn.MSELoss()
            print("Using MSE Loss")
        else:
            raise ValueError(f"不支持的损失函数类型: {opt.loss_fn}")

        print(f"Pixel loss weight: {opt.pixel_loss_weight}")

        if opt.loss_type == "vgg":
            # Initialize VGG19 loss
            self.vgg_loss = VGG19Loss(feature_layer=35)  # Using relu5_4 features
            print("Using VGG19 perceptual loss")
            print(f"VGG loss weight: {opt.vgg_loss_weight}")

        elif opt.loss_type == "multi_vgg":
            # Initialize multi-layer VGG loss
            self.vgg_loss = MultiLayerVGGLoss(
                feature_layers=opt.vgg_layers ,
                layer_weights=opt.vgg_layer_weights if hasattr(opt, 'vgg_layer_weights') else None
            )
            print("Using multi-layer VGG perceptual loss")
            print(f"VGG loss weight: {opt.vgg_loss_weight}")

        if  opt.fft_loss:
            self.fft_loss = FFTLoss(loss_weight=self.opt.fft_loss_weight, fft_loss_type=self.opt.fft_loss_fn)
            print("Using FFT loss")
            print(f"FFT loss weight: {opt.fft_loss_weight}")
            print(f"FFT loss fn: {opt.fft_loss_fn}")

        if opt.ms_ssim_loss:
            self.ms_ssim_fn = MSSSIMLoss(
                levels=opt.ms_ssim_levels,
                window_size=opt.ms_ssim_window_size
            )
            print("Using MS-SSIM loss")
            print(f"MS-SSIM loss weight: {opt.ms_ssim_weight}")

        if opt.psnr_loss:
            self.psnr_fn = PSNRLoss()
            print("Using PSNR loss")
            print(f"PSNR loss weight: {opt.psnr_weight}")

        if opt.aux_l2_loss:
            self.aux_l2_loss = nn.MSELoss()
            print("Using Aux L2 loss")
            print(f"Aux L2 loss weight: {opt.aux_l2_weight}")

        if opt.cosine_loss:
            self.cosine_loss = CosineSimilarityLoss()
            print("Using Cosine loss")
            print(f"Cosine loss weight: {opt.cosine_weight}")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):

        device = self.device
        # 确保VGG处于eval模式
        if hasattr(self, 'vgg_loss'):
            self.vgg_loss.eval()

        # raw_clean = batch['raw_clean']
        raw_noise = batch['raw_noise'].to(device)
        linear_RGB = batch['linear_RGB'].to(device)
        srgb_gt = batch['srgb'].to(device)
        metadata = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['metadata'].items()}

        # demosaic_raw = torch.zeros((raw_noise.shape[0], raw_noise.shape[1],
        #                             linear_RGB.shape[1], linear_RGB.shape[2], linear_RGB.shape[3])).to(device)
        # for bundle in range(raw_noise.shape[0]):
        #     demosaic_raw[bundle] = self.ham_demosaic(raw_noise[bundle])

        demosaic_raw = self.ham_demosaic(raw_noise)


        restored = self.net(demosaic_raw, metadata)

        if self.opt.gt_type == 'linear_rgb':
            gt_img = linear_RGB
        elif self.opt.gt_type == 'srgb':
            gt_img = srgb_gt
        else:
            raise ValueError(f"不支持的gt_type: {self.opt.gt_type}")

        loss = self.loss_fn(restored, gt_img) * self.opt.pixel_loss_weight
        pixel_loss = loss

        balance_loss = self.net.total_loss  # balance_loss是专家的损失
        balance_loss = self.balance_loss_weight * balance_loss
        loss = loss + balance_loss
        self.log("Train_expert_loss", balance_loss, sync_dist=True)


        if self.opt.loss_type == "vgg" or self.opt.loss_type == "multi_vgg":
            vgg_loss = self.vgg_loss(restored, gt_img)
            vgg_loss = self.opt.vgg_loss_weight * vgg_loss
            loss = loss + vgg_loss

            # Log individual layer losses if available
            if hasattr(self.vgg_loss, 'current_layer_losses'):
                for name, val in self.vgg_loss.current_layer_losses.items():
                    self.log(f"Train_{name}", val, sync_dist=True)
            self.log("Train_VGG_Loss", vgg_loss, sync_dist=True)

        if self.opt.fft_loss:
            fft_loss = self.fft_loss(restored, gt_img)
            loss = loss + fft_loss
            self.log("Train_FFT_Loss", fft_loss, sync_dist=True)

        # 添加MS-SSIM损失
        if self.opt.ms_ssim_loss:
            ms_ssim_loss = self.ms_ssim_fn(restored, gt_img)
            ms_ssim_loss = self.opt.ms_ssim_weight * ms_ssim_loss
            loss = loss + ms_ssim_loss
            self.log("Train_MS_SSIM_Loss", ms_ssim_loss, sync_dist=True)

        if self.opt.psnr_loss:
            psnr_loss = self.psnr_fn(restored, gt_img)
            psnr_loss = self.opt.psnr_weight * psnr_loss
            loss = loss + psnr_loss
            self.log("Train_PSNR_Loss", psnr_loss, sync_dist=True)

        if self.opt.aux_l2_loss:
            aux_l2_loss = self.aux_l2_loss(restored, gt_img)
            aux_l2_loss = self.opt.aux_l2_weight * aux_l2_loss
            loss = loss + aux_l2_loss
            self.log("Train_Aux_L2_Loss", aux_l2_loss, sync_dist=True)

        if self.opt.cosine_loss:
            cosine_loss = self.cosine_loss(restored, gt_img)
            cosine_loss = self.opt.cosine_weight * cosine_loss
            loss = loss + cosine_loss
            self.log("Train_Cosine_Loss", cosine_loss, sync_dist=True)

        self.log("Train_Pixel({})_Loss".format(self.opt.loss_fn), pixel_loss, sync_dist=True)
        self.log("Train_Loss", loss, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, sync_dist=True)

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=int(self.opt.epochs*0.15),
                                                  max_epochs=self.opt.epochs,
                                                  warmup_start_lr=0.1 * self.opt.lr)


        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=1,
                                                      max_epochs=self.opt.epochs)

        return [optimizer], [scheduler]


    def on_validation_start(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):

        device = self.device

        raw_noise = batch['raw_noise']
        linear_RGB = batch['linear_RGB']
        srgb_gt = batch['srgb']
        metadata = {k: v if isinstance(v, torch.Tensor) else v for k, v in batch['metadata'].items()}

        # demosaic_raw = torch.zeros((raw_noise.shape[0], raw_noise.shape[1],
        #                             linear_RGB.shape[1], linear_RGB.shape[2],linear_RGB.shape[3])).to(device)
        # for bundle in range(raw_noise.shape[0]):
        #     demosaic_raw[bundle] = self.ham_demosaic(raw_noise[bundle])

        demosaic_raw = self.ham_demosaic(raw_noise)


        restored = self.net(demosaic_raw, metadata)

        if self.opt.gt_type == 'linear_rgb':
            gt_img = linear_RGB
        elif self.opt.gt_type == 'srgb':
            gt_img = srgb_gt
        else:
            raise ValueError(f"不支持的gt_type: {self.opt.gt_type}")

        loss = self.loss_fn(restored, gt_img)
        pixel_loss = loss

        balance_loss = self.net.total_loss  # balance_loss是专家的损失
        balance_loss = self.balance_loss_weight * balance_loss
        loss = loss + balance_loss

        if self.opt.loss_type == "vgg" or self.opt.loss_type == "multi_vgg":
            vgg_loss = self.vgg_loss(restored, gt_img)
            vgg_loss = self.opt.vgg_loss_weight * vgg_loss
            loss = loss + vgg_loss

            # Log individual layer losses if available
            if hasattr(self.vgg_loss, 'current_layer_losses'):
                for name, val in self.vgg_loss.current_layer_losses.items():
                    self.log(f"Train_{name}", val, sync_dist=True)

        if self.opt.fft_loss:
            fft_loss = self.fft_loss(restored, gt_img)
            loss = loss + fft_loss

        if self.opt.ms_ssim_loss:
            ms_ssim_loss = self.ms_ssim_fn(restored, gt_img)
            ms_ssim_loss = self.opt.ms_ssim_weight * ms_ssim_loss
            loss = loss + ms_ssim_loss

        if self.opt.psnr_loss:
            psnr_loss = self.psnr_fn(restored, gt_img)
            psnr_loss = self.opt.psnr_weight * psnr_loss
            loss = loss + psnr_loss

        if self.opt.aux_l2_loss:
            aux_l2_loss = self.aux_l2_loss(restored, gt_img)
            aux_l2_loss = self.opt.aux_l2_weight * aux_l2_loss
            loss = loss + aux_l2_loss

        if self.opt.cosine_loss:
            cosine_loss = self.cosine_loss(restored, gt_img)
            cosine_loss = self.opt.cosine_weight * cosine_loss
            loss = loss + cosine_loss

        # 计算 PSNR
        if self.opt.trans12bit:
            restored = (restored - 240) / (4095 - 240)
            restored = restored.clamp(0, 1)
            gt_img = (gt_img - 240) / (4095 - 240)
            gt_img = gt_img.clamp(0, 1)

        psnr_value = psnr(gt_img.cpu().numpy(), restored.cpu().numpy(), data_range=1)

        if self.opt.gt_type == 'linear_rgb':

            srgb_restored = self.isp.process(restored, metadata['red_gain'],
                                           metadata['blue_gain'],
                                           metadata['cam2rgb'],
                                           metadata['rgb_gain'], dem=False)


            srgb_restored = srgb_restored.permute(0,3,1,2)

        else:

            srgb_restored = restored

        output = {
            "loss": loss,
            "psnr": psnr_value,
            "pixel_loss": pixel_loss,
            "balance_loss": balance_loss,
            "vgg_loss": vgg_loss if hasattr(self, 'vgg_loss') else None,
            "fft_loss": fft_loss if hasattr(self, 'fft_loss') else None,
            "ms_ssim_loss": ms_ssim_loss if hasattr(self, 'ms_ssim_fn') else None,
            "psnr_loss": psnr_loss if hasattr(self, 'psnr_fn') else None,
            "aux_l2_loss": aux_l2_loss if hasattr(self, 'aux_l2_loss') else None,
            "cosine_loss": cosine_loss if hasattr(self, 'cosine_loss') else None,
            "srgb_gt": srgb_gt.cpu(),
            "srgb_restored": srgb_restored.cpu(),
            "frame_info": batch['frame_idx'],
        }

        # 以 batch['frame_idx']['video']_batch['frame_idx']['frame'] 命名临时文件
        video_idx = batch['frame_idx']['video'].item()
        frame_idx = batch['frame_idx']['frame'].item()
        file_name = f"video_{video_idx}-frame_{frame_idx}.pth"
        temp_file_path = os.path.join(self.temp_dir, file_name)
        with open(temp_file_path, 'wb') as f:
            torch.save(output, f)
        self.temp_files.append(temp_file_path)

    def on_validation_epoch_end(self):

        torch.cuda.empty_cache()

        num_frames_per_clip = 166
        clip_frames = []
        clip_idx = 0

        # 初始化用于累加损失和 PSNR 的变量
        total_loss = 0
        total_psnr = 0
        total_balance_loss = 0
        total_vgg_loss = []
        total_fft_loss = []
        total_ms_ssim_loss = []
        total_psnr_loss = []
        total_aux_l2_loss = []
        total_pixel_loss = 0
        num_samples = 0

        frames_to_write = 30

        for i, file_path in enumerate(self.temp_files):
            with open(file_path, 'rb') as f:
                output = torch.load(f, weights_only=False)

            clean_img = output["srgb_gt"]
            restored_img = output["srgb_restored"]

            # 确保图像是 3D 张量 [C, H, W]
            if len(clean_img.shape) == 4:  # 如果有 batch 维度 [1, C, H, W]
                clean_img = clean_img.squeeze(0)
                restored_img = restored_img.squeeze(0)

            # 将 clean 和 restored 图像水平拼接
            comparison = torch.cat([clean_img, restored_img], dim=-1)  # 结果形状 [C, H, W*2]
            clip_frames.append(comparison)

            # 累加损失和 PSNR
            total_loss += output["loss"]
            total_psnr += output["psnr"]
            total_balance_loss += output["balance_loss"]
            total_pixel_loss += output["pixel_loss"]
            num_samples += 1

            if output["vgg_loss"] is not None:
                total_vgg_loss.append(output["vgg_loss"])
            if output["fft_loss"] is not None:
                total_fft_loss.append(output["fft_loss"])
            if output["ms_ssim_loss"] is not None:
                total_ms_ssim_loss.append(output["ms_ssim_loss"])
            if output["psnr_loss"] is not None:
                total_psnr_loss.append(output["psnr_loss"])
            if output["aux_l2_loss"] is not None:
                total_aux_l2_loss.append(output["aux_l2_loss"])



            # 每 num_frames_per_clip 帧写入一次视频
            if (i + 1) % num_frames_per_clip == 0 or i == len(self.temp_files) - 1:
                if clip_frames:

                    frames_to_use = clip_frames[:frames_to_write]
                    # 将帧列表堆叠成一个 4D 张量 [T, C, H, W]
                    clip_video = torch.stack(frames_to_use).unsqueeze(0)  # 添加 batch 维度 [1, T, C, H, W]

                    # 将视频写入 TensorBoard
                    if self.logger:
                        writer = self.logger.experiment
                        writer.add_video(
                            f"val_clip_{clip_idx}",
                            clip_video,
                            global_step=self.global_step,
                            fps=30
                        )
                        print('video write ok!')

                        # 从 clip_frames 中随机挑选 3 张图片
                        if len(clip_frames) >= 3:
                            random_frames = random.sample(clip_frames, 3)
                        else:
                            random_frames = clip_frames
                        # 将挑选的图片堆叠成一个张量
                        frame_grid = torch.stack(random_frames)
                        # 使用 make_grid 将图片拼接成网格图
                        grid = make_grid(frame_grid, nrow=1)
                        # 将网格图写入 TensorBoard
                        if self.logger:
                            writer = self.logger.experiment
                            writer.add_image(
                                f"val_frames_clip_{clip_idx}",
                                grid,
                                global_step=self.global_step
                            )
                            print('image write ok!')
                            clip_frames = []

                    clip_idx += 1


        self.temp_files = []  # 清空临时文件列表

        # 计算平均损失和 PSNR
        avg_loss = total_loss / num_samples
        avg_psnr = total_psnr / num_samples
        avg_balance_loss = total_balance_loss / num_samples
        avg_pixel_loss = total_pixel_loss / num_samples

        avg_vgg_loss = torch.tensor(total_vgg_loss).mean() if total_vgg_loss else torch.tensor(0.0)
        avg_fft_loss = torch.tensor(total_fft_loss).mean() if total_fft_loss else torch.tensor(0.0)
        avg_ms_ssim_loss = torch.tensor(total_ms_ssim_loss).mean() if total_ms_ssim_loss else torch.tensor(0.0)
        avg_psnr_loss = torch.tensor(total_psnr_loss).mean() if total_psnr_loss else torch.tensor(0.0)
        avg_aux_l2_loss = torch.tensor(total_aux_l2_loss).mean() if total_aux_l2_loss else torch.tensor(0.0)

        # 记录每个损失
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_psnr", avg_psnr, sync_dist=True)
        self.log("val_balance_loss", avg_balance_loss, sync_dist=True)
        self.log("val_pixel_loss", avg_pixel_loss, sync_dist=True)
        if hasattr(self, 'vgg_loss'):
            self.log("val_vgg_loss", avg_vgg_loss, sync_dist=True)
        if hasattr(self, 'fft_loss'):
            self.log("val_fft_loss", avg_fft_loss, sync_dist=True)
        if hasattr(self, 'ms_ssim_fn'):
            self.log("val_ms_ssim_loss", avg_ms_ssim_loss, sync_dist=True)
        if hasattr(self, 'psnr_fn'):
            self.log("val_psnr_loss", avg_psnr_loss, sync_dist=True)
        if hasattr(self, 'aux_l2_loss'):
            self.log("val_aux_l2_loss", avg_aux_l2_loss, sync_dist=True)


        # 清空 temp 文件夹
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        # print(f"已删除文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 时出错: {e}")


def main(opt):
    # print("Options")
    # print(opt)
    seed_everything(42)
    if opt.exp_name is None:
        time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
        time_stamp = opt.exp_name

    torch.set_float32_matmul_precision('high')

    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        if opt.loss_type == "vgg" or opt.loss_type == "multi_vgg":
            model = PLTrainModel.load_from_checkpoint(
                os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)
        else:
            model = PLTrainModel.load_from_checkpoint(
                os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt, strict=False)
    else:
        model = PLTrainModel(opt)

    # print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, every_n_train_steps=5000,
                                          save_top_k=3, monitor='val_psnr', mode='max', save_last=True)
    interrupt_callback = InterruptCallback(checkpoint_path)  # 中断回调


    # trainset = REDS_Train_Dataset(root_dir=opt.train_dir, patch_size=opt.patch_size, bundle_frame=opt.num_frames,
    #                               downsample_ratio=2, args=opt)
    # trainset.get_srgb = True
    #
    # # 混合训练集
    # red_dataset = REDS_Train_Dataset(root_dir=opt.train_dir, patch_size=opt.patch_size, bundle_frame=opt.num_frames,
    #                               downsample_ratio=2, args=opt)
    # tvd_dataset = TVD(root_dir=r'F:\datasets\Tencent_Video_Dataset\Video\frames', bundle_frame=opt.num_frames, patch_size=opt.patch_size)
    # red_dataset.get_srgb = True
    # tvd_dataset.get_srgb = True
    #
    # trainset = CombinedDataset([tvd_dataset, red_dataset])

    # 创建 REDS 训练集
    reds_train_dataset = FixedFramesDataset(
        root_dir=opt.train_dir,
        bundle_frame=opt.num_frames,
        patch_size=opt.patch_size,
        downsample_ratio=2,
        is_train=True,
        get_srgb=True
    )
    print(f"REDS Train dataset length: {len(reds_train_dataset)}")


    # 创建 TVD 数据集
    tvd_dataset = VariableFramesDataset(
        root_dir='F:/datasets/Tencent_Video_Dataset/Video/frames',
        bundle_frame=opt.num_frames,
        patch_size=opt.patch_size,
        is_train=True,
        get_srgb=True
    )
    print(f"TVD dataset length: {len(tvd_dataset)}")

    trainset = CombinedDataset([tvd_dataset, reds_train_dataset])

    valset = REDS_Val_Dataset(root_dir='H:/datasets/JDD/REDS120/val/part', bundle_frame=opt.num_frames,
                              downsample_ratio=3, args=opt)
    valset.get_srgb = True

    # valset = FixedFramesDataset(
    #     root_dir=opt.val_dir,
    #     bundle_frame=opt.num_frames,
    #     is_train=False,  # 关键参数，设为False，则不进行patch crop
    #     downsample_ratio=3,
    #     seed=42,
    #     get_srgb=True
    # )
    print(f"REDS Val dataset length: {len(valset)}")

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True,
                             num_workers=opt.num_workers)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    # Create trainer
    trainer = pl.Trainer(max_epochs=opt.epochs,
                         accelerator="gpu",
                         devices=opt.num_gpus,
                         # strategy="ddp_find_unused_parameters_true",#
                         logger=logger,
                         callbacks=[checkpoint_callback, interrupt_callback],
                         accumulate_grad_batches=opt.accum_grad,
                         deterministic=False,
                         # check_val_every_n_epoch=4,
                         val_check_interval=0.5,
                         num_sanity_val_steps=opt.batch_size,
                         )

    # Optionally resume from a checkpoint
    if opt.resume_from:
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
    else:
        checkpoint_path = None

    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=checkpoint_path  # Specify the checkpoint path to resume from
    )


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)

