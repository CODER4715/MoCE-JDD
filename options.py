import argparse
import os
import pathlib
from typing import Tuple

# Helpers
def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def base_parser():
    parser = argparse.ArgumentParser()

    # Basic training settings
    parser.add_argument('--model', type=str, required=False, default='MoCE_JDD')
    parser.add_argument('--exp_name', type=str, default=None, help='logs and ckpts dir name.')
    parser.add_argument('--num_frames', type=str, default=5, help='')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size per GPU.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--patch_size', type=int, default=128, help='Input patch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--accum_grad', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint.')
    parser.add_argument('--fine_tune_from', type=str, default=None, help='Fine-tune from checkpoint.')
    parser.add_argument('--save_results', action="store_true", help="Save restored outputs.")
    parser.add_argument('--trans12bit', action="store_true", help="Transfer raw data to 12 bit.(240-4095)")

    # Model settings
    parser.add_argument('--gt_type', type=str, default='linear_rgb', help='linear_rgb/srgb')
    parser.add_argument('--isp_bwd', action="store_true", help='if need isp in backword.')

    #BasicLoss
    parser.add_argument('--loss_type', default="L1", help='Loss type.   L1/L2/multi_vgg/vgg')
    parser.add_argument('--loss_fn', default="L1", help='Loss fn type.')
    parser.add_argument('--pixel_loss_weight', type=float, default=1.0, help='FFT loss weight.')

    # balance_loss_weight
    parser.add_argument('--balance_loss_weight', type=float, default=0.1)


    # fft loss
    parser.add_argument('--fft_loss', action="store_true")
    parser.add_argument('--fft_loss_fn', default="L1", help='FFT Loss fn type.')
    parser.add_argument('--fft_loss_weight', type=float, default=0.1, help='FFT loss weight.')

    # vgg loss
    parser.add_argument('--vgg_layers', type=int, nargs='+', default=[3, 8, 13],
                        help="VGG19 layer indices for perceptual loss (default: relu1_2,relu2_2,relu3_2)")
    parser.add_argument('--vgg_layer_weights', type=float, nargs='+', default=[0.1, 0.6, 0.3],
                        help="Weights for each VGG layer (should match vgg_layers length)")
    parser.add_argument('--vgg_loss_weight', type=float, default=0.5,
                        help="Weight for VGG perceptual loss")

    # MS-SSIM loss
    parser.add_argument('--ms_ssim_loss', action="store_true",
                        help="Use MS-SSIM loss")
    parser.add_argument('--ms_ssim_weight', type=float, default=0.5,
                        help="Weight for MS-SSIM loss component")
    parser.add_argument('--ms_ssim_levels', type=int, default=5,
                        help="Number of scales for MS-SSIM")
    parser.add_argument('--ms_ssim_window_size', type=int, default=11,
                        help="Window size for MS-SSIM")

    # PSNR loss
    parser.add_argument('--psnr_loss', action="store_true",
                        help="Use PSNR loss")
    parser.add_argument('--psnr_weight', type=float, default=0.005,
                        help="Weight for PSNR loss component")

    # aux L2 loss
    parser.add_argument('--aux_l2_loss', action="store_true")
    parser.add_argument('--aux_l2_weight', type=float, default=0.1, help='Aux L2 loss weight.')

    # cosine loss
    parser.add_argument('--cosine_loss', action="store_true")
    parser.add_argument('--cosine_weight', type=float, default=0.5, help='Cosine loss weight.')


    # Paths
    parser.add_argument('--train_dir', type=str, default='H:/datasets/JDD/REDS120/train/train_orig', help='Path to datasets.')
    parser.add_argument('--val_dir', type=str, default='H:/datasets/JDD/REDS120/val/val_orig', help='Path to val datasets.')
    parser.add_argument('--output_path', type=str, default="results", help='Output save path.')
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help='Checkpoint directory.')
    parser.add_argument('--checkpoint_id', type=str, default='MoCE_JDD', help='checkpoint id')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for training.')

    #inference
    parser.add_argument('--use_tiling', action='store_true', help='Use tiled inference with blending for large images to save memory.')

    return parser


def moce_jdd(parser):

    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[2, 4, 6, 8])
    parser.add_argument('--num_dec_blocks', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--num_exp_blocks', type=int, default=4)
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', nargs='+', type=int, default=[2, 2, 4, 8])
    parser.add_argument('--stage_depth', nargs='+', type=int, default=[1, 1, 1])
    parser.add_argument('--with_complexity', action="store_true")
    parser.add_argument('--complexity_scale', type=str, default="max")
    parser.add_argument('--rank_type', default="spread")
    parser.add_argument('--depth_type', default="constant")
    parser.add_argument('--topk', type=int, default=1)
    return parser

def train_options():
    base_args = base_parser().parse_known_args()[0]

    if base_args.model == "MoCE_JDD":
        parser = moce_jdd(base_parser())

    else:
        raise NotImplementedError(f"Model '{base_args.model}' not found.")

    options = parser.parse_args()



    # Adjust batch size if gradient accumulation is used
    if options.accum_grad > 1:
        options.batch_size = options.batch_size // options.accum_grad

    return options