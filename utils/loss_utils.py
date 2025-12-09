import math

import torch
import torch.nn as nn
from typing import Tuple
from torchvision.models import vgg19
import torch.nn.functional as F


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, fft_loss_type='L1', reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        if fft_loss_type == 'L1':
            self.criterion = torch.nn.L1Loss(reduction=reduction)  # 频域的L1损失
        elif fft_loss_type == 'L2':
            self.criterion = self.complex_mse_loss  # 频域的L2损失
        else:
            raise ValueError(f"Unsupported FFT loss type: {fft_loss_type}")

    def complex_mse_loss(self, pred, target):
        # 计算复数模的平方差
        return torch.mean(torch.abs(pred - target) ** 2)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class TemperatureScheduler:
    def __init__(self, start_temp, end_temp, total_steps):
        """
        Scheduler for Gumbel-Softmax temperature that decreases using a cosine annealing schedule.

        Args:
        - start_temp (float): Initial temperature (e.g., 5.0).
        - end_temp (float): Final temperature (e.g., 0.01).
        - total_steps (int): Total number of steps/epochs to anneal over.
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps

    def get_temperature(self, step):
        """
        Get the temperature value for the current step, following a cosine annealing schedule.

        Args:
        - step (int): Current step or epoch.

        Returns:
        - temperature (float): The temperature for the Gumbel-Softmax at this step.
        """
        if step >= self.total_steps:
            return self.end_temp

        # Cosine annealing formula to compute the temperature
        cos_inner = math.pi * step / self.total_steps
        # temp = self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + math.cos(cos_inner))
        temp = self.start_temp + 0.5 * (self.end_temp - self.start_temp) * (1 - math.cos(cos_inner))

        return temp


class VGG19Loss(nn.Module):
    def __init__(self, feature_layer=35, use_normalized_input=True):
        super(VGG19Loss, self).__init__()
        vgg = vgg19(pretrained=True).features[:feature_layer + 1].eval()
        self.vgg = vgg
        self.use_normalized_input = use_normalized_input

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # VGG normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # Normalize input if needed
        if self.use_normalized_input:
            input = (input + 1) / 2  # Scale from [-1,1] to [0,1]
            target = (target + 1) / 2

            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        # Extract features
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)

        # Calculate L1 loss between features
        return torch.mean(torch.abs(vgg_input - vgg_target))


class MultiLayerVGGLoss(nn.Module):
    def __init__(self,
                 feature_layers: Tuple[int] = (2, 7, 12, 21, 30),  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
                 use_normalized_input: bool = True,
                 layer_weights: Tuple[float] = None):
        super().__init__()

        # Load pretrained VGG19
        vgg = vgg19(pretrained=True).features.eval()
        # print(vgg)
        self.feature_layers = feature_layers
        self.use_normalized_input = use_normalized_input

        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad_(False)

        # Split VGG into segments for each feature layer
        self.vgg_layers = nn.ModuleList()
        prev_layer = 0
        for layer in feature_layers:
            self.vgg_layers.append(vgg[prev_layer:layer + 1])
            prev_layer = layer + 1

        # Set layer weights (default equal weights)
        if layer_weights is None:
            self.layer_weights = torch.ones(len(feature_layers)) / len(feature_layers)
        else:
            assert len(layer_weights) == len(feature_layers)
            self.layer_weights = torch.tensor(layer_weights)
            self.layer_weights /= self.layer_weights.sum()

        # VGG normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize input if needed
        if self.use_normalized_input:
            input = (input + 1) / 2  # Scale from [-1,1] to [0,1]
            target = (target + 1) / 2

            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        # Compute features and losses for each layer
        total_loss = 0.0
        input_features = input
        target_features = target

        for i, (layer, weight) in enumerate(zip(self.vgg_layers, self.layer_weights)):
            input_features = layer(input_features)
            target_features = layer(target_features)

            # Calculate L1 loss between features and weight it
            layer_loss = torch.mean(torch.abs(input_features - target_features))
            total_loss += weight * layer_loss

            # For debugging/logging individual layer losses
            if hasattr(self, 'current_layer_losses'):
                self.current_layer_losses[f'vgg_layer_{i}'] = layer_loss.detach()

        return total_loss


class MSSSIMLoss(nn.Module):
    def __init__(self, levels=5, window_size=11, channel=3):
        super().__init__()
        self.levels = levels
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    def _create_window(self, window_size, channel):
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                              for x in range(window_size)])
        gauss = gauss / gauss.sum()
        window = gauss.ger(gauss)
        window = window.unsqueeze(0).unsqueeze(0)  # [1,1,11,11]
        return window.repeat(channel, 1, 1, 1)  # [3,1,11,11]

    def _ssim(self, img1, img2, window):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 确保输入是4D [B,C,H,W]
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        # 计算均值
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, img1, img2):
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)

        msssim = torch.ones_like(img1[:, 0, 0, 0])
        for i in range(self.levels):
            if i > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2)
                img2 = F.avg_pool2d(img2, kernel_size=2)

            ssim_val = self._ssim(img1, img2, self.window)
            msssim = msssim * (ssim_val ** weights[i])

        return 1 - msssim.mean()


class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val  # 图像的最大像素值（如1.0或255）

    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse + 1e-10))
        return -psnr  # 取负值


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        """
        基于 torch.nn.CosineSimilarity 实现的余弦相似度损失模块。

        Args:
            dim (int): 计算余弦相似度的维度，默认为 1。
            eps (float): 用于数值稳定性的小常量，默认为 1e-8。
        """
        super(CosineSimilarityLoss, self).__init__()
        # 初始化 CosineSimilarity 模块
        self.cosine_similarity = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, input, target):
        """
        计算输入和目标之间的余弦相似度损失。

        Args:
            input (torch.Tensor): 输入张量。
            target (torch.Tensor): 目标张量。

        Returns:
            torch.Tensor: 余弦相似度损失。
        """
        # 计算余弦相似度
        cos_sim = self.cosine_similarity(input, target)
        # 计算损失，用 1 减去余弦相似度
        loss = 1 - cos_sim
        # 返回平均损失
        return loss.mean()


