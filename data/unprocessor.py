import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from data.Hamilton_Adam_demo import HamiltonAdam
from matplotlib import pyplot as plt

class ImageUnprocessor:
    """将sRGB图像转换为模拟原始数据并添加噪声的预处理类，支持GPU加速"""

    def __init__(self, device='cpu'):

        self.device = device

        # 定义常量并移至指定设备
        self.rgb2xyz = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32).to(self.device)

        # XYZ -> Camera 色彩校正矩阵
        self.xyz2cams = torch.tensor([
            [[1.0234, -0.2969, -0.2266],
             [-0.5625, 1.6328, -0.0469],
             [-0.0703, 0.2188, 0.6406]],

            [[0.4913, -0.0541, -0.0202],
             [-0.613, 1.3513, 0.2906],
             [-0.1564, 0.2151, 0.7183]],

            [[0.838, -0.263, -0.0639],
             [-0.2887, 1.0725, 0.2496],
             [-0.0627, 0.1427, 0.5438]],

            [[0.6596, -0.2079, -0.0562],
             [-0.4782, 1.3016, 0.1933],
             [-0.097, 0.1581, 0.5181]]
        ], dtype=torch.float32)

        self.rgb2xyz = self.rgb2xyz.to(self.device)
        self.xyz2cams = self.xyz2cams.to(self.device)


    def random_ccm(self):
        """生成随机的RGB->Camera色彩校正矩阵"""
        num_ccms = len(self.xyz2cams)
        weights = torch.rand((num_ccms, 1, 1), dtype=torch.float32, device=self.device) * (1e8 - 1e-8) + 1e-8
        weights_sum = torch.sum(weights, dim=0)
        xyz2cam = torch.sum(self.xyz2cams * weights, dim=0) / weights_sum

        # 计算RGB->Camera CCM
        rgb2cam = torch.matmul(xyz2cam, self.rgb2xyz)

        # 归一化每行
        rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
        return rgb2cam

    def get_rgb2cam(self):
         # come from the authors of CRVD.
        return self.rgb2cam

    def random_gains(self):
        """生成随机增益用于亮度和白平衡"""
        # RGB增益表示亮度
        mean = torch.tensor(0.8, device=self.device)
        std = torch.tensor(0.1, device=self.device)
        rgb_gain = 1.0 / torch.normal(mean=mean, std=std, size=())

        # 红色和蓝色增益表示白平衡
        red_gain = torch.rand((), device=self.device) * (2.4 - 1.9) + 1.9
        blue_gain = torch.rand((), device=self.device) * (1.9 - 1.5) + 1.5
        return rgb_gain, red_gain, blue_gain

    def inverse_smoothstep(self, image):
        """近似反转全局色调映射曲线"""
        image = torch.clamp(image, 0.0, 1.0)
        return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)

    def gamma_expansion(self, image):
        """从gamma空间转换到线性空间"""
        # 钳位以防止接近零处的梯度数值不稳定
        return torch.max(image, torch.tensor(1e-8, dtype=image.dtype, device=self.device)) ** 2.2

    def apply_ccm(self, image, ccm):
        """应用色彩校正矩阵"""
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=([-1], [-1]))
        return image.view(shape)

    def safe_invert_gains(self, image, rgb_gain, red_gain, blue_gain):
        """反转增益，同时安全处理饱和像素"""
        one = torch.tensor(1.0, dtype=red_gain.dtype, device=self.device)
        gains = torch.stack([1.0 / red_gain, one, 1.0 / blue_gain]) / rgb_gain
        gains = gains[None, None, :].to(self.device)
        safe_gains = gains

        # # 通过在白色附近平滑遮罩增益，防止饱和像素变暗
        # gray = torch.mean(image, dim=-1, keepdim=True)
        # inflection = 0.9
        # mask = (torch.max(gray - inflection, torch.tensor(0.0, dtype=image.dtype, device=self.device)) / (
        #             1.0 - inflection)) ** 2.0
        # safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)

        return image * safe_gains

    def mosaic(self, image):
        """从RGB图像中提取RGGB拜耳模式"""
        assert image.dim() == 3 and image.size(-1) == 3, "输入必须是[H,W,3]的RGB图像"
        red = image[0::2, 0::2, 0] #R
        green_red = image[0::2, 1::2, 1] #G
        green_blue = image[1::2, 0::2, 1] #G
        blue = image[1::2, 1::2, 2] #B
        image = torch.stack((red, green_red, green_blue, blue), dim=-1)
        return image

    def random_noise_levels(self):
        """从对数-对数线性分布生成随机噪声水平"""
        log_min_shot_noise = torch.log(torch.tensor(0.0001, dtype=torch.float32, device=self.device))
        log_max_shot_noise = torch.log(torch.tensor(0.012, dtype=torch.float32, device=self.device))
        log_shot_noise = torch.rand((), device=self.device) * (
                    log_max_shot_noise - log_min_shot_noise) + log_min_shot_noise
        shot_noise = torch.exp(log_shot_noise)

        line = lambda x: 2.18 * x + 1.20
        log_read_noise = line(log_shot_noise) + torch.normal(mean=0.0, std=0.26, size=(), device=self.device)
        read_noise = torch.exp(log_read_noise)
        return shot_noise, read_noise

    def add_noise(self, image, shot_noise=0.01, read_noise=0.0005):
        """添加随机散粒噪声(与图像成正比)和读取噪声(独立)"""
        variance = image * shot_noise + read_noise
        try:
            noise = torch.normal(mean=torch.zeros_like(variance), std=torch.sqrt(variance))
        except RuntimeError as e:
            print(f"Error adding noise: {e}")
            if any(item is None for item in image.flatten()):
                raise ValueError("输入的 image 数组中包含 None 元素")
            if any(item is None for item in variance.flatten()):
                raise ValueError("输入的 variance 数组中包含 None 元素")
            print(f"Image shape: {image.shape}, Shot noise: {shot_noise}, Read noise: {read_noise}")
            print('Min variance:', torch.min(variance))
            noise = torch.zeros_like(variance)
        return image + noise

    def unprocess(self, image, rgb2cam=None, rgb_gain=None, red_gain=None, blue_gain=None):
        """将sRGB图像转换为模拟原始数据（支持传入自定义参数）"""
        assert image.dim() == 3 and image.size(-1) == 3, "输入必须是[H,W,3]的RGB图像"

        # 确保图像在正确的设备上
        image = image.to(self.device)

        # 使用传入参数或随机生成参数
        if rgb2cam is None:
            rgb2cam = self.random_ccm()

        if None in (rgb_gain, red_gain, blue_gain):
            rgb_gain, red_gain, blue_gain = self.random_gains()

        # 近似反转全局色调映射
        image = self.inverse_smoothstep(image)
        # 反转gamma压缩
        image = self.gamma_expansion(image)
        # 反转色彩校正
        image = self.apply_ccm(image, rgb2cam)
        # 近似反转白平衡和亮度(这一步让画面整体偏绿)
        image = self.safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
        # 钳位饱和像素
        image = torch.clamp(image, 0.0, 1.0)
        linear_RGB = image
        # 应用拜耳马赛克
        image = self.mosaic(image)

        metadata = {
            'rgb2cam': rgb2cam,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
        }
        return image, linear_RGB, metadata

    def forward(self, image, add_noise=True, shot_noise=None, read_noise=None,
                rgb2cam=None, rgb_gain=None,red_gain=None, blue_gain=None):
        """处理图像的主函数"""
        noise_imgage = None
        if add_noise and (shot_noise is None or read_noise is None):
            shot_noise, read_noise = self.random_noise_levels()

        # if rgb_gain is None or red_gain is None or blue_gain is None:
        #     rgb_gain, red_gain, blue_gain = self.random_gains()
        # if rgb2cam is None:
        #     rgb2cam = self.random_ccm()

        image, linear_RGB, metadata = self.unprocess(image, rgb2cam, rgb_gain, red_gain, blue_gain)

        if add_noise:

            noise_imgage = self.add_noise(image, shot_noise, read_noise)
            metadata['shot_noise'] = shot_noise
            metadata['read_noise'] = read_noise

        return image, noise_imgage, linear_RGB, metadata


class IspProcessor(nn.Module):
    def __init__(self, device='cuda'):
        super(IspProcessor, self).__init__()
        self.device = device
        self.ham_dem = HamiltonAdam(pattern='rggb').to(device)

    def apply_gains(self, bayer_images, red_gains, blue_gains):
        """Applies white balance gains to Bayer images."""
        green_gains = torch.ones_like(red_gains, device=self.device)
        gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=-1)
        gains = gains[:, None, None, :]
        return bayer_images * gains

    def apply_gains_v2(self, images, rgb_gains, red_gains, blue_gains):
        """Applies white balance gains to RGB images."""
        gains = torch.stack([torch.tensor(1.0) / (red_gains * rgb_gains), torch.tensor(1.0) / rgb_gains,
                             torch.tensor(1.0) / (blue_gains * rgb_gains)],dim=-1).to(self.device)
        gains = gains[:, None, None, :]
        gains = gains.to(self.device)

        # H, W, _ = image.shape
        # mask = torch.ones(H, W, 1)
        # mask = mask.cuda()

        # safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
        # out   = image / safe_gains
        out = images / gains
        return out


    def demosaic(self, bayer_images):
        """Bilinearly demosaics a batch of RGGB Bayer images using PyTorch."""
        assert bayer_images.ndim == 4, "Input must be a 4D tensor: [batch, height, width, 4]"

        batch_size, height, width, _ = bayer_images.shape
        output_height, output_width = height * 2, width * 2

        # 提取RGGB通道
        red = bayer_images[..., 0:1]
        green_red = bayer_images[..., 1:2]
        green_blue = bayer_images[..., 2:3]
        blue = bayer_images[..., 3:4]

        # 对红色通道进行上采样
        red_upsampled = F.interpolate(red.permute(0, 3, 1, 2), size=(output_height, output_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # 对绿色通道进行处理
        green_red_upsampled = F.interpolate(green_red.permute(0, 3, 1, 2), size=(output_height, output_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        green_blue_upsampled = F.interpolate(green_blue.permute(0, 3, 1, 2), size=(output_height, output_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # 创建绿色通道
        green = torch.zeros((batch_size, output_height, output_width, 1), device=self.device)

        # 在红色像素位置插值绿色值
        green[:, 0::2, 0::2, 0] = (green_red_upsampled[:, 0::2, 1::2, 0] + green_blue_upsampled[:, 1::2, 0::2, 0]) / 2

        # 在绿色(红色对角线)像素位置使用绿色值
        green[:, 0::2, 1::2, 0] = green_red_upsampled[:, 0::2, 1::2, 0]

        # 在绿色(蓝色对角线)像素位置使用绿色值
        green[:, 1::2, 0::2, 0] = green_blue_upsampled[:, 1::2, 0::2, 0]

        # 在蓝色像素位置插值绿色值
        green[:, 1::2, 1::2, 0] = (green_red_upsampled[:, 1::2, 0::2, 0] + green_blue_upsampled[:, 0::2, 1::2, 0]) / 2

        # 对蓝色通道进行上采样
        blue_upsampled = F.interpolate(blue.permute(0, 3, 1, 2), size=(output_height, output_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # 合并RGB通道
        rgb_images = torch.cat([red_upsampled, green, blue_upsampled], dim=-1)

        return rgb_images

    def apply_ccms(self, images, ccms):
        """Applies color correction matrices."""
        images = images.unsqueeze(-2)
        ccms = ccms.unsqueeze(1).unsqueeze(1)
        return torch.sum(images * ccms, dim=-1)

    def gamma_compression(self, images, gamma=2.2):
        """Converts from linear to gamma space."""
        return torch.max(images, torch.tensor(1e-8, device=self.device, dtype=images.dtype)) ** (1.0 / gamma)

    def process(self, images, red_gains, blue_gains, cam2rgbs, rgb_gains, dem=True):
        """Full processing pipeline from Bayer to sRGB."""

        images = images.to(self.device)
        red_gains = red_gains.to(self.device)
        blue_gains = blue_gains.to(self.device)
        cam2rgbs = cam2rgbs.to(self.device)
        rgb_gains = rgb_gains.to(self.device)

        if images.shape[3] > 4:
            images = images.permute(0,2,3,1)

        if dem:
            bayer_images = images
            # Clip values before demosaicing
            bayer_images = torch.clamp(bayer_images, 0.0, 1.0)
            # Demosaic
            # images = self.demosaic(bayer_images)
            # ham_dem
            bayer_images = bayer_images.permute(0, 3, 1, 2)
            images = self.ham_dem(bayer_images)
            images = images.permute(0, 2, 3, 1)

        # images = torch.clamp(images, 0.0, 1.0)

        # White balance
        # bayer_images = bayer_images.permute(0, 2, 3, 1)
        # bayer_images = self.apply_gains(bayer_images, red_gains, blue_gains)
        images = self.apply_gains_v2(images, rgb_gains, red_gains, blue_gains)

        # Color correction
        images = self.apply_ccms(images, cam2rgbs)

        # Gamma compression
        images = self.gamma_compression(images)

        # tone mapping
        images = 3 * images ** 2 - 2 * images ** 3

        images = torch.clamp(images, 0.0, 1.0)

        return images



def main1():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建处理器实例并移至指定设备
    unprocessor = ImageUnprocessor(device=device)
    isp_processor = IspProcessor()

    # 创建一个随机sRGB图像 [H,W,3] 并移至指定设备
    # image = torch.rand(512, 512, 3, device=device)

    img = cv2.imread('00000000.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    image = torch.from_numpy(img).to(device)

    # 处理图像
    raw_claen, raw_noise, linear_RGB, metadata = unprocessor.forward(image, add_noise=True)

    print(linear_RGB.shape)


    raw = raw_claen.unsqueeze(0)
    raw_noise = raw_noise.unsqueeze(0)
    cam2rgb = torch.inverse(metadata['rgb2cam'].cpu().unsqueeze(0))
    processed_img = isp_processor.process(raw, metadata['red_gain'].cpu().unsqueeze(0),
                            metadata['blue_gain'].cpu().unsqueeze(0),
                            cam2rgb, metadata['rgb_gain'].cpu().unsqueeze(0))

    cv2.imwrite('1.png', cv2.cvtColor(np.uint8(processed_img[0].cpu().numpy() * 255.0), cv2.COLOR_RGB2BGR))

    print(f"原始图像形状: {image.shape}")
    print(f"处理后图像形状: {processed_img.shape}")
    print(f"元数据包含: {list(metadata.keys())}")

def main2():
    imgs = torch.rand(1, 512, 512, 3)


# 使用示例
if __name__ == "__main__":
    main1()