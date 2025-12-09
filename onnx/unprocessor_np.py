import numpy as np
import cv2
from scipy.signal import convolve2d

class HamiltonAdam:
    """
    NumPy implementation of Hamilton-Adams demosaicing.
    J. Hamilton Jr. and J. Adams Jr. Adaptive color plan interpolation
    in single sensor color electronic camera, 1997, US Patent 5,629,734.
    """

    def __init__(self, pattern):
        """
        Initializes the demosaicing algorithm.
        Args:
            pattern (str): The Bayer pattern, e.g., 'rggb', 'grbg'.
        """
        if pattern not in ['rggb', 'grbg', 'gbrg', 'bggr']:
            raise ValueError("Unsupported Bayer pattern: {}".format(pattern))
        self.pattern = pattern

        # Caching for generated masks to avoid re-computation
        self.mem_mosaic_bayer_mask = {}
        self.mem_algo2_mask = {}

        # Initialize convolution kernels for algorithm 1 and 2
        self._init_kernels_algo1()
        self._init_kernels_algo2()

    def _init_kernels_algo1(self):
        """Initializes the 6 kernels for the green channel interpolation (Algorithm 1)."""
        kernels = np.zeros((6, 5, 5), dtype=np.float32)
        # Kh: Horizontal average
        kernels[0, 2, 1] = 0.5
        kernels[0, 2, 3] = 0.5
        # Kv: Vertical average
        kernels[1, 1, 2] = 0.5
        kernels[1, 3, 2] = 0.5
        # Deltah: Horizontal second derivative
        kernels[2, 2, 0] = 1.0
        kernels[2, 2, 2] = -2.0
        kernels[2, 2, 4] = 1.0
        # Deltav: Vertical second derivative
        kernels[3, 0, 2] = 1.0
        kernels[3, 2, 2] = -2.0
        kernels[3, 4, 2] = 1.0
        # Diffh: Horizontal difference
        kernels[4, 2, 1] = 1.0
        kernels[4, 2, 3] = -1.0
        # Diffv: Vertical difference
        kernels[5, 1, 2] = 1.0
        kernels[5, 3, 2] = -1.0
        self.kernels_algo1 = kernels

    def _init_kernels_algo2(self):
        """Initializes the kernels for red/blue channel interpolation (Algorithm 2)."""
        kernels_chan = np.zeros((6, 3, 3), dtype=np.float32)
        kernels_green = np.zeros((4, 3, 3), dtype=np.float32)

        # Kh
        kernels_chan[0, 1, 0] = 0.5
        kernels_chan[0, 1, 2] = 0.5
        # Kv
        kernels_chan[1, 0, 1] = 0.5
        kernels_chan[1, 2, 1] = 0.5
        # Kp
        kernels_chan[2, 0, 0] = 0.5
        kernels_chan[2, 2, 2] = 0.5
        # Kn
        kernels_chan[3, 0, 2] = 0.5
        kernels_chan[3, 2, 0] = 0.5
        # Diffp
        kernels_chan[4, 0, 0] = -1.0
        kernels_chan[4, 2, 2] = 1.0
        # Diffn
        kernels_chan[5, 0, 2] = -1.0
        kernels_chan[5, 2, 0] = 1.0

        # Deltah
        kernels_green[0, 1, 0] = 0.25
        kernels_green[0, 1, 1] = -0.5
        kernels_green[0, 1, 2] = 0.25
        # Deltav
        kernels_green[1, 0, 1] = 0.25
        kernels_green[1, 1, 1] = -0.5
        kernels_green[1, 2, 1] = 0.25
        # Deltap
        kernels_green[2, 0, 0] = 1.0
        kernels_green[2, 1, 1] = -2.0
        kernels_green[2, 2, 2] = 1.0
        # Deltan
        kernels_green[3, 0, 2] = 1.0
        kernels_green[3, 1, 1] = -2.0
        kernels_green[3, 2, 0] = 1.0

        self.kernels_algo2_chan = kernels_chan
        self.kernels_algo2_green = kernels_green

    def _convolve(self, images, kernels):
        """Helper to apply multiple convolutions to a batch of images."""
        B, H, W = images.shape
        C, kH, kW = kernels.shape

        # Pad images once before all convolutions
        pad_h, pad_w = kH // 2, kW // 2
        padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        output = np.zeros((B, C, H, W), dtype=images.dtype)
        for b in range(B):
            for c in range(C):
                output[b, c] = convolve2d(padded_images[b], kernels[c], mode='valid')
        return output

    def _algo1(self, x, green_mask):
        """Green channel interpolation."""
        # Get the raw CFA data (sum of R, G, B channels)
        rawq = np.sum(x, axis=1)  # [B, 2H, 2W]

        conv_rawq = self._convolve(rawq, self.kernels_algo1)  # [B, 6, 2H, 2W]

        rawh = conv_rawq[:, 0] - conv_rawq[:, 2] / 4
        rawv = conv_rawq[:, 1] - conv_rawq[:, 3] / 4
        CLh = np.abs(conv_rawq[:, 4]) + np.abs(conv_rawq[:, 2])
        CLv = np.abs(conv_rawq[:, 5]) + np.abs(conv_rawq[:, 3])

        # Directional interpolation based on gradients
        CLlocation = np.sign(CLh - CLv)
        green = (1 + CLlocation) * rawv / 2 + (1 - CLlocation) * rawh / 2

        # Combine interpolated green with original green pixels
        green = green * (1 - green_mask) + rawq * green_mask
        return green[:, np.newaxis, :, :]  # [B, 1, 2H, 2W]

    def _algo2(self, green, x_chan, mask_ochan, mode):
        """Red and Blue channel interpolation."""
        _, H, W = green.shape[1:]
        maskGr, maskGb = self._algo2_mask(H, W)

        if mode == 2:  # Swap masks for Blue channel processing
            maskGr, maskGb = maskGb, maskGr

        conv_mosaic = self._convolve(x_chan[:, 0], self.kernels_algo2_chan)
        conv_green = self._convolve(green[:, 0], self.kernels_algo2_green)

        Ch = maskGr * (conv_mosaic[:, 0] - conv_green[:, 0])
        Cv = maskGb * (conv_mosaic[:, 1] - conv_green[:, 1])
        Cp = mask_ochan * (conv_mosaic[:, 2] - conv_green[:, 2] / 4)
        Cn = mask_ochan * (conv_mosaic[:, 3] - conv_green[:, 3] / 4)
        CLp = mask_ochan * (np.abs(conv_mosaic[:, 4]) + np.abs(conv_green[:, 2]))
        CLn = mask_ochan * (np.abs(conv_mosaic[:, 5]) + np.abs(conv_green[:, 3]))

        CLlocation = np.sign(CLp - CLn)
        chan = (1 + CLlocation) * Cn / 2 + (1 - CLlocation) * Cp / 2
        chan = (chan + Ch + Cv) + x_chan[:, 0]
        return chan[:, np.newaxis, :, :]

    def _algo2_mask(self, H, W):
        """Generates masks for red/blue interpolation locations."""
        if (H, W) in self.mem_algo2_mask:
            return self.mem_algo2_mask[(H, W)]

        maskGr = np.zeros((H, W), dtype=np.float32)
        maskGb = np.zeros((H, W), dtype=np.float32)

        if self.pattern == 'grbg':
            maskGr[0::2, 0::2] = 1
            maskGb[1::2, 1::2] = 1
        elif self.pattern == 'rggb':
            maskGr[0::2, 1::2] = 1
            maskGb[1::2, 0::2] = 1
        elif self.pattern == 'gbrg':
            maskGb[0::2, 0::2] = 1
            maskGr[1::2, 1::2] = 1
        elif self.pattern == 'bggr':
            maskGb[0::2, 1::2] = 1
            maskGr[1::2, 0::2] = 1

        self.mem_algo2_mask[(H, W)] = (maskGr, maskGb)
        return maskGr, maskGb

    def _mosaic_bayer_mask(self, H, W):
        """Generates masks for R, G, B channels based on the Bayer pattern."""
        if (H, W) in self.mem_mosaic_bayer_mask:
            return self.mem_mosaic_bayer_mask[(H, W)]

        # Map pattern characters to channel indices (R=0, G=1, B=2)
        c_map = {'r': 0, 'g': 1, 'b': 2}

        mask = np.zeros((3, H, W), dtype=np.float32)
        mask[c_map[self.pattern[0]], 0::2, 0::2] = 1
        mask[c_map[self.pattern[1]], 0::2, 1::2] = 1
        mask[c_map[self.pattern[2]], 1::2, 0::2] = 1
        mask[c_map[self.pattern[3]], 1::2, 1::2] = 1

        self.mem_mosaic_bayer_mask[(H, W)] = mask
        return mask

    def _pack_in_one(self, x):
        """Packs 4 bayer channels into a single 2D image."""
        B, _, H, W = x.shape
        y = np.zeros((B, 2 * H, 2 * W), dtype=x.dtype)
        y[:, 0::2, 0::2] = x[:, 0]
        y[:, 0::2, 1::2] = x[:, 1]
        y[:, 1::2, 0::2] = x[:, 2]
        y[:, 1::2, 1::2] = x[:, 3]
        return y

    def __call__(self, x):
        """
        Executes the Hamilton-Adams demosaicing.
        Args:
            x (np.ndarray): Input Bayer image with shape (B, 4, H, W).
                            The 4 channels correspond to the RGGB (or other pattern) pixels.
        Returns:
            np.ndarray: Demosaiced RGB image with shape (B, 3, 2*H, 2*W).
        """
        B_orig, _, H, W = x.shape
        x = x.reshape(-1, 4, H, W)  # Handle potential extra dimensions

        # Pack the 4 channels into a single channel CFA mosaic
        x_packed = self._pack_in_one(x)  # [B, 2H, 2W]

        # Generate masks for R, G, B locations
        mask = self._mosaic_bayer_mask(2 * H, 2 * W)  # [3, 2H, 2W]

        # Create a masked version of the input for each channel
        # Broadcasting: (B, 1, 2H, 2W) * (1, 3, 2H, 2W) -> (B, 3, 2H, 2W)
        x_masked = x_packed[:, np.newaxis, :, :] * mask[np.newaxis, :, :, :]

        # 1. Green interpolation (Algorithm 1)
        green = self._algo1(x_masked, mask[1])

        # 2. Red and Blue demosaicing (Algorithm 2)
        red = self._algo2(green, x_masked[:, 0][:, np.newaxis], mask[2], 1)
        blue = self._algo2(green, x_masked[:, 2][:, np.newaxis], mask[0], 2)

        # 3. Stack channels to form the final RGB image
        y = np.concatenate((red, green, blue), axis=1)

        # Reshape to original batch size
        return y.reshape(B_orig, -1, 2 * H, 2 * W)


class ImageUnprocessor:
    """
    将sRGB图像转换为模拟原始数据并添加噪声的预处理类。
    (NumPy版本)
    """

    def __init__(self):
        # 定义常量
        self.rgb2xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)

        # XYZ -> Camera 色彩校正矩阵
        self.xyz2cams = np.array([
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
        ], dtype=np.float32)

        np.random.seed(42)

    def random_ccm(self):
        """生成随机的RGB->Camera色彩校正矩阵"""
        num_ccms = len(self.xyz2cams)
        weights = (np.random.rand(num_ccms, 1, 1) * (1e8 - 1e-8) + 1e-8).astype(np.float32)
        weights_sum = np.sum(weights, axis=0)
        xyz2cam = np.sum(self.xyz2cams * weights, axis=0) / weights_sum

        # 计算RGB->Camera CCM
        rgb2cam = xyz2cam @ self.rgb2xyz

        # 归一化每行
        rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
        return rgb2cam

    def random_gains(self):
        """生成随机增益用于亮度和白平衡"""
        # RGB增益表示亮度
        rgb_gain = 1.0 / np.random.normal(loc=0.8, scale=0.1)

        # 红色和蓝色增益表示白平衡
        red_gain = np.random.rand() * (2.4 - 1.9) + 1.9
        blue_gain = np.random.rand() * (1.9 - 1.5) + 1.5
        return np.float32(rgb_gain), np.float32(red_gain), np.float32(blue_gain)

    def inverse_smoothstep(self, image):
        """近似反转全局色调映射曲线"""
        image = np.clip(image, 0.0, 1.0)
        return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)

    def gamma_expansion(self, image):
        """从gamma空间转换到线性空间"""
        return np.maximum(image, 1e-8) ** 2.2

    def apply_ccm(self, image, ccm):
        """应用色彩校正矩阵"""
        shape = image.shape
        image = image.reshape(-1, 3)
        # torch.tensordot(image, ccm, dims=([-1], [-1])) is equivalent to image @ ccm.T
        image = image @ ccm.T
        return image.reshape(shape)

    def safe_invert_gains(self, image, rgb_gain, red_gain, blue_gain):
        """反转增益"""
        gains = np.array([1.0 / red_gain, 1.0, 1.0 / blue_gain], dtype=np.float32) / rgb_gain
        gains = gains[np.newaxis, np.newaxis, :]
        return image * gains

    def mosaic(self, image):
        """从RGB图像中提取RGGB拜耳模式"""
        red = image[0::2, 0::2, 0]
        green_red = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 2]
        # Stack along the last axis to create a 4-channel image
        return np.stack((red, green_red, green_blue, blue), axis=-1)

    def random_noise_levels(self):
        """从对数-对数线性分布生成随机噪声水平"""
        log_min_shot_noise = np.log(0.0001)
        log_max_shot_noise = np.log(0.012)
        log_shot_noise = np.random.rand() * (log_max_shot_noise - log_min_shot_noise) + log_min_shot_noise
        shot_noise = np.exp(log_shot_noise)

        line = lambda x: 2.18 * x + 1.20
        log_read_noise = line(log_shot_noise) + np.random.normal(loc=0.0, scale=0.26)
        read_noise = np.exp(log_read_noise)
        return np.float32(shot_noise), np.float32(read_noise)

    def add_noise(self, image, shot_noise=0.01, read_noise=0.0005):
        """添加随机散粒噪声和读取噪声"""
        variance = image * shot_noise + read_noise
        # Ensure variance is non-negative
        variance = np.maximum(variance, 0)
        noise = np.random.normal(loc=0.0, scale=np.sqrt(variance))
        return image + noise.astype(np.float32)

    def unprocess(self, image, rgb2cam=None, rgb_gain=None, red_gain=None, blue_gain=None):
        """将sRGB图像转换为模拟原始数据"""
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
        # 近似反转白平衡和亮度
        image = self.safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
        # 钳位饱和像素
        image = np.clip(image, 0.0, 1.0)
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
                rgb2cam=None, rgb_gain=None, red_gain=None, blue_gain=None):
        """处理图像的主函数"""
        noise_image = None
        if add_noise and (shot_noise is None or read_noise is None):
            shot_noise, read_noise = self.random_noise_levels()

        image, linear_RGB, metadata = self.unprocess(image, rgb2cam, rgb_gain, red_gain, blue_gain)

        if add_noise:
            noise_image = self.add_noise(image, shot_noise, read_noise)
            metadata['shot_noise'] = shot_noise
            metadata['read_noise'] = read_noise

        return image, noise_image, linear_RGB, metadata


class IspProcessor:
    """
    模拟ISP处理流程，将RAW图像转为sRGB图像。
    (NumPy版本)
    """

    def __init__(self):
        # 使用 'rggb' 模式，因为 ImageUnprocessor.mosaic 方法生成的是RGGB模式
        self.hamilton_demosaic = HamiltonAdam(pattern='rggb')

    def apply_gains_v2(self, images, rgb_gains, red_gains, blue_gains):
        """将白平衡增益应用于RGB图像"""
        gains = np.stack([1.0 / (red_gains * rgb_gains),
                          1.0 / rgb_gains,
                          1.0 / (blue_gains * rgb_gains)], axis=-1)
        gains = gains[:, np.newaxis, np.newaxis, :]
        # 注意：这里的操作是除法，等效于乘以增益的倒数
        # safe_invert_gains 是乘法，所以这里是除法，两者互为逆操作，逻辑正确
        return images / gains

    def apply_ccms(self, images, ccms):
        """应用色彩校正矩阵"""
        # 对批次中的每个 3x3 矩阵进行转置（交换轴1和轴2）
        ccms_transposed = ccms.transpose(0, 2, 1)
        # (B, H, W, 3) @ (B, 3, 3) -> (B, H, W, 3)
        return np.matmul(images, ccms_transposed)

    def gamma_compression(self, images, gamma=2.2):
        """从线性空间转换到gamma空间"""
        return np.maximum(images, 1e-8) ** (1.0 / gamma)

    def process(self, images, red_gains, blue_gains, cam2rgbs, rgb_gains, dem=True):
        """从Bayer到sRGB的完整处理流程"""
        if dem:
            bayer_images = images
            bayer_images = np.clip(bayer_images, 0.0, 1.0)

            # (B, H, W, 4) -> (B, 4, H, W)
            bayer_images_ch_first = bayer_images.transpose(0, 3, 1, 2)
            images_ch_first = self.hamilton_demosaic(bayer_images_ch_first)
            # (B, 3, H, W) -> (B, H, W, 3)
            images = images_ch_first.transpose(0, 2, 3, 1)

        # 白平衡和亮度
        images = self.apply_gains_v2(images, rgb_gains, red_gains, blue_gains)

        # 色彩校正
        images = self.apply_ccms(images, cam2rgbs)

        # Gamma 压缩
        images = self.gamma_compression(images)

        # 色调映射
        images = 3 * images ** 2 - 2 * images ** 3

        images = np.clip(images, 0.0, 1.0)

        return images

def main1():

    np.random.seed(42)

    # 创建处理器实例
    unprocessor = ImageUnprocessor()
    isp_processor = IspProcessor()

    # 读取图像并转换为 float32 [0, 1]范围的RGB图像
    try:
        img = cv2.imread('00000000.png')
        if img is None:
            raise FileNotFoundError("找不到图像文件 '00000000.png'。请确保文件在当前目录。")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = (img / 255.0).astype(np.float32)
    except Exception as e:
        print(f"读取图像时出错: {e}")
        # 如果读取失败，创建一个随机图像用于演示
        print("将使用随机图像进行演示。")
        image = np.random.rand(512, 512, 3).astype(np.float32)

    # 逆向ISP处理 (sRGB -> RAW)
    raw_clean, raw_noise, linear_RGB, metadata = unprocessor.forward(image, add_noise=True)

    # 为ISP处理准备数据 (增加batch维度)
    # 使用带噪声的RAW图进行处理
    raw_to_process = raw_noise[np.newaxis, ...]

    # 准备元数据，同样增加batch维度
    cam2rgb = np.linalg.inv(metadata['rgb2cam'][np.newaxis, ...])
    red_gain_b = np.array([metadata['red_gain']])
    blue_gain_b = np.array([metadata['blue_gain']])
    rgb_gain_b = np.array([metadata['rgb_gain']])

    # 正向ISP处理 (RAW -> sRGB)
    processed_img = isp_processor.process(raw_to_process,
                                          red_gain_b,
                                          blue_gain_b,
                                          cam2rgb,
                                          rgb_gain_b)

    # 保存结果图像
    # 从batch中取出第一张图，转换颜色空间并保存
    output_image_bgr = cv2.cvtColor(processed_img[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite('1.png', (output_image_bgr * 255.0).astype(np.uint8))

    print("处理完成，结果已保存为 '1.png'")
    print(f"原始图像形状: {image.shape}")
    print(f"生成的带噪RAW图像形状: {raw_noise.shape}")
    print(f"最终处理后图像形状: {processed_img.shape}")
    print(f"元数据包含: {list(metadata.keys())}")


# 使用示例
if __name__ == "__main__":
    main1()