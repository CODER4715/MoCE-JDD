import torch
from torch import nn


class BatchHamiltonAdam(nn.Module):
    """
    一个向量化的、支持视频序列批处理的Hamilton-Adams去马赛克算法实现。

    该版本完整地封装了原始的、基于固定卷积的复杂Hamilton-Adams算法。
    它专门设计用于处理形状为 [B, N, 4, H, W] 的批次RAW视频序列，
    其中4个通道代表了Bayer 2x2单元中的R, G, G, B值。

    Args:
        pattern (str): Bayer CFA (Color Filter Array) 的模式。
                       支持 'rggb', 'grbg', 'gbrg', 'bggr'。
    """

    def __init__(self, pattern: str = 'rggb'):
        super().__init__()

        self.pattern = pattern.lower()

        # 使用字典缓存生成的mask，避免重复计算
        self.mem_mosaic_bayer_mask = {}
        self.mem_algo2_mask = {}

        # 算法1使用的卷积层 (用于绿色通道插值)
        self.conv_algo1 = nn.Sequential(
            nn.ReplicationPad2d((2, 2, 2, 2)),
            nn.Conv2d(1, 6, kernel_size=5, bias=False)
        )
        self.conv_algo1.requires_grad_(False)  # 权重是固定的，不参与训练
        self.init_algo1()

        # 算法2使用的卷积层 (用于红/蓝通道插值)
        self.conv_algo2_chan = nn.Sequential(
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 6, kernel_size=3, bias=False)
        )
        self.conv_algo2_green = nn.Sequential(
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 4, kernel_size=3, bias=False)
        )
        self.conv_algo2_chan.requires_grad_(False)
        self.conv_algo2_green.requires_grad_(False)
        self.init_algo2()

    def init_algo1(self):
        """初始化算法1的固定卷积核权重"""
        weight = torch.zeros(6, 1, 5, 5)
        # Kh, Kv, Deltah, Deltav, Diffh, Diffv
        weight[0, 0, 2, 1] = .5;
        weight[0, 0, 2, 3] = .5
        weight[1, 0, 1, 2] = .5;
        weight[1, 0, 3, 2] = .5
        weight[2, 0, 2, 0] = 1.;
        weight[2, 0, 2, 2] = -2.;
        weight[2, 0, 2, 4] = 1.
        weight[3, 0, 0, 2] = 1.;
        weight[3, 0, 2, 2] = -2.;
        weight[3, 0, 4, 2] = 1.
        weight[4, 0, 2, 1] = 1.;
        weight[4, 0, 2, 3] = -1.
        weight[5, 0, 1, 2] = 1.;
        weight[5, 0, 3, 2] = -1.
        self.conv_algo1[1].weight.data = weight

    def init_algo2(self):
        """初始化算法2的固定卷积核权重"""
        weight1 = torch.zeros(6, 1, 3, 3)
        weight2 = torch.zeros(4, 1, 3, 3)
        # Kh, Kv, Kp, Kn, Diffp, Diffn
        weight1[0, 0, 1, 0] = .5;
        weight1[0, 0, 1, 2] = .5
        weight1[1, 0, 0, 1] = .5;
        weight1[1, 0, 2, 1] = .5
        weight1[2, 0, 0, 0] = .5;
        weight1[2, 0, 2, 2] = .5
        weight1[3, 0, 0, 2] = .5;
        weight1[3, 0, 2, 0] = .5
        weight1[4, 0, 0, 0] = -1.;
        weight1[4, 0, 2, 2] = 1.
        weight1[5, 0, 0, 2] = -1.;
        weight1[5, 0, 2, 0] = 1.

        # Deltah, Deltav, Deltap, Deltan
        weight2[0, 0, 1, 0] = .25;
        weight2[0, 0, 1, 1] = -.5;
        weight2[0, 0, 1, 2] = .25
        weight2[1, 0, 0, 1] = .25;
        weight2[1, 0, 1, 1] = -.5;
        weight2[1, 0, 2, 1] = .25
        weight2[2, 0, 0, 0] = 1.;
        weight2[2, 0, 1, 1] = -2.;
        weight2[2, 0, 2, 2] = 1.
        weight2[3, 0, 0, 2] = 1.;
        weight2[3, 0, 1, 1] = -2.;
        weight2[3, 0, 2, 0] = 1.

        self.conv_algo2_chan[1].weight.data = weight1
        self.conv_algo2_green[1].weight.data = weight2

    def algo1(self, x_masked, green_mask):
        """绿色通道插值核心算法"""
        green_mask = green_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        rawq = x_masked.sum(1, keepdim=True)  # [B, 1, H, W]
        conv_rawq = self.conv_algo1(rawq)
        rawh = conv_rawq[:, 0] - conv_rawq[:, 2] / 4
        rawv = conv_rawq[:, 1] - conv_rawq[:, 3] / 4
        CLh = conv_rawq[:, 4].abs() + conv_rawq[:, 2].abs()
        CLv = conv_rawq[:, 5].abs() + conv_rawq[:, 3].abs()
        CLlocation = torch.sign(CLh - CLv)
        green = (1 + CLlocation) * rawv / 2 + (1 - CLlocation) * rawh / 2
        green = green.unsqueeze(1) * (1 - green_mask) + rawq * green_mask
        return green

    def algo2(self, green, x_chan, mask_ochan, mode):
        """红/蓝通道插值核心算法"""
        _, _, H, W = green.size()
        maskGr, maskGb = self.algo2_mask(H, W)
        maskGr, maskGb = maskGr.to(green.device), maskGb.to(green.device)
        mask_ochan = mask_ochan.to(green.device)
        maskGr, maskGb = maskGr.unsqueeze(0), maskGb.unsqueeze(0)
        mask_ochan = mask_ochan.unsqueeze(0)
        if mode == 2:
            maskGr, maskGb = maskGb, maskGr
        conv_mosaic = self.conv_algo2_chan(x_chan)
        conv_green = self.conv_algo2_green(green)
        Ch = maskGr * (conv_mosaic[:, 0] - conv_green[:, 0])
        Cv = maskGb * (conv_mosaic[:, 1] - conv_green[:, 1])
        Cp = mask_ochan * (conv_mosaic[:, 2] - conv_green[:, 2] / 4)
        Cn = mask_ochan * (conv_mosaic[:, 3] - conv_green[:, 3] / 4)
        CLp = mask_ochan * (conv_mosaic[:, 4].abs() + conv_green[:, 2].abs())
        CLn = mask_ochan * (conv_mosaic[:, 5].abs() + conv_green[:, 3].abs())
        CLlocation = torch.sign(CLp - CLn)
        chan = (1 + CLlocation) * Cn / 2 + (1 - CLlocation) * Cp / 2
        chan = (chan + Ch + Cv).unsqueeze(1) + x_chan
        return chan

    def algo2_mask(self, H, W):
        """生成并缓存算法2所需的mask"""
        code = (H, W, self.pattern)
        if code in self.mem_algo2_mask:
            return self.mem_algo2_mask[code]
        maskGr = torch.zeros(H, W)
        maskGb = torch.zeros(H, W)
        if self.pattern == 'grbg':
            maskGr[0::2, 0::2] = 1;
            maskGb[1::2, 1::2] = 1
        elif self.pattern == 'rggb':
            maskGr[0::2, 1::2] = 1;
            maskGb[1::2, 0::2] = 1
        elif self.pattern == 'gbrg':
            maskGb[0::2, 0::2] = 1;
            maskGr[1::2, 1::2] = 1
        elif self.pattern == 'bggr':
            maskGb[0::2, 1::2] = 1;
            maskGr[1::2, 0::2] = 1
        self.mem_algo2_mask[code] = (maskGr, maskGb)
        return maskGr, maskGb

    def mosaic_bayer_mask(self, H, W):
        """生成并缓存Bayer CFA的 R,G,B 位置mask"""
        code = (H, W, self.pattern)
        if code in self.mem_mosaic_bayer_mask:
            return self.mem_mosaic_bayer_mask[code]
        pattern_map = {'r': 0, 'g': 1, 'b': 2}
        num = [pattern_map[c] for c in self.pattern]
        mask = torch.zeros(3, H, W)
        mask[num[0], 0::2, 0::2] = 1
        mask[num[1], 0::2, 1::2] = 1
        mask[num[2], 1::2, 0::2] = 1
        mask[num[3], 1::2, 1::2] = 1
        self.mem_mosaic_bayer_mask[code] = mask
        return mask

    def forward(self, raw_batch: torch.Tensor) -> torch.Tensor:
        """
        对一个批次的4通道Packed RAW视频序列进行去马赛克处理。

        Args:
            raw_batch (torch.Tensor): 输入的RAW图像批次，
                                      形状为 [B, N, 4, H, W]。
        Returns:
            torch.Tensor: 输出的RGB图像批次，形状为 [B, N, 3, 2*H, 2*W]。
        """
        # 1. 获取原始维度并验证输入
        B_orig, N_orig, C_orig, H, W = raw_batch.shape
        if C_orig != 4:
            raise ValueError(f"输入通道数必须为4 (packed RGGB)，但接收到 {C_orig}")

        # 2. 将 [B, N, 4, H, W] -> [B*N, 4, H, W] 以便批处理
        x_4chan = raw_batch.view(B_orig * N_orig, 4, H, W)

        # 3. 将4通道packed格式解包为单通道CFA马赛克格式
        # 输出形状为 [B*N, 1, 2*H, 2*W]
        x_packed_cfa = torch.zeros(
            B_orig * N_orig, 1, 2 * H, 2 * W,
            dtype=x_4chan.dtype, device=x_4chan.device
        )
        x_packed_cfa[:, 0, 0::2, 0::2] = x_4chan[:, 0]  # R
        x_packed_cfa[:, 0, 0::2, 1::2] = x_4chan[:, 1]  # G
        x_packed_cfa[:, 0, 1::2, 0::2] = x_4chan[:, 2]  # G
        x_packed_cfa[:, 0, 1::2, 1::2] = x_4chan[:, 3]  # B

        # --- 从这里开始，我们复用原始代码的核心逻辑，处理双倍分辨率的CFA图像 ---

        # 4. 生成Bayer CFA的R,G,B三通道mask
        mask = self.mosaic_bayer_mask(2 * H, 2 * W).to(x_packed_cfa.device)

        # 5. 将单通道CFA数据分离到对应的R, G, B稀疏通道中
        x_masked = x_packed_cfa * mask.unsqueeze(0)

        # 6. 绿色通道插值 (算法1)
        green = self.algo1(x_masked, mask[1])

        # 7. 红色和蓝色通道插值 (算法2)
        red = self.algo2(green, x_masked[:, 0].unsqueeze(1), mask[2], 1)
        blue = self.algo2(green, x_masked[:, 2].unsqueeze(1), mask[0], 2)

        # 8. 合并三个通道得到RGB图像
        y = torch.cat((red, green, blue), 1)

        # 9. 将结果从 [B*N, 3, 2*H, 2*W] 恢复为 [B, N, 3, 2*H, 2*W]
        output = y.view(B_orig, N_orig, 3, 2 * H, 2 * W)

        return output


if __name__ == '__main__':
    # ================================================================
    #                  这是一个简单的使用和验证示例
    # ================================================================

    # 模拟输入参数
    batch_size = 4
    num_frames = 5
    height, width = 64, 64  # packed raw的高度和宽度

    # 创建一个模拟的、4通道Packed RGGB格式的RAW数据批次
    # 形状: [B, N, 4, H, W]
    dummy_raw_batch = torch.rand(batch_size, num_frames, 4, height, width)

    # 实例化我们的批处理去马赛克模块
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    batch_demosaic_module = BatchHamiltonAdam(pattern='rggb').to(device)
    dummy_raw_batch = dummy_raw_batch.to(device)

    print(f"\n输入张量形状: {dummy_raw_batch.shape}")

    # 执行前向传播
    with torch.no_grad():
        output_rgb_batch = batch_demosaic_module(dummy_raw_batch)

    print(f"输出张量形状: {output_rgb_batch.shape}")

    # 验证输出形状是否正确
    expected_shape = (batch_size, num_frames, 3, height * 2, width * 2)
    assert output_rgb_batch.shape == expected_shape, \
        f"形状不匹配! 期望得到 {expected_shape}, 实际得到 {output_rgb_batch.shape}"

    print("\n✅ 验证成功：输出形状符合预期！")
