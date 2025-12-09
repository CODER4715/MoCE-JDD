
#--------------------------------------------------------------depcv--------------------------------
import torch
import torch.nn as nn

from data.unprocessor import IspProcessor

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CvBlock(nn.Module):
    '''(DepthwiseSeparableConv => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
       super(CvBlock, self).__init__()
       self.convblock = nn.Sequential(
          DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True)
       )

    def forward(self, x):
       return self.convblock(x)

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (DepthwiseSeparableConv => BN => ReLU)'''
    def __init__(self, num_in_frames, out_ch):
       super(InputCvBlock, self).__init__()
       self.interm_ch = 30
       self.convblock = nn.Sequential(
          nn.Conv2d(num_in_frames*3, num_in_frames*self.interm_ch, \
                  kernel_size=3, padding=1, groups=num_in_frames, bias=False), # Still a grouped conv for initial frame processing
          nn.BatchNorm2d(num_in_frames*self.interm_ch),
          nn.ReLU(inplace=True),
          DepthwiseSeparableConv(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True)
       )

    def forward(self, x):
       return self.convblock(x)

class DownBlock(nn.Module):
    '''Downscale + (DepthwiseSeparableConv => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch):
       super(DownBlock, self).__init__()
       self.convblock = nn.Sequential(
          # Using DepthwiseSeparableConv for downscaling. Stride is applied in depthwise part.
          DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          CvBlock(out_ch, out_ch) # CvBlock now uses DepthwiseSeparableConv
       )

    def forward(self, x):
       return self.convblock(x)

class UpBlock(nn.Module):
    '''(DepthwiseSeparableConv => BN => ReLU)*2 + Upscale'''
    def __init__(self, in_ch, out_ch):
       super(UpBlock, self).__init__()
       self.convblock = nn.Sequential(
          CvBlock(in_ch, in_ch), # CvBlock now uses DepthwiseSeparableConv
          # The final convolution before PixelShuffle is also replaced
          DepthwiseSeparableConv(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
          nn.PixelShuffle(2)
       )

    def forward(self, x):
       return self.convblock(x)

class OutputCvBlock(nn.Module):
    '''DepthwiseSeparableConv => BN => ReLU => DepthwiseSeparableConv'''
    def __init__(self, in_ch, out_ch):
       super(OutputCvBlock, self).__init__()
       self.convblock = nn.Sequential(
          DepthwiseSeparableConv(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(in_ch),
          nn.ReLU(inplace=True),
          DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
       )

    def forward(self, x):
       return self.convblock(x)

class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
       num_input_frames: int. number of input frames
    Inputs of forward():
       xn: input frames of dim [N, C, H, W], (C=3 RGB)
       noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=3):
       super(DenBlock, self).__init__()
       self.chs_lyr0 = 32
       self.chs_lyr1 = 48
       self.chs_lyr2 = 64

       self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
       self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
       self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
       self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
       self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
       self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

       self.reset_params()

    @staticmethod
    def weight_init(m):
       if isinstance(m, nn.Conv2d) or isinstance(m, DepthwiseSeparableConv):
          # Kaiming Normal for ReLU activation
          if hasattr(m, 'pointwise'): # For DepthwiseSeparableConv
             nn.init.kaiming_normal_(m.pointwise.weight, nonlinearity='relu')
             nn.init.kaiming_normal_(m.depthwise.weight, nonlinearity='relu')
          else: # For other Conv2d layers (like the grouped conv in InputCvBlock)
             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


    def reset_params(self):
       for _, m in enumerate(self.modules()):
          self.weight_init(m)

    def forward(self, in0, in1, in2):
       '''Args:
          inX: Tensor, [N, C, H, W] in the [0., 1.] range
          noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
       '''
       # Input convolution block
       x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
       # Downsampling
       x1 = self.downc0(x0)
       x2 = self.downc1(x1)
       # Upsampling
       x2 = self.upc2(x2)
       x1 = self.upc1(x1+x2)
       # Estimation
       x = self.outc(x0+x1)

       # Residual
       x = in1 - x

       return x

class FastDVDnet(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
       xn: input frames of dim [N, C, H, W], (C=3 RGB)
       noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, need_isp=False,):
       super(FastDVDnet, self).__init__()
       self.num_input_frames = num_input_frames
       # Define models of each denoising stage
       self.temp1 = DenBlock(num_input_frames=3)
       self.temp2 = DenBlock(num_input_frames=3)
       # Init weights
       self.reset_params()

       self.isp = IspProcessor()
       self.need_isp = need_isp

    @staticmethod
    def weight_init(m):
       if isinstance(m, nn.Conv2d) or isinstance(m, DepthwiseSeparableConv):
          if hasattr(m, 'pointwise'):
             nn.init.kaiming_normal_(m.pointwise.weight, nonlinearity='relu')
             nn.init.kaiming_normal_(m.depthwise.weight, nonlinearity='relu')
          else:
             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
       for _, m in enumerate(self.modules()):
          self.weight_init(m)

    def forward(self, x, metadata=None):
       '''Args:
          x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
          noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
       '''

       # Assuming x is [N, num_frames, C, H, W]
       x0 = x[:, 0, :, :, :]
       x1 = x[:, 1, :, :, :]
       x2 = x[:, 2, :, :, :]
       x3 = x[:, 3, :, :, :]
       x4 = x[:, 4, :, :, :]

       # First stage
       x20 = self.temp1(x0, x1, x2)
       x21 = self.temp1(x1, x2, x3)
       x22 = self.temp1(x2, x3, x4)

       #Second stage
       x = self.temp2(x20, x21, x22)

       if self.need_isp:
           x = self.isp.process(x, metadata['red_gain'],
                                metadata['blue_gain'],
                                metadata['cam2rgb'],
                                metadata['rgb_gain'], dem=False)
           x = x.permute(0, 3, 1, 2)

       return x

if __name__ == '__main__':

	with torch.no_grad():
		#test
		model = FastDVDnet().cuda()

		from calflops import calculate_flops
		# 定义多输入的参数，使用 args
		args = [
			torch.randn(1, 5, 3, 2160, 3840).cuda(), # N, num_frames, C, H, W
		]
		flops, macs, params = calculate_flops(
			model=model,
			args=args,
			output_as_string=True,
			output_precision=5
		)

		print('{:>16s} : {}'.format('FLOPs', flops))
		print('{:>16s} : {}'.format('MACs', macs))
		print('{:>16s} : {}'.format('Params', params))
		if type(flops) == int:
			print(flops-1099511627776)