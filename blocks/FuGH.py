import torch
import torch.nn as nn

# DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Indistinct-Boundary Object Segmentation
# https://arxiv.org/pdf/2311.00483
# Fourier Group Harmonics (FuGH) module
class FuGH(nn.Module):
    def __init__(self, channels, groups):
        super(FuGH, self).__init__()
        self.group_linear1 = nn.Conv3d(channels, channels, kernel_size=1, groups=groups)
        self.gelu = nn.GELU()
        self.group_linear2 = nn.Conv3d(channels, channels, kernel_size=1, groups=groups)

    def forward(self, x):
        x_fft = torch.fft.fftn(x, dim=(2, 3, 4))
        x_fft_real = torch.real(x_fft)
        x_fft_imag = torch.imag(x_fft)

        y_real = self.group_linear1(x_fft_real)
        y_real = self.gelu(y_real)
        y_real = self.group_linear2(y_real)
        y_real = y_real + x_fft_real

        y_imag = self.group_linear1(x_fft_imag)
        y_imag = self.gelu(y_imag)
        y_imag = self.group_linear2(y_imag)
        y_imag = y_imag + x_fft_imag

        y = torch.complex(y_real, y_imag)

        y_ifft = torch.fft.ifftn(y, dim=(2, 3, 4))
        y_ifft_real = y_ifft.real

        return y_ifft_real


if __name__ == '__main__':
    # 实例化 FuGH 模块
    channels = 64  # 输入通道数
    groups = 64  # 分组卷积的组数
    model = FuGH(channels, groups)

    # 创建一个示例输入张量
    input_tensor = torch.randn(1, channels, 32, 32, 32)  # (batch_size, channels, depth, height, width)

    # 打印输入的形状
    print("输入张量的形状:", input_tensor.shape)

    # 前向传播
    output_tensor = model(input_tensor)

    # 打印输出的形状
    print("输出张量的形状:", output_tensor.shape)