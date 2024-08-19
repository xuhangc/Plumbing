import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
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

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# Harmonic Squeeze-and-Excitation Module(HSE)
class HSE(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(dim*2, dim*2, kernel_size=3, padding=1, groups=dim*2),
            LayerNorm(dim*2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(dim*2, dim*2, kernel_size=3, padding=1, groups=dim*2),
        )
        self.block2 = nn.Sequential(
            FuGH(channels=dim, groups=dim),
            nn.Conv3d(dim, dim*2, kernel_size=7, stride=2, padding=3),
            LayerNorm(dim*2, eps=1e-6, data_format="channels_first")
        )
        self.se = SEBlock(dim*2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.block2(x)
        identity = x
        out = self.block(x)
        out = self.se(out)
        out = self.drop_path(out)
        out += identity
        return out


if __name__ == '__main__':
    # 实例化 HSE_conv 模块
    model = HSE(dim=16)

    # 创建一个示例输入张量
    input_tensor = torch.randn(1, 16, 64, 64, 64)  # (batch_size, channels, depth, height, width)

    # 打印输入的形状
    print("输入张量的形状:", input_tensor.shape)

    # 前向传播
    output_tensor = model(input_tensor)

    print(f"输出张量的形状:", output_tensor.shape)