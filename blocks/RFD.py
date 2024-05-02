import torch
import torch.nn as nn
from collections import namedtuple
Stream = namedtuple('Stream', ['ptr'])


"""鲁棒特征下采样模块
浅层 RFD（SRFD）：
浅层残差特征降采样（SRFD）模块旨在通过残差方式对特征进行降采样，既增强特征表征又减少空间维度。以下是其组成部分的详细介绍：

初始化卷积（conv_init）：

使用步幅为1的7x7卷积层开始。该层将输入通道数减少为用户指定的输出通道数的四分之一（out_c14）。
填充设置为保持空间维度。
降采样层（conv_1、conv_x1、cut_c、fusion1）：

在初始化之后，通过因子2进行降采样。
使用深度可分离卷积（conv_1）来降低计算成本。
然后，使用步幅为2的3x3卷积（conv_x1）进一步降低特征图的分辨率。
切割操作（cut_c）用于调整通道数和空间维度。
最后，特征融合（fusion1）将降采样后的特征与原始特征进行融合。
进一步降采样和融合（conv_2、conv_x2、max_m、cut_r、fusion2）：

接下来进行一组操作，将特征从2倍降采样到4倍。
使用卷积层（conv_2、conv_x2）和最大池化（max_m）进行降采样。
切割操作（cut_r）再次调整通道数和空间维度。
融合操作（fusion2）将来自不同路径的特征组合起来。
# ------------------------------------------------------------------------------------------------------------------------------

深度 RFD（DRFD）：
深度残差特征降采样（DRFD）模块旨在在保持特征丰富性的同时进行更深层次的特征降采样。以下是其主要特点：

初始化和降采样（cut_c、conv、conv_x）：

与SRFD类似，它首先使用切割操作（cut_c）进行通道调整。
然后，使用深度可分离卷积（conv）后跟步幅为2的3x3卷积（conv_x）进行降采样。
激活和归一化（act_x、batch_norm_x、max_m、batch_norm_m）：

应用激活函数（GELU）（act_x），然后进行批归一化（batch_norm_x）。
最大池化（max_m）进一步降低特征分辨率。
对特征进行批归一化（batch_norm_m）以稳定训练过程。
特征融合（fusion）：

将来自不同路径的特征连接起来，并通过1x1卷积层（fusion）进行融合。
两种模块都旨在高效地降采样特征同时保留重要信息。然而，SRFD更注重浅层特征处理，而DRFD则更深入地进行特征提取。每个模块都设计以适应不同的架构需求，并在计算复杂度和特征表示方面进行权衡。
"""

# original size to 4x downsampling layer
class SRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = int(out_channels / 4)  # out_channels / 4
        out_c12 = int(out_channels / 2)  # out_channels / 2

        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # original size to 2x downsampling layer
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        self.cut_c = Cut(out_c14, out_c12)
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # 2x to 4x downsampling layer
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(out_c12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]

    # original size to 2x downsampling layer
        c = x                   # c = [B, C/4, H, W]
        # CutD
        c = self.cut_c(c)       # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]
        # ConvD
        x = self.conv_1(x)      # x = [B, C/4, H, W] --> [B, C/2, H/2, W/2]
        x = self.conv_x1(x)     # x = [B, C/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        # Concat + conv
        x = torch.cat([x, c], dim=1)    # x = [B, C, H/2, W/2]
        x = self.fusion1(x)     # x = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]

    # 2x to 4x downsampling layer
        r = x                   # r = [B, C/2, H/2, W/2]
        x = self.conv_2(x)      # x = [B, C/2, H/2, W/2] --> [B, C, H/2, W/2]
        m = x                   # m = [B, C, H/2, W/2]
        # ConvD
        x = self.conv_x2(x)     # x = [B, C, H/4, W/4]
        x = self.batch_norm_x2(x)
        # MaxD
        m = self.max_m(m)       # m = [B, C, H/4, W/4]
        m = self.batch_norm_m(m)
        # CutD
        r = self.cut_r(r)       # r = [B, C, H/4, W/4]
        # Concat + conv
        x = torch.cat([x, r, m], dim=1)  # x = [B, C*3, H/4, W/4]
        x = self.fusion2(x)     # x = [B, C*3, H/4, W/4] --> [B, C, H/4, W/4]
        return x                # x = [B, C, H/4, W/4]


# CutD
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x



# Deep feature downsampling
class DRFD(nn.Module):
    def __init__(self,  in_channels=3, out_channels=96):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):       # input: x = [B, C, H, W]
        c = x                   # c = [B, C, H, W]
        x = self.conv(x)        # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x                   # m = [B, 2C, H, W]

        # CutD
        c = self.cut_c(c)       # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)      # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)       # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)      # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x                # x = [B, 2C, H/2, W/2]

if __name__ == '__main__':
    # block = SRFD()
    # block = DRFD()
    # Create a random input tensor with appropriate dimensions
    input = torch.rand(1, 3, 256, 256)
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())