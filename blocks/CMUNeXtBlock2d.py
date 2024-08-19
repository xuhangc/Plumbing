import torch
import torch.nn as nn
#CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion
#https://arxiv.org/pdf/2308.01239
"""
conv_block 类：这是一个基本的卷积块，包括一个卷积层（Conv2d）、一个批量归一化层（BatchNorm2d）和一个ReLU激活函数。

Residual 类：实现了残差连接，即将输入加上函数fn的输出。

CMUNeXtBlock 类：这是CMUNeXt架构中的核心模块，它使用了深度级联的残差块和扩展卷积（利用groups参数实现深度卷积）。
每个残差块首先通过一个具有大卷积核的深度卷积进行特征提取，然后通过1x1卷积进行特征融合和通道数的增（ch_in到ch_in*4）减（ch_in*4到ch_in）。
最后，通过一个卷积块将特征图的通道数从ch_in增加到ch_out。

CMUNeXtBlock通过在深度卷积层之间添加残差连接来增强特征的复用，同时使用1x1卷积调整通道数，通过这种方式提高网络的表示能力，同时保持计算的高效性。
此外，大卷积核的使用能够扩大感受野，更好地捕捉医学图像中的大尺度结构。这种设计使得CMUNeXtBlock在处理医学图像分割任务时既能高效又能保持良好的性能。
"""
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    """
     depth: length of cmunext blocks
     k: kernal size of cmunext blocks
    """
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)
        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)#h和w减半

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        # x = self.Maxpool(x)
        return x


if __name__ == '__main__':
    # 实例化CMUNeXtBlock，设定输入通道数和输出通道数
    # 假设depth=2以增加网络深度，k设为3作为大卷积核的示例
    cmunext_block = CMUNeXtBlock(ch_in=64, ch_out=128, depth=2, k=3)

    # 创建一个模拟输入张量，例如：batch_size=1, 通道数=64, 图像尺寸为256x256
    input_tensor = torch.rand(1, 64, 256, 256)

    # 打印输入的shape
    print(f"输入的shape: {input_tensor.shape}")

    # 使用定义好的CMUNeXtBlock处理输入tensor
    output_tensor = cmunext_block(input_tensor)

    # 打印输出的shape
    print(f"输出的shape: {output_tensor.shape}")