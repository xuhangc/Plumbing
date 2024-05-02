import torch.nn as nn
import math
import torch
#GhostNet: More Features from Cheap Operations
"""
以下是代码的详细说明：
GhostModule类继承自nn.Module，表明它是一个PyTorch模块。
在构造函数__init__()中，参数inp和oup分别表示输入通道数和输出通道数。其他参数控制卷积层的设置，
如内核大小、步幅以及是否应用ReLU激活。
在构造函数内部，定义了两个连续的卷积块：
primary_conv：该块由一个卷积层、后接的批量归一化和ReLU激活（如果指定）组成。它负责处理输入张量。
cheap_operation：该块使用较少数量的输出通道执行额外的卷积操作。它用于生成“廉价”的额外特征以增强主要特征。
该块由一个深度卷积层、后接的批量归一化和ReLU激活（如果指定）组成。
forward()方法定义了模块的前向传播。它将输入张量x通过primary_conv和cheap_operation块。
这些块的输出沿着通道维度（dim=1）进行拼接，并且仅保留前oup个通道。
该操作有效地将主要特征与cheap_operation生成的“廉价”特征组合在一起。
在if __name__ == "__main__":块中，创建了一个GhostModule实例，输入和输出通道大小为128，内核大小为3。
然后，创建了一个样本输入张量，形状为(2, 128, 64, 64)，表示具有128个通道和64x64空间维度的2个图像批次。
将样本输入张量通过GhostModule实例，并打印输出张量的形状。
打印的输出形状将为(2, 128, 64, 64)，表示一个批次包含2个图像，每个图像有128个通道，空间维度为64x64。
"""
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size,
                      stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


if __name__ == "__main__":
    model = GhostModule(128, 256)

    in_tensor = torch.zeros((2, 128, 64, 64))

    out_tensor = model(in_tensor)

    print(out_tensor.shape)