import torch
from torch import nn
#Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
"""
cSE：执行通道方向的挤压和激励。
sSE：执行空间方向的挤压和激励。
scSE：结合了通道和空间方向的挤压和激励。
在 __main__ 块中，创建了一个 scSE 的实例，并将其应用于一个随机生成的输入张量，以演示输出形状。

下面是每个模块的简要摘要：

cSE：该模块首先在空间维度上计算全局平均池化，然后通过一个1x1卷积层和ReLU激活来产生通道方向的缩放因子。然后将该因子广播并应用于输入张量。

sSE：该模块对每个空间位置独立地应用一个1x1卷积层，然后通过sigmoid函数。输出用作空间注意力掩码，然后以元素方式应用于输入张量。

scSE：该模块结合了通道方向和空间方向的挤压和激励，通过将cSE和sSE模块都应用于输入张量，并取其输出的元素最大值。
"""

class cSE(nn.Module):

    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)

class sSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y

class scSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.cSE = cSE(in_channel)
        self.sSE = sSE(in_channel)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return torch.max(U_cse, U_sse)  # Taking the element-wise maximum


if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)
    model = scSE(in_channel=32)
    output = model(input)
    print(output.shape)