import torch.nn as nn
import torch.nn.functional as F
import torch
#DSNet: A Novel Way to Use Atrous Convolutions in Semantic Segmentation
#https://arxiv.org/pdf/2406.03702
#Multi-Scale Attention Fusion Module(MSAF): Balancing the Details and Contexts
"""
初始化（__init__） - 模块使用表示输入特征数量的通道数初始化，并使用减少因子r来计算用于注意力机制中的降维inter_channels。

局部注意力（local_att） - 包含一系列操作，这些操作减少通道数，应用规范化，使用ReLU激活函数，将通道数扩展回去，并再次应用规范化。这旨在专注于捕获局部细节。

上下文子模块（context1、context2、context3） - 这些块旨在通过不同尺度（适应性池化到不同大小）后面跟着与local_att相似的操作序列来捕获更广泛的上下文信息。
其思想是从不同接收场（4x4x4、8x8x8和16x16x16）获取上下文信息。

全局注意力（global_att） - 该序列类似于自注意力机制，其中应用了全局平均池化以考虑整个上下文，随后是类似于local_att的一系列操作。

Sigmoid激活（sigmoid） - 将用于生成权重系数，这些系数介于[0, 1]之间，用于应用注意力。

前向方法（forward） - 在前向传播中，输入通过注意力机制进行处理。注意力模块被应用到与残差连接结合的输入上。
各种上下文尺度被插值回原始大小并组合。通过所有这些注意力机制的sigmoid激活组合应用最终的权重。
输出结合了经过注意力应用的特征和残差，可能为下游任务（如体积分割）提供了精炼的特征表示。
"""
class MSAF3D(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSAF3D, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool3d((8, 8, 8)),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels)
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool3d((16, 16, 16)),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        d, h, w = x.shape[2], x.shape[3], x.shape[4]

        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        c1 = F.interpolate(c1, size=[d, h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[d, h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[d, h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


if __name__ == '__main__':
    # 实例化MSAF3D模块
    msaf3d = MSAF3D(channels=64, r=4)

    # 创建一个示例输入张量和残差张量
    input_tensor = torch.randn(2, 64, 32, 64, 64)  # 假定的输入，考虑到3D的情景
    residual_tensor = torch.randn(2, 64, 32, 64, 64)  # 假定的残差，大小与输入相同

    # 获取并打印输出形状
    output_tensor = msaf3d(input_tensor, residual_tensor)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)