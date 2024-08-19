import torch
import torch.nn as nn
#FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
#https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848
"""
构造函数__init__
定义了两个卷积层conv_r1和conv_d1，用于处理RGB特征和深度特征，分别。
conv_c1卷积层用于将融合后的特征进一步处理。
conv_c2卷积层将处理后的特征映射到两个通道上，这两个通道代表了融合权重。
avgpool是一个自适应平均池化层，尽管在forward方法中没有使用。

fusion方法
输入包括两个特征f1和f2，以及一个特征向量f_vec（从conv_c2得到的融合权重）。
通过f_vec生成权重w1和w2，并使用这些权重对f1和f2进行加权融合，包括加法融合和乘法融合。
最终返回加法融合和乘法融合的结果的总和。

forward方法
首先，输入的RGB和深度特征分别通过conv_r1和conv_d1。
将这两个特征沿通道维拼接，然后通过conv_c1和conv_c2进行进一步处理。
使用conv_c2的输出作为fusion方法的融合权重对Fr和Fd进行融合。
返回融合后的输出特征Fo。
"""

class WCMF(nn.Module):
    def __init__(self,channel=256):
        super(WCMF, self).__init__()
        self.conv_r1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_d1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())

        self.conv_c1 = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def fusion(self,f1,f2,f_vec):

        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2
    def forward(self,rgb,depth):
        Fr = self.conv_r1(rgb)
        Fd = self.conv_d1(depth)
        f = torch.cat([Fr, Fd],dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        # f = self.avgpool(f)
        Fo = self.fusion(Fr, Fd, f)
        return Fo


if __name__ == '__main__':
    # 假定的输入参数
    batch_size = 1
    channels = 256
    height = 224
    width = 224

    # 实例化WCMF
    wcmf = WCMF(channel=channels)

    # 创建RGB和深度输入的假设张量
    rgb_input = torch.randn(batch_size, channels, height, width)
    depth_input = torch.randn(batch_size, channels, height, width)

    # 通过WCMF模型
    output = wcmf(rgb_input, depth_input)

    # 打印输入和输出的shape
    print("RGB 输入形状:", rgb_input.shape)
    print("深度 输入形状:", depth_input.shape)
    print("输出形状:", output.shape)