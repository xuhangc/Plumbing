import torch.nn as nn
import torch.nn.functional as F
import torch
#SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation
#https://ieeexplore.ieee.org/document/9895210
# dense multiplicative connection fusion module
"""
DMC_fusion模块使用了序列块，包括卷积、批量标准化和ReLU激活。
它处理高级特征，并通过乘性交互和上采样（使用F.interpolate）逐步细化这些特征，以逐步融合跨尺度的信息。
"""

class DMC_fusion(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(DMC_fusion, self).__init__()

        self.up_kwargs = up_kwargs

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels[3], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3, x4):
        x4_1 = x4
        x4_2 = F.interpolate(self.conv4_2(x4_1), scale_factor=2, **self.up_kwargs)
        x3_1 = x4_2 * (self.conv3_1(x3))
        x3_2 = F.interpolate(self.conv3_2(x3_1), scale_factor=2, **self.up_kwargs)
        x2_1 = x3_2 * (self.conv2_1(x2))
        x2_2 = F.interpolate(self.conv2_2(x2_1), scale_factor=2, **self.up_kwargs)
        x1_1 = x2_2 * (self.conv1_1(x1))

        return x1_1, x2_1, x3_1, x4_1


# 测试代码
if __name__ == '__main__':
    # 假定的输入通道数
    in_channels = [64, 128, 256, 512]

    # 初始化DMC融合模块
    dmc_fusion_module = DMC_fusion(in_channels)

    # 创建假的多尺度特征图
    x1 = torch.rand(1, 64, 128, 128)  # 最低层
    x2 = torch.rand(1, 128, 64, 64)  # 中间层1
    x3 = torch.rand(1, 256, 32, 32)  # 中间层2
    x4 = torch.rand(1, 512, 16, 16)  # 最高层

    # 执行前向传播
    outputs = dmc_fusion_module(x1, x2, x3, x4)

    # 打印输入和输出的尺寸
    print(f'输入x1的尺寸: {x1.size()}')
    print(f'输入x2的尺寸: {x2.size()}')
    print(f'输入x3的尺寸: {x3.size()}')
    print(f'输入x4的尺寸: {x4.size()}')
    for i, output in enumerate(outputs, start=1):
        print(f'输出{i}的尺寸: {output.size()}')