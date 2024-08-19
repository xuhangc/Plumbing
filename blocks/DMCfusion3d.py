import torch.nn as nn
import torch.nn.functional as F
import torch

class DMC_fusion(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'trilinear', 'align_corners': True}):
        super(DMC_fusion, self).__init__()

        self.up_kwargs = up_kwargs

        self.conv4_2 = nn.Sequential(
            nn.Conv3d(in_channels[3], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(in_channels[2], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv3d(in_channels[2], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(in_channels[1], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv3d(in_channels[1], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels[0]),
            nn.ReLU(inplace=True))
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels[0], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels[0]),
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
    x1 = torch.rand(1, 64, 32, 128, 128)  # 最低层
    x2 = torch.rand(1, 128, 16, 64, 64)   # 中间层1
    x3 = torch.rand(1, 256, 8, 32, 32)    # 中间层2
    x4 = torch.rand(1, 512, 4, 16, 16)    # 最高层

    # 执行前向传播
    outputs = dmc_fusion_module(x1, x2, x3, x4)

    # 打印输入和输出的尺寸
    print(f'输入x1的尺寸: {x1.size()}')
    print(f'输入x2的尺寸: {x2.size()}')
    print(f'输入x3的尺寸: {x3.size()}')
    print(f'输入x4的尺寸: {x4.size()}')
    for i, output in enumerate(outputs, start=1):
        print(f'输出{i}的尺寸: {output.size()}')