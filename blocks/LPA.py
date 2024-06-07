import torch
import torch.nn as nn
#SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation
#https://ieeexplore.ieee.org/document/9895210
#local pyramid attention (LPA) module
"""
通道注意力模块：该模块专注于学习输入特征图的通道间关系。
它使用平均池化和最大池化操作捕获通道特征的不同方面，接着通过一系列的卷积操作和sigmoid激活函数学习一个通道注意力图。该注意力图随后用于重新校准输入的通道特征。

空间注意力模块：该模块旨在学习输入特征图内的空间依赖性。
它计算通道平均值和最大值，将它们合并以突出信息丰富的区域，然后应用卷积操作跟随一个sigmoid激活函数生成一个空间注意力图。这个图用于强调重要的空间特征，同时抑制不太有用的部分。

局部金字塔注意力（LPA）模块：LPA模块以金字塔方式整合以上两种注意力机制，以处理不同尺度和方面的特征，增强模型捕获细节和粗略信息的能力。
它将输入特征图分成更小的部分，分别应用通道和空间注意力机制，然后重新组合它们以生成增强的特征图。最后，通过加法将增强的特征图与原输入聚合，确保精炼的特征有效融合，用于后续的分割任务。
"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4
        x = x3 + x4
        return x


if __name__ == '__main__':
    # 假设的输入数据的维度
    batch_size = 1
    channels = 128
    height = 64
    width = 64

    # 初始化一个符合这些维度的随机张量作为输入
    input_tensor = torch.rand(batch_size, channels, height, width)

    # 实例化LPA模块
    lpa = LPA(in_channel=channels)

    # 执行前向传播
    output_tensor = lpa(input_tensor)

    # 打印输入和输出的shape
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")