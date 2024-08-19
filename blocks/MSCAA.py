import torch
import torch.nn as nn
#MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention
#https://arxiv.org/pdf/2312.08866
"""
第一个类MSCAA定义了核心的注意力机制。它创建了几个不同内核大小和填充的卷积层，以捕捉不同规模和方向的特征。
它使用分组卷积以通过将卷积分别应用于每个输入通道来保持计算效率。前向方法通过聚合这些不同卷积的输出并应用最后的卷积来计算注意图。
然后将此操作的结果用于通过逐元素乘法调制输入特征。
第二个类MSCAAttention以第一个为基础，增加了额外的抽象层。它在1x1卷积的开始和结束对输入特征进行投影，以便在第一个类中定义的MSCA注意力可以更有效地应用。
在这些投影间，它应用了第一个类中定义的MSCA注意力机制。它还包括一个围绕注意力模块的快捷连接，遵循残差学习的原则。
这意味着最终输出是通过注意力机制修改后的特征与原始输入特征的和，这有助于在仍然受益于注意力机制所做的更改的同时保留原始信息。
代码以一个基本测试结束，它创建了一个输入通道大小为128的MSCASpatialAttention实例，并将其应用于形状为(3, 128, 256, 256)的随机张量，
这可以代表一批3幅图像，每幅图像的高度和宽度为256像素，具有128个特征图/通道。然后它打印输入和输出的形状，预期保持相同，
这表明注意力机制可以以不改变输入张量维度的方式进行应用，使得它可以轻松地集成到更大的神经网络架构的各个点中，而无需调整周围层。
"""
class MSCAA(nn.Module):
    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):

        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        x = attn * u

        return x


class MSCAAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]]
                 ):

        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = torch.nn.GELU()
        self.spatial_gating_unit = MSCAA(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):

        shorcut = x.clone()

        x = self.proj_1(x)

        x = self.activation(x)

        x = self.spatial_gating_unit(x)

        x = self.proj_2(x)

        x = x + shorcut
        return x


if __name__ == '__main__':
    x = torch.rand(1, 64, 572, 572)
    model = MSCAAttention(in_channels=64)
    pred = model(x)
    print(x.shape)
    print(pred.shape)