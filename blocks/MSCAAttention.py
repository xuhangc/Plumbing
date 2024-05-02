import torch
import torch.nn as nn
from mmengine.model import BaseModule


"""
这个代码实现了一个称为"MSCAAttention"（Multi-Scale Channel Attention）的注意力模块。这种注意力模块的主要作用是增强神经网络在特定通道和空间维度上的感知能力，从而有助于提取更加丰富和有用的特征。

这个注意力模块的特点如下：

多尺度特征提取：它使用了多个卷积核大小和填充的卷积操作，以提取不同尺度的特征信息。这些卷积操作包括一个具有较大卷积核的初始卷积 (self.conv0) 和多个后续的卷积操作（self.conv0_1，self.conv0_2，self.conv1_1，self.conv1_2，self.conv2_1，self.conv2_2），每个都针对不同的核大小和填充。

通道混合：在提取多尺度特征之后，通过对这些特征进行通道混合来整合不同尺度的信息。通道混合操作由最后一个卷积层 self.conv3 完成。

卷积注意力：最后，通过将通道混合后的特征与输入特征进行逐元素乘法，实现了一种卷积注意力机制。这意味着模块通过对不同通道的特征赋予不同的权重来选择性地强调或抑制输入特征。

总的来说，MSCAAttention的主要作用是增强特征图的表示能力，它能够自动学习特定通道和空间位置的重要性，从而更好地捕捉图像或特征图中的关键信息。这有助于改善模型在各种计算机视觉任务中的性能，例如图像分类、目标检测和语义分割。
"""

class MSCAAttention(BaseModule):

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
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x

if __name__ == '__main__':
    block = MSCAAttention(channels=64)
    input = torch.rand(64, 64, 9, 9)
    output = block(input)
    print(input.size())
    print(output.size())
