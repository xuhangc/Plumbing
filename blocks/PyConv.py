# arxiv: https://arxiv.org/abs/2006.11538
# title：Pyramidal Convolution:Rethinking Convolutional Neural Networks for Visual Recognition
# source: https://github.com/iduta/pyconv/blob/master/models/pyconvresnet.py
import torch
import torch.nn as nn

"""
初始化：构造函数（__init__方法）初始化PyConv2d层。它接受几个参数：
in_channels：输入通道的数量。
out_channels：一个列表，指定卷积产生的每个金字塔级别的输出通道数。
pyconv_kernels：一个列表，指定每个金字塔级别的核的空间大小。
pyconv_groups：一个列表，指定每个金字塔级别从输入通道到输出通道的阻塞连接数。
stride，dilation，bias：用于卷积操作的可选参数（提供了默认值）。

初始化：在构造函数中，创建了一系列卷积层，每个卷积层对应于输入参数指定的一个金字塔级别。这些卷积层存储在nn.ModuleList中。

前向方法：forward方法定义了PyConv2d层的前向传递。它将每个卷积层应用于输入张量x，并收集输出。最后，它沿着通道维度（dim=1）连接这些输出，并返回结果。
"""
class PyConv2d(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
    """
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


if __name__ == '__main__':
    # PyConv with two pyramid levels, kernels: 3x3, 5x5
    m1 = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
    input1 = torch.randn(4, 64, 56, 56)
    output1 = m1(input1)
    print(output1.shape)
    # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
    m2 = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
    input2 = torch.randn(4, 64, 56, 56)
    output2 = m2(input2)
    print(output2.shape)
