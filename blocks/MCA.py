import torch
from torch import nn
import math


"""
这个模块实现了一个称为"MCALayer"（Multi-modal Channel Attention Layer）的注意力机制，它主要用于增强神经网络在不同通道之间的交互和信息整合。MCALayer具有以下几个关键组件和特点：

MCAGate模块：MCALayer包含了MCAGate模块，这是一个多模态的注意力机制。它利用池化操作（平均池化、最大池化、标准差池化）来提取不同的通道间的特征信息。这些不同类型的池化操作有助于捕捉通道间的不同统计特性。

通道间的交互：MCALayer具有三种不同类型的通道间交互方式，分别是水平-通道（h-cw）、垂直-通道（w-hc）和通道-通道（c-hw）交互。这些交互方式分别针对不同的维度，有助于模型更好地理解和整合不同通道之间的信息。

空间维度的处理：根据no_spatial参数的设置，MCALayer可以选择是否进行空间维度上的交互。如果no_spatial为True，只会进行通道间的交互；如果为False，还会进行空间维度上的交互。

权重融合：在不同的通道交互之后，MCALayer使用权重融合来整合不同池化方式的信息。通过学习的方式，模型可以决定如何分配不同池化方式的重要性。

多尺度核大小：MCALayer中的核大小会根据输入通道数自动选择，以增强模块的适应性。

总的来说，MCALayer通过多模态的注意力机制，引入不同类型的通道交互和池化操作，从而可以更好地捕捉特征之间的关系，提高模型的特征表示能力，有助于在计算机视觉任务中提高性能，如图像分类、目标检测和语义分割。此外，MCALayer的模块化设计使得它可以方便地嵌入到神经网络中，以增强模型的特征提取和表示能力。
"""


__all__ = ['MCALayer', 'MCAGate']


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out


if __name__ == '__main__':
    block = MCALayer(inp=64)
    input = torch.rand(64, 64, 9, 9)
    output = block(input)
    print(input.size())
    print(output.size())
