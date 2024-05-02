#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" TAdaConv.

空间卷积广泛用于许多深度视频模型中。它从根本上假设时空不变性，即对不同帧中的每个位置使用共享权重。
用于视频理解的时间自适应卷积（TAdaConv），表明沿时间维度的自适应权重校准是促进视频中复杂时间动态建模的有效方法。
具体来说，TAdaConv通过根据每个帧的局部和全局时间上下文校准其卷积权重，从而赋予空间卷积以时间建模能力。与现有的时间建模操作相比，
TAdaConv的效率更高，因为它在卷积核上操作，而不是在要素上操作，其维度比空间分辨率小一个数量级。此外，内核校准带来了更大的模型容量。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d 
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)

        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in//self.groups, kh, kw)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0,2,1,3,4).unsqueeze(3) * self.weight).reshape(-1, c_in//self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D, 
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)

        return output
        
    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " +\
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"


if __name__ == '__main__':
    # 创建TAdaConv2d实例
    tada_conv2d = TAdaConv2d(in_channels=64, out_channels=64, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1])

    # 生成随机输入张量，形状为(batch_size, channels, depth, height, width)
    input_tensor = torch.rand(2, 64, 10, 32, 32)  # 例如，批量大小为2，时间深度为10，高和宽为32

    # 生成alpha张量，形状根据cal_dim参数来定，这里我们假设其形状和输入形状相关
    # 如果cal_dim="cin"，则alpha形状应该与输入通道数相对应
    # 如果cal_dim="cout"，则alpha形状应该与输出通道数相对应
    alpha_tensor = torch.rand(2, 64, 10, 1, 1)  # 随机生成校准权重

    # 使用TAdaConv2d实例处理输入
    output_tensor = tada_conv2d(input_tensor, alpha_tensor)

    # 打印输入和输出张量的形状
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output tensor shape: {output_tensor.shape}")
