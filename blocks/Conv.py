import torch
import torch.nn as nn
# 深度可分离卷积（Depthwise Separable Convolutions）https://arxiv.org/pdf/1610.02357
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

from mmcv.ops import DeformConv2d
# 可变形卷积（Deformable Convolution）https://arxiv.org/pdf/1703.06211
class DeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DeformableConv, self).__init__()
        self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size * deformable_groups, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups)

    def forward(self, x):
        offsets = self.offsets(x)
        return self.deform_conv(x, offsets)


if __name__ == '__main__':
    input = torch.randn(1, 64, 64, 64)
    model = DeformableConv(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
    output = model(input)
    print(output.shape)
