import torch
import torch.nn as nn
import math

"""
有效利用多模态输入进行准确的 RGB-D 显着性检测是人们高度关注的主题。
大多数现有作品利用跨模式交互来融合 RGB-D 的两个流以增强中间特征。
在此过程中，尚未充分考虑可用深度质量低的实际问题。
我们的目标是实现对低质量深度具有鲁棒性的 RGB-D 显着性检测，低质量深度主要以两种形式出现：噪声引起的不准确和 RGB 未对准。
一方面，逐层注意力（LWA）根据深度精度学习 RGB 和深度特征的早期和晚期融合之间的权衡。
一方面，三叉戟空间注意力（TSA）聚合了更广泛的空间上下文中的特征，以解决深度错位问题。
所提出的 LWA 和 TSA 机制使我们能够有效地利用多模态输入进行显着性检测，同时对低质量深度具有鲁棒性。
"""

"""
显著性检测在计算机视觉和图像处理领域起着重要作用，其具体用途包括但不限于以下几个方面：
图像检索和标记：显著性检测能够帮助计算机系统自动识别图像中最显著的对象或区域，从而帮助改进图像的标记和检索效果。
视觉注意力模型：人类视觉系统会优先关注图像中的显著对象或区域，显著性检测可以帮助设计和实现计算机视觉系统中的视觉注意力模型，从而改善计算机对图像的处理和理解能力。
图像分割：显著性检测可以作为图像分割的预处理步骤，帮助识别并提取出图像中的显著对象，为后续的分割和分析提供重要线索和信息。
图像压缩和传输：通过显著性检测可以识别图像中重要的信息，有助于进行有损压缩或者优化图像传输，以保留图像中最重要的部分。
总的来说，显著性检测在图像处理和计算机视觉中具有广泛的应用前景，可以帮助改善图像分析、理解和处理的效率和准确性。
"""

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(_ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class lwa(nn.Module):
    def __init__(self, channel, outsize):
        super().__init__()

        # self.rgbd = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
        self.dept = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
        self.rgb = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(_ConvBNReLU(channel, 24, 1, 1), _ConvBNSig(24, outsize, 1, 1))

    def forward(self, rgb, dep):
        assert rgb.size() == dep.size()

        rgbd = rgb + dep
        m_batchsize, C, width, height = rgb.size()

        proj_rgb = self.rgb(rgb).view(m_batchsize, -1, height * width).permute(0, 2, 1)  # B X (H*W) X C
        proj_dep = self.dept(dep).view(m_batchsize, -1, height * width)  # B X C x (H*W)
        energy = torch.bmm(proj_rgb, proj_dep) / math.sqrt(C)  # B X (H*W) X (H*W)
        attention1 = self.softmax1(energy)  # B X (H*W) X (H*W)

        att_r = torch.bmm(proj_rgb.permute(0, 2, 1), attention1)
        att_b = torch.bmm(proj_dep, attention1)
        # proj_rgbd = self.rgbd(rgbd).view(m_batchsize,-1,height*width) # B X C X (H*W)
        # attention2 = torch.bmm(proj_rgbd,attention1.permute(0,2,1) )
        attention2 = att_r + att_b
        output = attention2.view(m_batchsize, C, width, height) + rgbd

        GapOut = self.GAP(output)
        gate = self.mlp(GapOut)

        return gate


if __name__ == "__main__":
    K = lwa(20, 5)
    K.to('cuda')

    # 这两个输入可以换成自己的模型，没必要非得是 rgb 和 depth
    rgb = torch.rand([2, 64, 1, 1]).cuda()
    depth = torch.rand((2, 64, 1, 1)).cuda()
    model = lwa(64, 64).cuda()
    k = model(rgb, depth)
    print(k.size())
