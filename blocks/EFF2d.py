import torch
import torch.nn as nn
import math
#SUnet: A multi-organ segmentation network based on multiple attention
#https://www.sciencedirect.com/science/article/abs/pii/S0010482523010612
"""
SpatialAttention 模块： 这个模块计算空间注意力图。它接受一个输入特征图，并通过对特征通道进行平均池化和最大池化来计算一个注意力图，
将结果合并，然后通过一个卷积层处理，后接一个 Sigmoid 激活函数。这个注意力图随后被用于在空间上重调原始特征图。

Efficient_Attention_Gate (EAG) 模块： EAG模块旨在选择性地集中关注门控机制内的相关特性。
它对门控 (g) 和输入特征图 (x) 均使用分组卷积，后续接ReLU激活函数。
这些卷积的输出被添加在一起，通过‘psi’网络传递（包含一个卷积层和Sigmoid激活函数），并被用来调节输入特征图通过元素逐一相乘，将门控信息与原始特性相结合。

EfficientChannelAttention (ECA) 模块： 该模块通过运用一种基于通道尺寸的自适应卷积，专注于重新校准通道特征响应。
它首先应用全局平均池化来挤压空间维度，然后使用动态确定核大小的1D卷积来捕捉通道依赖性。这个卷积的输出经过Sigmoid激活产生通道权重，随后被用来缩放原始特征图。

EFF 模块： 这是一个综合模块，结合了上述的注意力机制。它根据是否位于网络的底层来有条件地应用 Efficient Attention Gate (EAG)，
如果不是底层，则将输出通过合并与跳跃连接结合，应用 Efficient Channel Attention (ECA) 进行通道特征的重新标定，
随后通过 SpatialAttention 模块进行空间注意力的重新标定。这种全面的注意力机制帮助网络集中关注空间和通道上的重要特征，可能提高分割性能。
"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
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

class Efficient_Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=32):
        super(Efficient_Attention_Gate, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(x1 + g1))
        out = x * psi
        out += x

        return out

class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return v

class EFF(nn.Module):
    def __init__(self, in_dim, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.EAG = Efficient_Attention_Gate(in_dim, in_dim, in_dim)
        else:
            self.EAG = nn.Identity()
        self.ECA = EfficientChannelAttention(in_dim*2)
        self.SA = SpatialAttention()

    def forward(self, x, skip):
        if not self.is_bottom:
            EAG_skip = self.EAG(x, skip)
            x = torch.cat((EAG_skip, x), dim=1)
            # x = EAG_skip + x
        else:
            x = self.EAG(x)
        x = self.ECA(x) * x
        x = self.SA(x) * x
        return x


if __name__ == '__main__':
    # 实例化 EFF 类
    eff_module = EFF(in_dim=512, is_bottom=False)

    batch_size = 1
    height = 71
    width = 71
    in_dim = 512

    x1 = torch.randn(1, 512, 71, 71)
    x2 = torch.randn(batch_size, 512, 71, 71)

    # 将张量通过 EFF 模块
    output = eff_module(x1, x2)

    print(f"Input shape1: {x1.shape}")
    print(f"Input shape2: {x2.shape}")
    print(f"Output shape: {output.shape}")