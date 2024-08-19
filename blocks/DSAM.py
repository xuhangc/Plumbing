import torch
import torch.nn as nn
import torch.nn.functional as F
#Dual-domain strip attention for image restoration
#https://www.sciencedirect.com/science/article/abs/pii/S0893608023006974
"""
DSAMBlock：这是一个主要的网络块，它集成了双域条带注意力机制。这个模块首先通过SpecAtte处理输入特征图，
然后将特征图分为两半，分别应用两种不同大小的立方体注意力机制——cubic_11和cubic_7。这两部分的输出最后被合并。

cubic_attention：这是一个注意力模块，它对给定的特征图使用立方体注意力，通过spatial_strip_att在水平和垂直方向分别应用条带注意力。
然后，它通过可学习的参数gamma和beta来调整注意力的强度和原始特征图的影响程度。

spatial_strip_att：这个模块实现了空间条带注意力机制，它通过分组卷积来捕获特定的空间关系。条带是通过控制卷积核来定义的，这决定了注意力的方向是水平条带还是垂直条带。

GlobalPoolStripAttention：这个模块应用全局条带池化，目标是捕捉全局的水平和垂直信息。使用适应性池化操作，并借助gamma和beta参数融合池化后的信息。

LocalPoolStripAttention：这个模块与全局条带注意力类似，但它关注在局部条带级别。它使用普通的池化操作，然后添加反射填充来保持尺寸。

SpecAtte：这是一个专门处理光谱注意力的模块，它结合了全局和局部条带注意力，通过GlobalPoolStripAttention和两个LocalPoolStripAttention来融合全局和局部上下文的信息。
"""
class DSAMBlock(nn.Module):
    def __init__(self, in_channel):
        super(DSAMBlock, self).__init__()
        self.cubic_11 = cubic_attention(in_channel // 2, group=1, kernel=11)
        self.cubic_7 = cubic_attention(in_channel // 2, group=1, kernel=7)
        self.pool_att = SpecAtte(in_channel)

    def forward(self, x):
        out = self.pool_att(x)
        out = torch.chunk(out, 2, dim=1)
        out_11 = self.cubic_11(out[0])
        out_7 = self.cubic_7(out[1])
        out = torch.cat((out_11, out_7), dim=1)

        return out


class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)

        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class GlobalPoolStripAttention(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()

        self.channel = k

        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.vert_pool = nn.AdaptiveAvgPool2d((1, None))
        self.hori_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        hori_l = self.hori_pool(x)  # 1,3,10,1
        hori_h = x - hori_l

        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h
        vert_l = self.vert_pool(hori_out)  # 1,3,1,10
        vert_h = hori_out - vert_l

        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h

        return x * self.beta + vert_out * self.gamma


class LocalPoolStripAttention(nn.Module):
    def __init__(self, k, kernel=7) -> None:
        super().__init__()

        self.channel = k

        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)

        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))

        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        hori_l = self.hori_pool(self.pad_hori(x))
        hori_h = x - hori_l

        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h

        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l

        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h

        return x * self.beta + vert_out * self.gamma


class SpecAtte(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()

        self.global_att = GlobalPoolStripAttention(k)
        self.local_att_7 = LocalPoolStripAttention(k, kernel=7)
        self.local_att_11 = LocalPoolStripAttention(k, kernel=11)
        self.conv = nn.Conv2d(k, k, 1)

    def forward(self, x):
        global_out = self.global_att(x)
        local_7_out = self.local_att_7(x)
        local_11_out = self.local_att_11(x)

        out = global_out + local_7_out + local_11_out

        return self.conv(out)


if __name__ == '__main__':
    # 这里假设输入通道数in_channel为64
    in_channel = 64
    height = 64
    width = 64

    # 实例化DSAMBlock
    dsam_block = DSAMBlock(in_channel)

    # 创建输入张量，随机初始化数据
    input_tensor = torch.randn(1, in_channel, height, width)

    # 通过DSAMBlock传递输入张量
    output_tensor = dsam_block(input_tensor)

    # 打印输入输出的shape
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)