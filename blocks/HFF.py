import torch
import torch.nn as nn
import torch.nn.functional as F
#HiFuse: Hierarchical multi-scale feature fusion network for medical image classification
#https://www.sciencedirect.com/science/article/abs/pii/S1746809423009679
"""
LayerNorm：一个执行层归一化的模块，支持channels_last和channels_first数据格式。它对指定维度上的输入数据进行归一化，改善训练稳定性和性能。

DropPath（随机深度）：实现了DropPath技术，该技术在训练期间随机丢弃网络中的整个路径。这种规范化方法有助于减少过拟合，提高模型鲁棒性。

Conv：一个可配置的卷积块，支持各种选项，如内核大小、步幅、批量归一化和ReLU激活，使其适用于网络内不同的卷积操作。

反转剩余MLP（IRMLP）：该模块引入了反转剩余结构，结合了MLP风格的操作。它涉及深度可分离卷积和GELU激活，旨在用最小的计算开销增强特征提取。

层次化特征融合块（HFF块）：HiFuse架构的核心，该块融合了网络不同级别和尺度的特征。
它利用自适应池化、挤压激励块、空间注意力和层归一化有效地组合局部和全局特征，以及中间特征图，以进行丰富的表示学习。

空间和通道注意力：在HFF块中，通过卷积和Sigmoid激活对局部特征应用空间注意力，使模型专注于相关的空间信息。
通过最大和平均池化，随后的挤压激励操作对全局特征应用通道注意力，强调有信息的通道。

融合和残差连接：HFF块通过连接和规范操作将全局、局部和中间特征结合起来，然后通过卷积和GELU激活进行处理。
一个残差连接，结合DropPath规范化，促进有效的信息流动并缓解梯度消失问题。
"""
class LayerNorm(nn.Module):
    """
    channels_last corresponds to inputs with shape (batch_size, height, width, channels)
    channels_first corresponds to inputs with shape (batch_size, channels, height, width)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out

# Hierachical Feature Fusion Block
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):

        W_local = self.W_l(l)   # local feature from Local Feature Block
        W_global = self.W_g(g)   # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # spatial attention for ConvNeXt branch
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump

        # channel attetion for transformer branch
        g_jump = g
        max_result=self.maxpool(g)
        avg_result=self.avgpool(g)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        g = self.sigmoid(max_out+avg_out) * g_jump

        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse


if __name__ == '__main__':

    # 实例化HFF_block
    hff_block1 = HFF_block(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192, drop_rate=0)
    hff_block2 = HFF_block(ch_1=128, ch_2=128, r_2=16, ch_int=128, ch_out=128, drop_rate=0)

    # 生成模拟输入
    l1 = torch.rand(1, 192, 28, 28)  # 局部特征
    g1 = torch.rand(1, 192, 28, 28)  # 全局特征
    f1 = torch.rand(1, 96 , 56, 56)  # 中间特征

    l2 = torch.rand(1, 128, 64, 64)  # 局部特征
    g2 = torch.rand(1, 128, 64, 64)  # 全局特征
    f2 = torch.rand(1, 64, 128, 128)  # 中间特征

    # 传递输入并获取输出
    output = hff_block1(l1, g1, f1)
    # output = hff_block2(l2, g2, None)
    # output = hff_block2(l2, g2, f2)


    # 打印输入输出的尺寸
    print(f"Input shapes: \nl1: {l1.shape} \ng1: {g1.shape} \nf1: {f1.shape}")
    print(f"Output shape: \n{output.shape}")