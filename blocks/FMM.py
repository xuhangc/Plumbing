import torch
import torch.nn as nn
import torch.nn.functional as F
#Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution (ICCV 2023)
#https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.pdf
"""
LayerNorm（层归一化）：自定义的层归一化实现，支持“channels_first”和“channels_last”两种数据格式。

CCM（Convolutional Channel Mixer，卷积通道混合器）：该模块用于混合输入特征的通道。它由一系列卷积层组成，首先扩展通道数，然后再减少通道数，中间使用了GELU激活函数。

SAFM（Spatially-Adaptive Feature Modulation，空间自适应特征调制）：该模块旨在以不同的空间层次自适应地调制特征。
它将输入特征沿通道维度划分为若干块，并对每一块进行空间加权和特征聚合。处理后的特征会被组合起来，并经过激活处理。

FMM（Feature Mixing Module，特征混合模块）：该模块通过依次应用层归一化、SAFM和CCM来结合空间和通道特征。
它首先对输入进行归一化，然后应用SAFM，最后使用CCM混合特征，同时保留原始输入特征的残差连接。
"""
# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
# convolutional channel mixer (CCM)
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


# spatially-adaptive feature modulation (SAFM)
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

# feature mixing module(FMM)
class FMM(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x

if __name__ == '__main__':
    # 实例化 FMM 模块
    dim = 64
    fmm = FMM(dim)

    # 创建一个示例输入张量
    input_tensor = torch.randn(1, dim, 32, 32)

    # 传递输入张量并获取输出
    output_tensor = fmm(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)