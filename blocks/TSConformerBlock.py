import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
#CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement
#https://arxiv.org/pdf/2209.11112v3
"""
FeedForwardModule（前馈模块）：实现了一个标准的前馈网络，包含层归一化、线性变换、SiLU激活和dropout。

ConformerConvModule（卷积模块）：结合了带门机制（GLU）的深度可分离卷积和层归一化，适用于捕捉序列中的局部依赖关系。

AttentionModule（注意力模块）：利用多头自注意力机制和层归一化来建模输入序列中的全局依赖关系。

ConformerBlock（Conformer块）：以残差方式结合了上述模块（前馈、注意力、卷积），并包含后置层归一化。

TSConformerBlock（两阶段Conformer块）：依次应用时域和频域的Conformer块，捕捉时域和频域特征。
"""
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x,
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

#Two-stage conformer (TS-Conformer)
class TSConformerBlock(nn.Module):
    def __init__(self, num_channel=64):
        super(TSConformerBlock, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':
    # 实例化TSConformerBlock
    num_channels = 64  # 设置通道数
    model = TSConformerBlock(num_channel=num_channels)

    # 创建一个输入tensor，假设输入的形状为 (batch_size, channels, time_steps, freq_bins)
    batch_size = 1
    channels = num_channels
    time_steps = 100
    freq_bins = 80
    x = torch.randn(batch_size, channels, time_steps, freq_bins)

    # 前向传播
    output = model(x)

    # 打印输入和输出的shape
    print("输入的shape:", x.shape)
    print("输出的shape:", output.shape)