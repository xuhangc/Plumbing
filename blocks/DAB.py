import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_
#DEFORMTIME: Capturing Variable Dependencies with Deformable Attention for Time Series Forecasting
#https://arxiv.org/pdf/2406.07438
"""
DeformAtten1D类实现了一个可变形注意力机制，专为时间序列预测而设计。以下是其组件的详细介绍：

初始化
参数：

seq_len：输入序列的长度。
d_model：模型的维度。
n_heads：注意力头的数量。
dropout：Dropout率。
kernel：卷积层的核大小。
n_groups：将输入通道划分成的组数。
no_off：如果为True，则在注意力机制中不使用偏移。
rpb：如果为True，则使用相对位置偏差。
层：

用于查询（proj_q）、键（proj_k）和值（proj_v）投影的卷积层。
用于计算偏移的卷积层（proj_offset）。
用于输出投影的线性层（proj_out）。
可选的相对位置偏差表（relative_position_bias_table）。
前向传播
输入重塑：

输入张量x从形状 [B, L, C] 转换为 [B, C, L]，以便进行卷积操作。
查询、键、值投影：

使用1D卷积计算 q、k 和 v。
查询被分组以便于并行处理。
偏移计算：

使用 proj_offset 网络计算偏移。
根据 offset_range_factor 调整偏移值。
网格采样：

使用 F.grid_sample 对输入序列应用偏移，基于学习到的偏移对输入进行采样。
缩放点积注意力：

使用 q 和 k 的缩放点积计算注意力分数。
应用 softmax 得到注意力权重。
根据注意力权重计算值（v）的加权和，得到最终输出。
输出投影：

输出被重塑并通过线性层投影回原始维度。
辅助函数
grid_sample_1d：通过重塑输入为2D，使 F.grid_sample 适用于1D序列。
normalize_grid：将序列索引归一化到范围[-1, 1]，以便于 F.grid_sample 使用。
"""
class DeformAtten1D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''

    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4, no_off=False, rpb=True) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv1d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride,
                      padding=pad_size),
            nn.Conv1d(self.n_group_channels, 1, kernel_size=1, stride=self.stride, padding=pad_size),
        )

        self.scale_factor = self.d_model ** -0.5  # 1/np.sqrt(dim)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.d_model, self.seq_len))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        dtype, device = x.dtype, x.device
        x = x.permute(0, 2, 1)  # B, C, L

        q = self.proj_q(x)  # B, C, L

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g=self.n_groups)

        grouped_queries = group(q)

        offset = self.proj_offset(grouped_queries)  # B * g 1 Lg
        offset = rearrange(offset, 'b 1 n -> b n')

        def grid_sample_1d(feats, grid, *args, **kwargs):
            # does 1d grid sample by reshaping it to 2d
            grid = rearrange(grid, '... -> ... 1 1')
            grid = F.pad(grid, (1, 0), value=0.)
            feats = rearrange(feats, '... -> ... 1')
            # the backward of F.grid_sample is non-deterministic
            # See for details: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            out = F.grid_sample(feats, grid, **kwargs)
            return rearrange(out, '... 1 -> ...')

        def normalize_grid(arange, dim=1, out_dim=-1):
            # normalizes 1d sequence to range of -1 to 1
            n = arange.shape[-1]
            return 2.0 * arange / max(n - 1, 1) - 1.0

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        if self.no_off:
            x_sampled = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        else:
            grid = torch.arange(offset.shape[-1], device=device)
            vgrid = grid + offset
            vgrid_scaled = normalize_grid(vgrid)

            x_sampled = grid_sample_1d(
                group(x),
                vgrid_scaled,
                mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, :L]

        if not self.no_off:
            x_sampled = rearrange(x_sampled, '(b g) d n -> b (g d) n', g=self.n_groups)
        q = q.reshape(B * self.n_heads, self.n_head_channels, L)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)
        if self.rpb:
            v = self.proj_v(x_sampled)
            v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, self.n_head_channels, L)
        else:
            v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)  # softmax: attention[0,0,:].sum() = 1

        out = torch.einsum('b i j , b j d -> b i d', attention, v)

        return self.proj_out(rearrange(out, '(b g) l c -> b c (g l)', b=B))


if __name__ == '__main__':
    # 示例用法：
    seq_len = 100
    d_model = 512
    n_heads = 4
    dropout = 0.1
    kernel = 5
    n_groups = 4
    no_off = False
    rpb = True

    model = DeformAtten1D(seq_len, d_model, n_heads, dropout, kernel, n_groups, no_off, rpb)
    x = torch.randn(32, seq_len, d_model)  # 批量大小32，序列长度100，特征维度512
    output = model(x)
    print(output.shape)  # 应与输入形状匹配 [32, seq_len, d_model]
