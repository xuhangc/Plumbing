from functools import partial
import torch
import torch.nn as nn
#SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization
#https://arxiv.org/pdf/2405.11582
from einops import rearrange


class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))  # 可学习参数 alpha，初始值为 1
        self.bn = nn.BatchNorm1d(channels)  # 批量归一化层

    def forward(self, x):
        x = x.transpose(1, 2)  # 转置输入张量
        x = self.bn(x) + self.alpha * x  # 应用批量归一化，并加上 alpha 控制的残差连接
        x = x.transpose(1, 2)  # 转置回原来的形状
        return x

class SlabLinearAttentionReplaceSwinTransformer(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        # print('Linear Attention window{} f{} kernel{}'.
        #       format(window_size, focusing_factor, kernel_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding

        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LinearNorm(nn.Module):
    def __init__(self, dim, norm1, norm2, step=300000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer('iter', torch.tensor(step))  # 当前迭代计数器
        self.register_buffer('total_step', torch.tensor(step))  # 线性衰减的总步数
        self.r0 = r0  # 初始比例
        self.norm1 = norm1(dim)  # 第一个归一化层
        self.norm2 = norm2(dim)  # 第二个归一化层

    def forward(self, x):
        if self.training:
            lamda = self.r0 * self.iter / self.total_step  # lamda 的线性衰减
            if self.iter > 0:
                self.iter.copy_(self.iter - 1)  # 递减迭代计数器
            x1 = self.norm1(x)  # 应用第一个归一化
            x2 = self.norm2(x)  # 应用第二个归一化
            x = lamda * x1 + (1 - lamda) * x2  # 混合两个归一化
        else:
            x = self.norm2(x)  # 在评估时只使用第二个归一化
        return x


if __name__ == '__main__':
    linearnorm = partial(LinearNorm, norm1=nn.LayerNorm, norm2=RepBN)