import torch
from torch import nn, einsum
from torch.nn import Module
from einops import rearrange, pack, unpack
import torch.nn.functional as F
#XCiT: Cross-Covariance Image Transformers
#XCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

class XCAttention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads
        x, ps = pack_one(x, 'b * d')

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = einsum('b h i n, b h j n -> b h i j', q, k) * self.temperature.exp()

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j n -> b h i n', attn, v)
        out = rearrange(out, 'b h d n -> b n (h d)')

        out = unpack_one(out, ps, 'b * d')
        return self.to_out(out)