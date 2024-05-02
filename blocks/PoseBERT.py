import torch
from torch import nn
import roma
from einops import rearrange

"""
训练用于视频中人体姿势估计的最先进模型需要带有注释的数据集，而获得这些数据集非常困难且昂贵。尽管变压器最近已被用于身体姿势序列建模，但相关方法依赖于伪地面事实来增强当前可用于学习此类模型的有限训练数据。
在本文中，我们介绍了 PoseBERT，这是一个通过掩模建模对 3D 运动捕捉 (MoCap) 数据进行全面训练的转换器模块。它简单、通用且用途广泛，因为它可以插入任何基于图像的模型之上，以利用时间信息将其转换为基于视频的模型。
我们展示了具有不同输入的 PoseBERT 变体，从 3D 骨架关键点到全身 (SMPL) 或仅手 (MANO) 的 3D 参数模型的旋转。由于 PoseBERT 训练与任务无关，因此该模型可以应用于多种任务，例如姿势细化、未来姿势预测或运动完成，而无需微调。
我们的实验结果证实，在各种最先进的姿态估计方法之上添加 PoseBERT 可以持续提高其性能
"""

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForwardResidual(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., out_dim=24 * 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, x, init, n_iter=1):
        pred_pose = init
        for _ in range(n_iter):
            xf = torch.cat([x, init], -1)
            pred_pose = pred_pose + self.net(xf)
        return pred_pose


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        Args:
            - x: [batch_size,seq_len,dim]
            - mask: [batch_size,seq_len] - dytpe= torch.bool - default True everywhere, if False it means that we don't pay attention to this timestep
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B,H,T,T]
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:  # always true
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, n, 1)  # updating masked timesteps with context
            dots.masked_fill_(~mask, mask_value)  # ~ do the opposite i.e. move True to False here
            del mask
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x


class TransformerRegressor(nn.Module):

    def __init__(self, dim, depth=2, heads=8, dim_head=32, mlp_dim=32, dropout=0.1, out=[22 * 6, 3],
                 share_regressor=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(depth):
            list_modules = [
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]

            # Regressor
            if i == 0 or not share_regressor:
                # N regressor per layer
                for out_i in out:
                    list_modules.append(PreNorm(dim, FeedForwardResidual(dim, mlp_dim, dropout=dropout, out_dim=out_i)))
            else:
                # Share regressor across layers
                for j in range(2, len(self.layers[0])):
                    list_modules.append(self.layers[0][j])
            self.layers.append(nn.ModuleList(list_modules))

    def forward(self, x, init, mask=None):
        batch_size, seq_len, *_ = x.size()
        y = init
        for layers_i in self.layers:
            # attention and feeforward module
            attn, ff = layers_i[0], layers_i[1]
            x = attn(x, mask=mask) + x
            x = ff(x) + x

            # N regressors
            for j, reg in enumerate(layers_i[2:]):
                y[j] = reg(x, init=y[j], n_iter=1)

        return y


class PoseBERT(nn.Module):
    def __init__(self,
                 in_dim=24 * 6, n_jts_out=24, init_pose=None,
                 dim=512, depth=4, heads=8, dim_head=64, mlp_dim=512, dropout=0.1,
                 share_regressor=1,
                 *args, **kwargs):
        super(PoseBERT, self).__init__()

        self.pos = PositionalEncoding(dim, 1024)
        self.emb = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))

        self.decoder = TransformerRegressor(dim, depth, heads, dim_head, mlp_dim, dropout,
                                            [n_jts_out * 6],
                                            share_regressor == 1)

        if init_pose is None:
            init_pose = torch.zeros(n_jts_out * 6).float()
        self.register_buffer('init_pose', init_pose.reshape(1, 1, -1))

        # Type of input
        if in_dim == 24 * 6:
            self.input = 'rotmat'
        elif in_dim == 16 * 3 + 6:
            self.input = 'h36m'
        else:
            raise NameError

    def forward(self, rotmat, root=None, rel=None, mask=None):
        """
        Args:
            - rotmat: torch.Tensor - torch.float32 - [batch_size, seq_len, 24, 3, 3]
            - root: torch.Tensor - torch.float32 - [batch_size, seq_len, 3, 3]
            - rel: torch.Tensor - torch.float32 - [batch_size, seq_len, 17, 3]
            - mask: torch.Tensor - torch.bool - [batch_size, seq_len]
        Return:
            - y: torch.Tensor - [batch_size, seq_len, 24, 3, 3] - torch.float32
        """

        # Handling input
        if self.input == 'rotmat':
            assert rotmat is not None
            # Keep 6D representation only and concat
            x = rotmat[..., :2].flatten(2)  # [batch_size, seq-len, in_dim]
        elif self.input == 'h36m':
            assert root is not None and rel is not None
            # 6D repr of the root rotation and keep the relative pose only (discard the hip because it is centered)
            x = torch.cat([root[..., :2].flatten(2), rel[:, :, 1:].flatten(2)], -1)
        else:
            raise NameError

        batch_size, seq_len, *_ = x.size()

        # Default masks
        if mask is None:
            mask = torch.ones(batch_size, seq_len).type_as(x).bool()

        # Input embedding
        x = self.emb(x)
        x = x * mask.float().unsqueeze(-1) + self.mask_token * (1. - mask.float().unsqueeze(-1))  # masked token
        x = self.pos(x)  # inject position info

        # Transformer
        init = [self.init_pose.repeat(batch_size, seq_len, 1)]  # init mean pose
        y = self.decoder(x, init, mask)[0]

        # Move from rotation representation from 6D to 9D
        y = roma.special_gramschmidt(y.reshape(batch_size, seq_len, -1, 3, 2))

        return y

if __name__ == '__main__':
    # 初始化模型
    model = PoseBERT()

    # 生成随机旋转矩阵输入，形状为 [batch_size, seq_len, n_jts, rotmat_size, rotmat_size]
    input_rotmat = torch.randn(2, 10, 24, 3, 3)

    # 调用模型
    output = model(rotmat=input_rotmat)

    # 打印输入和输出尺寸
    print("Input size:", input_rotmat.size())
    print("Output size:", output.size())