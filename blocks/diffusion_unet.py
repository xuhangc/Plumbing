import warnings
warnings.filterwarnings("ignore")

from monai.networks.blocks import UnetBasicBlock, UnetResBlock, UnetUpBlock, Convolution, UnetOutBlock
import math
import torch
import torch.nn as nn

from monai.networks.layers.utils import get_act_layer


"""
对噪声进行建模的神经网络通常被选择为U-Net，因为噪声输入和噪声输出必须具有相同的大小。为了支持 3D 输入，我们用 3D 卷积替换了原始 U-Net 架构中的 2D 卷积。
U-Net 编码器部分中的每个块都由卷积层组成。这些层使用大小为 3 × 3 × 1 的内核对图像进行下采样，并且仅在 3D 体积的高分辨率平面上运行。随后，实现了Ho等人提出的空间和深度注意层3。
空间注意力层通过计算高分辨率图像平面上每个元素的键、查询和值向量来利用全局注意力机制，从而将深度维度视为批量大小的扩展。利用 Vaswani 等人提出的注意机制组合所得向量.
我们在空间注意力层之后引入了深度注意力层。该阶段将高分辨率图像平面的轴视为批处理轴，从而使高分辨率平面上的每个特征向量能够关注不同深度切片上的特征向量。
U-Net 的解码器的构建方式类似，其中每个块中应用了卷积层，然后是空间和深度注意块。此外，图像通过转置卷积35进行上采样，同时将跳跃连接添加到每个块的输出。
"""

class SinusoidalPosEmb(nn.Module):
    def __init__(self, emb_dim=16, downscale_freq_shift=1, max_period=10000, flip_sin_to_cos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, x):
        device = x.device
        half_dim = self.emb_dim // 2
        emb = math.log(self.max_period) / \
            (half_dim - self.downscale_freq_shift)
        emb = torch.exp(-emb*torch.arange(half_dim, device=device))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        half_dim = emb_dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        if self.emb_dim % 2 == 1:
            fouriered = torch.nn.functional.pad(fouriered, (0, 1, 0, 0))
        return fouriered


class TimeEmbbeding(nn.Module):
    def __init__(
        self,
        emb_dim=64,
        pos_embedder=SinusoidalPosEmb,
        pos_embedder_kwargs={},
        act_name=("SWISH", {})  # Swish = SiLU
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.pos_emb_dim = pos_embedder_kwargs.get('emb_dim', emb_dim//4)
        pos_embedder_kwargs['emb_dim'] = self.pos_emb_dim
        self.pos_embedder = pos_embedder(**pos_embedder_kwargs)

        self.time_emb = nn.Sequential(
            self.pos_embedder,
            nn.Linear(self.pos_emb_dim, self.emb_dim),
            get_act_layer(act_name),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

    def forward(self, time):
        return self.time_emb(time)



class DownBlock(nn.Module):
    def __init__(
            self,
            spatial_dims,
            in_ch,
            out_ch,
            time_emb_dim,
            cond_emb_dim,
            act_name=("swish", {}),
            **kwargs):
        super(DownBlock, self).__init__()
        self.loca_time_embedder = nn.Sequential(
            get_act_layer(name=act_name),
            nn.Linear(time_emb_dim, in_ch)  # in_ch * 2
        )
        if cond_emb_dim is not None:
            self.loca_cond_embedder = nn.Sequential(
                get_act_layer(name=act_name),
                nn.Linear(cond_emb_dim, in_ch),
            )
        self.down_op = UnetBasicBlock(
            spatial_dims, in_ch, out_ch, act_name=act_name, **kwargs)

    def forward(self, x, time_emb, cond_emb):
        b, c, *_ = x.shape
        sp_dim = x.ndim-2

        # ------------ Time ----------
        time_emb = self.loca_time_embedder(time_emb)
        time_emb = time_emb.reshape(b, c, *((1,)*sp_dim))
        # scale, shift = time_emb.chunk(2, dim = 1)

        # ------------ Combine ------------
        # x = x * (scale + 1) + shift
        x = x + time_emb

        # ----------- Condition ------------
        if cond_emb is not None:
            cond_emb = self.loca_cond_embedder(cond_emb)
            cond_emb = cond_emb.reshape(b, c, *((1,)*sp_dim))
            x = x + cond_emb

        # ----------- Image ---------
        y = self.down_op(x)
        return y


class UpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims,
            skip_ch,
            enc_ch,
            time_emb_dim,
            cond_emb_dim,
            act_name=("swish", {}),
            **kwargs):
        super(UpBlock, self).__init__()
        self.up_op = UnetUpBlock(spatial_dims, enc_ch,
                                 skip_ch, act_name=act_name, **kwargs)
        self.loca_time_embedder = nn.Sequential(
            get_act_layer(name=act_name),
            nn.Linear(time_emb_dim, skip_ch * 2),
        )
        if cond_emb_dim is not None:
            self.loca_cond_embedder = nn.Sequential(
                get_act_layer(name=act_name),
                nn.Linear(cond_emb_dim, skip_ch * 2),
            )

    def forward(self, x_skip, x_enc, time_emb, cond_emb):
        b, c, *_ = x_enc.shape
        sp_dim = x_enc.ndim-2

        # ----------- Time --------------
        time_emb = self.loca_time_embedder(time_emb)
        time_emb = time_emb.reshape(b, c, *((1,)*sp_dim))
        # scale, shift = time_emb.chunk(2, dim = 1)

        # -------- Combine -------------
        # y = x * (scale + 1) + shift
        x_enc = x_enc + time_emb

        # ----------- Condition ------------
        if cond_emb is not None:
            cond_emb = self.loca_cond_embedder(cond_emb)
            cond_emb = cond_emb.reshape(b, c, *((1,)*sp_dim))
            x_enc = x_enc + cond_emb

        # ----------- Image -------------
        y = self.up_op(x_enc, x_skip)

        # -------- Combine -------------
        # y = y * (scale + 1) + shift

        return y


class UNet(nn.Module):

    def __init__(self,
                 in_ch=1,
                 out_ch=3,
                 spatial_dims=3,
                 hid_chs=[32,       64,      128,      256,  512],
                 kernel_sizes=[(1, 3, 3), (1, 3, 3), (1, 3, 3),    3,   3],
                 strides=[1,     (1, 2, 2), (1, 2, 2),    2,   2],
                 upsample_kernel_sizes=None,
                 act_name=("SWISH", {}),
                 norm_name=("INSTANCE", {"affine": True}),
                 time_embedder=TimeEmbbeding,
                 time_embedder_kwargs={},
                 cond_embedder=None,
                 cond_embedder_kwargs={},
                 # True = all but last layer, 0/False=disable, 1=only first layer, ...
                 deep_ver_supervision=True,
                 estimate_variance=False,
                 use_self_conditioning=False,
                 **kwargs
                 ):
        super().__init__()
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = strides[1:]

        # ------------- Time-Embedder-----------
        self.time_embedder = time_embedder(**time_embedder_kwargs)

        # ------------- Condition-Embedder-----------
        if cond_embedder is not None:
            self.cond_embedder = cond_embedder(**cond_embedder_kwargs)
            cond_emb_dim = self.cond_embedder.emb_dim
        else:
            self.cond_embedder = None
            cond_emb_dim = None

        # ----------- In-Convolution ------------
        in_ch = in_ch*2 if use_self_conditioning else in_ch
        self.inc = UnetBasicBlock(spatial_dims, in_ch, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0],
                                  act_name=act_name, norm_name=norm_name, **kwargs)

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(spatial_dims, hid_chs[i-1], hid_chs[i],  time_emb_dim=self.time_embedder.emb_dim,
                      cond_emb_dim=cond_emb_dim,  kernel_size=kernel_sizes[
                          i], stride=strides[i], act_name=act_name,
                      norm_name=norm_name, **kwargs)
            for i in range(1, len(strides))
        ])

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(spatial_dims, hid_chs[i], hid_chs[i+1], time_emb_dim=self.time_embedder.emb_dim,
                    cond_emb_dim=cond_emb_dim, kernel_size=kernel_sizes[i +
                                                                        1], stride=strides[i+1], act_name=act_name,
                    norm_name=norm_name, upsample_kernel_size=upsample_kernel_sizes[i], **kwargs)
            for i in range(len(strides)-1)
        ])

        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = UnetOutBlock(
            spatial_dims, hid_chs[0], out_ch_hor, dropout=None)
        if isinstance(deep_ver_supervision, bool):
            deep_ver_supervision = len(
                strides)-2 if deep_ver_supervision else 0
        self.outc_ver = nn.ModuleList([
            UnetOutBlock(spatial_dims, hid_chs[i], out_ch, dropout=None)
            for i in range(1, deep_ver_supervision+1)
        ])

    def forward(self, x_t, t, cond=None, self_cond=None, **kwargs):
        condition = cond
        # x_t [B, C, (D), H, W]
        # t [B,]

        # -------- In-Convolution --------------
        x = [None for _ in range(len(self.encoders)+1)]
        x_t = torch.cat([x_t, self_cond],
                        dim=1) if self_cond is not None else x_t
        x[0] = self.inc(x_t)

        # -------- Time Embedding (Gloabl) -----------
        time_emb = self.time_embedder(t)  # [B, C]

        # -------- Condition Embedding (Gloabl) -----------
        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None
        else:
            cond_emb = self.cond_embedder(condition)  # [B, C]

        # --------- Encoder --------------
        for i in range(len(self.encoders)):
            x[i+1] = self.encoders[i](x[i], time_emb, cond_emb)

        # -------- Decoder -----------
        for i in range(len(self.decoders), 0, -1):
            x[i-1] = self.decoders[i-1](x[i-1], x[i], time_emb, cond_emb)

        # ---------Out-Convolution ------------
        y_hor = self.outc(x[0])
        y_ver = [outc_ver_i(x[i+1])
                 for i, outc_ver_i in enumerate(self.outc_ver)]

        return y_hor   , y_ver

    def forward_with_cond_scale(self, *args, cond_scale=0., **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    model = UNet(in_ch=3)
    input = torch.randn((1, 3, 16, 128, 128))  # 16 - 输入数据的深度 (Depth)，表示有16个连续的切片或帧
    time = torch.randn((1,))
    out_hor, out_ver = model(input, time)
    print(out_hor.shape)
