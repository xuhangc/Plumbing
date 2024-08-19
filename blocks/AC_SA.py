import torch
import torch.nn as nn
# https://arxiv.org/pdf/2308.03364
# Dual Aggregation Transformer for Image Super-Resolution, ICCV 2023
# https://github.com/zhengchen1999/DAT

class Adaptive_Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """ Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


if __name__ == '__main__':
    # 定义输入参数
    batch_size = 2
    height = 32  # 假设图像高度为32
    width = 32  # 假设图像宽度为32
    channels = 64  # 输入通道数 (dim)

    # 实例化Adaptive_Channel_Attention
    aca = Adaptive_Channel_Attention(dim=channels)

    # 创建随机输入数据 (B, H*W, C)
    x = torch.randn(batch_size, height * width, channels)

    # 前向传播并打印输入输出的形状
    output = aca(x, height, width)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)