import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import einops
#SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications[ICCV'23]
#https://openaccess.thecvf.com/content/ICCV2023/papers/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.pdf
"""
该论文提出了一种名为SwiftFormer的高效、准确的图像分类模型。SwiftFormer采用了多级设计，通过在不同尺度下学习局部和全局特征来提高性能。
具体来说，SwiftFormer包括一个有效的卷积编码器和一个SwiftTransformer编码器。
卷积编码器用于提取空间信息并学习局部表示，而SwiftTransformer编码器则利用自注意力机制来捕捉全局上下文，并进一步丰富了局部和全局表示。

SwiftTransformer编码器中的自注意力机制被改进为一种高效的加法注意力机制，去除了不必要的键值交互，仅关注查询和键之间的交互。
这种改进使得SwiftTransformer具有更快的推理速度和更稳健的上下文表示能力。

此外，SwiftFormer还使用了一种称为“有效卷积编码器”的新型卷积编码器，它将深度可分离卷积和pointwise卷积相结合，以提高性能而不增加参数或延迟。
"""
class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out


class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class SwiftFormerEncoder(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.local_representation = SwiftFormerLocalRepresentation(dim=dim, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        x = self.local_representation(x)
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(
                    0, 3, 1, 2))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.linear(x))
        return x


if __name__ == '__main__':
    encoder = SwiftFormerEncoder(dim=128)
    input_tensor = torch.randn(1, 128, 32, 32)  # Example input tensor with shape [B, C, H, W]
    output_tensor = encoder(input_tensor)
    print(output_tensor.shape)  # Output tensor shape should be [B, C, H, W]
