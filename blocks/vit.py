import torch
import torch.nn as nn
from XCAttention import XCAttention
from FFNs import FeedForward
from LPI import LocalPatchInteraction_bnc
#ViT的简单实现
"""
首先，我们将输入图像划分为多个小块（补丁），并对每个补丁进行嵌入处理。我们使用一个卷积层（nn.Conv2d）来实现这个功能。
self.proj 是一个卷积层，将每个 16x16x3 的补丁映射到一个 768 维的向量。
x.flatten(2) 将卷积后的结果展平成一个 2D 张量。
x.transpose(1, 2) 调整张量的维度，使得每个补丁变成一个行向量。
"""
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # 经过卷积层，x的形状变为 (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # 展平操作，将形状变为 (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # 转置操作，将形状变为 (B, num_patches, embed_dim)
        return x

"""
Transformer编码器由多个层组成，每一层包含一个多头自注意力机制和一个前馈神经网络。

nn.LayerNorm(embed_dim) 对每个输入进行标准化。

nn.MultiheadAttention(embed_dim, num_heads, bias=qkv_bias) 是多头注意力机制，关注输入的不同部分。

nn.Sequential 中的两层全连接层和激活函数组成前馈神经网络(FFN)，包含了以下部分：
第一层全连接层 (nn.Linear(embed_dim, int(embed_dim * mlp_ratio)))：
这层将输入的向量从 embed_dim 维度（例如768维）映射到一个更高的维度 embed_dim * mlp_ratio（例如 768 * 4 = 3072 维）。
激活函数 (nn.GELU())：
使用GELU激活函数增加非线性。
第二层全连接层 (nn.Linear(int(embed_dim * mlp_ratio), embed_dim))：
这层将向量的维度从 embed_dim * mlp_ratio（例如3072维）映射回原来的 embed_dim 维度（例如768维）。

nn.Dropout(0.1) 防止过拟合。
"""
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4.0, depth=12, qkv_bias=True):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(embed_dim),#0
                    # nn.MultiheadAttention(embed_dim, num_heads, bias=qkv_bias),#这里就是要替换的注意力，注意这里的参数，和XCA对应上,可以看到，我这边已经替换成功了
                    XCAttention(embed_dim, num_heads),#1
                    # nn.Sequential(
                    #     nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    #     nn.GELU(),
                    #     nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
                    # ),#还是找到这个地方，这里我直接把这两个参数弄成一样的啦，对比一下，差距还是很大的，这里也可以变成3072的
                    FeedForward(embed_dim, embed_dim * 4),#改一下hiddendim,应该是4倍，768的，我这里先把我做的两个改动都弄上了#2
                    nn.LayerNorm(embed_dim),#3
                    nn.Dropout(0.1),#4
                    LocalPatchInteraction_bnc(embed_dim)#5
                ])
            )

    def forward(self, x):#这里layer[0]和上面的nn.LayerNorm对应，layer[1]和nn.MultiheadAttention对应，以此类推，放在注意力后面，这里的残差连接看得我好晕啊
#我就不完全按照图上的来改了，头晕了,直接放在注意力后面了
        for layer in self.layers:
            x_norm = layer[0](x)  # LayerNorm
            # attn_output, _ = layer[1](x_norm, x_norm, x_norm)  # 多头注意力机制
            attn_output = layer[1](x_norm)  # 多头注意力机制,我这里遇到维度问题了，打印出来看看，我这里应该传入的是nchw，而不是torch.Size([1, 197, 768])
            # print(attn_output.shape)这样的话，我们就成功把这个LPI模块加在注意力后面了，虽然不是图里那个样子，但可以尽可能的还原，比如在前面加一个layernorm
            attn_output = layer[5](attn_output)
            attn_output = layer[4](attn_output + x)  # 残差连接 + Dropout
            mlp_output = layer[2](layer[3](attn_output))  # 前馈神经网络 (MLP)
            x = mlp_output + attn_output  # 残差连接
        return x

"""
结合补丁嵌入和Transformer编码器，并添加一个分类头。
self.patch_embed 将输入图像转换为补丁嵌入。
self.cls_token 是分类 token，用于分类任务。
self.pos_embed 是位置嵌入，帮助保留补丁的位置信息。
self.encoder 是Transformer编码器，处理嵌入向量。
self.norm 对输出进行标准化。
self.head 是分类头，将分类 token 的向量映射到目标类别数（1000）。
"""
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(embed_dim, num_heads, mlp_ratio, depth, qkv_bias)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # 分类头

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 提取分类 token 的向量
        x = self.head(cls_token_final)  # 映射到类别数
        return x

"""
创建ViT实例并测试其输入和输出形状

"""
if __name__ == '__main__':

    # 实例化Vision Transformer
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 768
    depth = 12
    num_heads = 12
    num_classes = 1000  # 分类类别数

    vit = VisionTransformer(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                            num_classes=num_classes)

    print(vit)

    # 模拟一个输入
    x = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image
    output = vit(x)

    # 打印输入输出的shape
    print(f"Input shape: {x.shape}")  # Expected: torch.Size([1, 3, 224, 224])
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([1, 1000])
