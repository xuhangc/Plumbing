import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
"""
1. **注意力（Attention）**：`Attention`模块定义了多头自注意力机制，包括层归一化（Layer Normalization）、线性投影层用于查询、键和值向量。注意力机制计算注意力分数，应用softmax，然后根据注意力分数计算值的加权和。

2. **前馈（FeedForward）**：`FeedForward`模块定义了一个简单的前馈神经网络，包括层归一化、GELU激活函数和dropout。

3. **并行（Parallel）**：`Parallel`模块并行应用一组函数，并对结果进行求和。在这种情况下，它用于并行化多个注意力和前馈块。

4. **Transformer**：`Transformer`模块堆叠了多层多头自注意力和前馈块。每一层由`Parallel`模块定义的块并行化。

5. **ViT**：`ViT`类定义了完整的视觉Transformer模型。它包括用于将图像块转换为嵌入（embeddings）、位置嵌入（position embeddings）、类别标记（class tokens）的函数，应用变换器层，并且有用于分类的多层感知器（MLP）头。

6. **前向传播**：`ViT`类中的`forward`函数负责通过模型的数据流。它将图像块转换为嵌入，添加位置嵌入和类别标记，应用变换器层，执行全局平均池化（或者取第一个标记的表示，取决于所选择的池化策略），通过MLP头，并返回最终的预测。

7. **主要模块**：`if __name__ == '__main__':`块用于实例化`ViT`模型并对示例输入图像张量执行前向传播。

这段代码可用于创建和训练用于图像分类任务的视觉Transformer模型。
"""

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_parallel_branches = 2, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        attn_block = lambda: Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        ff_block = lambda: FeedForward(dim, mlp_dim, dropout = dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Parallel(*[attn_block() for _ in range(num_parallel_branches)]),
                Parallel(*[ff_block() for _ in range(num_parallel_branches)]),
            ]))

    def forward(self, x):
        for attns, ffs in self.layers:
            x = attns(x) + x
            x = ffs(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', num_parallel_branches = 2, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_parallel_branches, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
if __name__ == '__main__':
    v = ViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        num_parallel_branches = 2,  # in paper, they claimed 2 was optimal
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(4, 3, 256, 256)

    preds = v(img) # (4, 1000)
    print(preds.shape)