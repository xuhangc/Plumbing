## Multi-DConv Head Transposed Self-Attention (MDTA)
import torch
from einops import rearrange
from torch import nn
"""
这段代码实现了一个多头自注意力机制，其中还包括深度卷积（dwconv）等额外操作。以下是各个组件和操作的详细解释：

初始化：__init__ 方法用于初始化注意力模块，其中包括输入维度（dim）、注意头的数量（num_heads）以及是否包含偏置项（bias）等参数。

卷积层：

self.qkv：这是一个卷积层，对输入张量 x 进行操作以产生查询（query）、键（key）和值（value）串联在一起的结果。它是一个 1x1 的卷积，将输入特征转换为查询、键和值向量。输出维度为 dim * 3，以适应三个组件：查询、键和值。
self.qkv_dwconv：这是一个深度卷积层，应用于串联的查询、键和值。它独立地操作每个输入通道，增强了空间建模能力。
self.project_out：这是另一个 1x1 的卷积层，将注意力机制的输出重新转换为原始维度 dim。
前向传播：

forward 方法接受输入张量 x 并执行注意力机制。
首先应用卷积层以获取查询、键和值张量。
这些张量被重塑为多头格式，以便进行并行计算。
查询和键张量沿着最后一个维度进行了归一化处理。
通过取查询和键的点积，并使用可学习参数 self.temperature 进行缩放，计算注意力分数。在最后一个维度上应用 softmax 函数以获取注意力权重。
使用这些注意力权重计算值张量的加权和。
最终结果张量被重塑回其原始形状，并通过 self.project_out 卷积层以获得最终输出。
输出：

输出张量与输入张量 x 具有相同的形状，但具有通过自注意力机制得到的增强表示。
这种架构似乎专为捕捉跨通道和通道内空间依赖关系而设计，适用于图像处理或序列建模等任务。深度卷积和多头自注意力的使用使得模型能够有效地捕捉通道间和空间间的信息。
"""

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape


        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # [B, head, C/head, HW] * [B, head, HW, C/head] * [head, 1, 1] ==> [B, head, C/head, C/head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # [B, head, C/head, C/head] * [B, head, C/head, HW] ==> [B, head, C/head, HW]
        out = (attn @ v)

        # [B, head, C/head, HW] ==> [B, head, C/head, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

if __name__ == '__main__':
    block = Attention(64).cuda()
    input_tensor = torch.rand(3, 64, 128, 128).cuda()
    output_tensor = block(input_tensor)
    # 打印输入和输出张量的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
