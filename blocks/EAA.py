import torch
import torch.nn as nn
import einops

#SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications[ICCV'23]
#https://openaccess.thecvf.com/content/ICCV2023/papers/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.pdf
"""
与传统的自注意力机制相比，SwiftTransformer的加法注意力机制消除了不必要的键值交互，仅关注查询和键之间的交互。
这减少了计算量，提高了推理速度，同时也保证了更好的上下文表示能力。

SwiftFormer的目标是设计一个高效、准确的图像分类模型，能够在多个尺度上学习局部和全局特征。
传统的自注意力机制在处理长序列时效率较低，而SwiftTransformer的加法注意力机制能够有效地减少计算量，提高推理速度。
同时，SwiftFormer还引入了一个新的卷积编码器，以进一步提高性能。这些改进使得SwiftFormer成为一个非常有潜力的图像分类模型。
"""
class EfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims, token_dim, num_heads=1):
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


if __name__ == '__main__':
    # 假设输入维度为512，token维度为512，头数为1
    attention_layer = EfficientAdditiveAttention(in_dims=512, token_dim=512)

    # 创建一个随机输入张量，形状为[B, N, D]
    B, N, D = 1, 10, 512
    x = torch.randn(B, N, D)

    # 通过注意力层传递输入
    output = attention_layer(x)

    # 打印输入和输出的形状
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)