import torch
import torch.nn as nn
from math import sqrt

"""
由于其全局范围建模能力，Transformer 在时间序列预测方面表现出了强大的功能。然而，在非平稳的真实世界数据上，它们的性能可能会严重下降，在这些数据中，联合分布会随时间而变化。
以往的研究主要采用稳态化来减弱原始序列的非平稳性，以获得更好的可预测性。但是，被剥夺了固有非平稳性的稳态化序列对于现实世界的突发事件预测可能不太有指导意义。这个问题在本文中被称为过度平稳化，导致Transformers对不同序列产生难以区分的时间注意力，并阻碍了深度模型的预测能力。
为了解决级数可预测性和模型能力之间的困境，我们提出非稳态变压器作为一个通用框架，具有两个相互依赖的模块：级数稳态化和去稳态注意力(De-stationary Attention)。
具体而言，级数稳态化统一了每个输入的统计数据，并将输出与恢复的统计数据进行转换，以实现更好的可预测性。
为了解决过度平稳化问题，设计了去平稳注意力，通过近似从原始序列中学习到的可区分注意力，将内在的非平稳信息恢复为时间依赖关系。
"""

class DSAttention(nn.Module):
    '''De-stationary Attention'''
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag and attn_mask is not None:
            # 直接使用attn_mask进行掩码操作
            scores.masked_fill_(attn_mask == float('-inf'), float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



def generate_square_subsequent_mask(size):
    """生成序列掩码"""
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask


if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    n_heads = 4  # 注意力头数
    d_model = 16  # 特征维度


    # 模拟输入数据
    queries = torch.rand(batch_size, seq_len, n_heads, d_model)
    keys = torch.rand(batch_size, seq_len, n_heads, d_model)
    values = torch.rand(batch_size, seq_len, n_heads, d_model)

    # 生成注意力掩码
    attn_mask = generate_square_subsequent_mask(seq_len).to(queries.device)

    # 实例化DSAttention模块
    block = DSAttention()

    # 运行前向传播
    output, _ = block(queries, keys, values, attn_mask)  # 假设我们不需要输出注意力矩阵

    print('Output size:', output.size())

