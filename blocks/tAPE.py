import math
import torch
import torch.nn as nn

"""
Transformer 在深度学习的许多应用中表现出了出色的性能。当应用于时间序列数据时，变压器需要有效的位置编码来捕获时间序列数据的顺序。
位置编码在时间序列分析中的功效尚未得到充分研究，并且仍然存在争议，例如，注入绝对位置编码或相对位置编码或它们的组合是否更好。为了澄清这一点，我们首先回顾一下应用于时间序列分类时现有的绝对和相对位置编码方法。
然后，我们提出了一种专用于时间序列数据的新绝对位置编码方法，称为时间绝对位置编码（tAPE）。我们的新方法将序列长度和输入嵌入维度合并到绝对位置编码中。
"""


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=32, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)

if __name__ == '__main__':
    block = tAPE(d_model=512)
    input = torch.rand(20, 32, 512) # [sequence length, batch size, embed dim]
    output = block(input)
    print(input.size())
    print(output.size())
