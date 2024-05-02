import torch
import torch.nn as nn
import torch.fft as fft

from einops import rearrange
from scipy.fftpack import next_fast_len

"""
近年来，人们一直在积极研究Transformer的时间序列预测。
虽然传统 Transformer 在各种场景中经常显示出有希望的结果，但其设计并不是为了充分利用时间序列数据的特性而设计的，
因此存在一些基本的局限性，例如，它们通常缺乏分解能力和可解释性，对于长期预测既无效也不高效。
在本文中，我们提出了一种新的时间序列 Transformer 架构 ETSFormer，它利用指数平滑原理改进 Transformer 进行时间序列预测。
特别是，受时间序列预测中经典指数平滑方法的启发，我们提出了新的指数平滑注意力（ESA）和频率注意力（FA）从而提高了准确性和效率。
"""

def conv1d_fft(f, g, dim=-1):
    N = f.size(dim)
    M = g.size(dim)

    fast_len = next_fast_len(N + M - 1)

    F_f = fft.rfft(f, fast_len, dim=dim)
    F_g = fft.rfft(g, fast_len, dim=dim)

    F_fg = F_f * F_g.conj()
    out = fft.irfft(F_fg, fast_len, dim=dim)
    out = out.roll((-1,), dims=(dim,))
    idx = torch.as_tensor(range(fast_len - N, fast_len)).to(out.device)
    out = out.index_select(dim, idx)

    return out


class ExponentialSmoothing(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1, aux=False):
        super().__init__()
        self._smoothing_weight = nn.Parameter(torch.randn(nhead, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, nhead, dim))
        self.dropout = nn.Dropout(dropout)
        if aux:
            self.aux_dropout = nn.Dropout(dropout)

    def forward(self, values, aux_values=None):
        b, t, h, d = values.shape

        init_weight, weight = self.get_exponential_weight(t)
        output = conv1d_fft(self.dropout(values), weight, dim=1)
        output = init_weight * self.v0 + output

        if aux_values is not None:
            aux_weight = weight / (1 - self.weight) * self.weight
            aux_output = conv1d_fft(self.aux_dropout(aux_values), aux_weight)
            output = output + aux_output

        return output

    def get_exponential_weight(self, T):
        # Generate array [0, 1, ..., T-1]
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)

        # (1 - \alpha) * \alpha^t, for all t = T-1, T-2, ..., 0]
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))

        # \alpha^t for all t = 1, 2, ..., T
        init_weight = self.weight ** (powers + 1)

        return rearrange(init_weight, 'h t -> 1 t h 1'), \
               rearrange(weight, 'h t -> 1 t h 1')

    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)

if __name__ == '__main__':
    # 创建输入数据：批量大小为1，序列长度为10，头数为4，特征维度为16
    input = torch.randn(1, 10, 4, 16)

    # 初始化ExponentialSmoothing模块
    # 假设我们的序列长度为10，头数为4，特征维度为16
    dim = 16
    nhead = 4
    block = ExponentialSmoothing(dim=dim, nhead=nhead, dropout=0.1)

    # 将输入数据传递给模块
    output = block(input)

    # 打印输出结果
    print(input.size())
    print(output.size())

