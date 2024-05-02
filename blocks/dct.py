# https://github.com/Zero-coder/FECAM/blob/main/layers/dctnet.py

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import numpy as np
import torch

"""
这段代码实现了一个名为 `dct_channel_block` 的 PyTorch 模块，该模块包含了离散余弦变换（DCT）和通道注意力机制的功能。

首先，定义了一个函数 `dct`，用于对输入的时间序列数据进行离散余弦变换。在函数内部，首先将输入的数据进行处理，然后利用快速傅里叶变换（FFT）相关的操作实现了频域的变换，并最终得到了变换后的频域数据。这个函数将在模块的正向传播过程中被调用，用来对每个通道的数据进行 DCT 变换。

接着，定义了一个名为 `dct_channel_block` 的 PyTorch 模块，它继承自 `nn.Module`。在初始化函数 `__init__` 中，该模块包含了一个神经网络模型，其中通过两个线性层和激活函数构成了一个全连接神经网络。此外，还定义了一个层归一化操作 `dct_norm`，用于对 DCT 变换后的频域数据进行归一化处理。

在前向传播函数 `forward` 中，输入数据 `x` 的形状为 `(B, C, L)`，其中 `B` 表示批次大小，`C` 表示通道数，`L` 表示时间序列的长度。首先对每个通道的数据分别调用之前定义的 `dct` 函数，得到频域数据，并将这些频域数据存储在一个列表中。接着，将列表中的频域数据堆叠起来，得到一个新的张量 `stack_dct`，其形状为 `(B, C, L)`。然后对 `stack_dct` 进行归一化处理，并通过前面定义的全连接神经网络模块 `fc` 对频域数据进行权重调整。最后，将输入数据 `x` 与调整后的权重 `lr_weight` 相乘，得到最终的输出结果。

总之，这个 `dct_channel_block` 模块实现了对输入数据进行离散余弦变换和通道注意力机制的功能，可以被用于深度学习模型中对时间序列数据的处理和特征提取。
"""

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.dct_norm = nn.LayerNorm([96], eps=1e-6)  # for lstm on length-wise

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = torch.tensor(stack_dct)
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight)

        return x * lr_weight  # result



if __name__ == '__main__':
    block = dct_channel_block(channel=96)
    input = torch.rand(8,7,96)
    output = block(input)
    print(input.size())
    print(output.size())