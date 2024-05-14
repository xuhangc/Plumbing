import torch.nn as nn
import numpy as np
import torch
#FECAM: Frequency Enhanced Channel Attention Mechanism for Time Series Forecasting
#https://arxiv.org/abs/2212.01209
"""
这段Python代码定义了一个特别设计用于时间序列预测的频率增强通道注意力机制，被命名为FECAM。
它利用离散傅里叶变换和余弦变换处理时间序列数据。这个机制的关键组成部分包括自定义的离散余弦变换（DCT）函数、一个应用这种变换的DCT通道块模块，
以及通过一系列全连接（FC）层实现的通道注意力机制。

以下是主要组成部分和功能的概述：

1. **离散余弦变换（DCT）：** 输入信号的自定义DCT实现。
这部分代码负责将信号从时间域转换到频率域，这对于提取信号的频率成分特别有用。值得注意的是，这种DCT的实现特别是执行了DCT的第二型变换。

2. **rfft和irfft兼容性：** 由于PyTorch版本之间可能存在的差异，
代码尝试直接从PyTorch中导入`rfft`和`irfft`。
如果这些函数不可用（在新版本中，自PyTorch 1.8起，`torch.fft`已替代这些函数），
它会使用较新的`torch.fft`模块来定义它们。这确保了代码跨不同的PyTorch版本兼容。

3. **DCT通道块模块：** 定义为一个`nn.Module`，这个组件利用DCT构建了实际的通道注意力机制。它包括一系列操作，
从对输入的各个通道应用DCT开始，接着是层归一化，然后通过一个全连接神经网络处理，产生通道级别的注意力权重。
这些注意力权重用于缩放原始输入，强调对时间序列预测任务更有信息量的通道。

4. **示例用法：** 脚本以一个`dct_channel_block`类的实例化及其应用于随机张量的示例结束，模拟了一个时间序列数据的批量。
这演示了如何将该机制集成到预测模型管道中。

这段代码展示了一个将频率域分析与现代深度学习范式巧妙结合的创新方法，用于增强时间序列预测。
特别值得注意的是使用DCT生成通道级别注意力权重的方法，它巧妙地将经典信号处理技术与现代深度学习范式桥接。
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
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

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
        b, c, l = x.size()  # (B,C,L) (32,96,512)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)

        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(lr_weight)
        lr_weight = self.dct_norm(lr_weight)

        return x * lr_weight  # result


if __name__ == '__main__':
    tensor = torch.rand(8, 7, 96)
    dct_model = dct_channel_block(96)
    result = dct_model(tensor)
    print("input_tensor.shape:", tensor.shape)
    print("result.shape:", result.shape)