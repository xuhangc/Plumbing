import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
用于高光谱图像去噪的混合注意力网络

高光谱图像去噪对于高度相似和相关的光谱信息来说是独一无二的，应该适当考虑。
然而，现有方法在探索不同波段的光谱相关性和每个波段内的特征相互作用方面存在局限性。
此外，低级和高级特征通常对不同的空间光谱区域表现出不同的重要性，这在当前的算法中也没有得到充分的探索。
在本文中，我们提出了一个混合注意力网络（MAN），该网络同时考虑了光谱间和光谱内的相关性以及低级和高级空间光谱有意义的特征之间的相互作用。
具体来说，我们引入了一种多头递归光谱注意力，可以有效地整合所有光谱波段的光谱间特征。通过探索光谱内关系，渐进式光谱通道注意力进一步增强了这些特征。
此外，我们提出了一种细心的跳跃连接，自适应地控制来自编码器和解码器的低级和高级空间光谱特征的比例，以更好地增强聚合特征。
大量实验表明，我们的MAN在模拟和真实噪声设置方面优于现有的先进方法，同时保持了较低的参数成本和运行时间。
"""

# class ASC(nn.Module):
#     """ Attentive Skip Connection
#     """
#
#     def __init__(self, channel):
#         super().__init__()
#         self.weight = nn.Sequential(
#             nn.Conv3d(channel * 2, channel, 1),
#             nn.LeakyReLU(),
#             nn.Conv3d(channel, channel, 3, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x, y):
#         w = self.weight(torch.cat([x, y], dim=1))
#         out = (1 - w) * x + w * y
#         return out
#
#
# if __name__ == '__main__':
#     # 定义输入张量的大小参数
#     batch_size = 1
#     channels = 64
#     depth = 32
#     height = 32
#     width = 32
#
#     # 创建ASC模块实例
#     block = ASC(channel=channels)
#
#     # 生成随机输入张量
#     input1 = torch.rand(batch_size, channels, depth, height, width)
#     input2 = torch.rand(batch_size, channels, depth, height, width)
#
#     # 通过ASC模块传递输入
#     output = block(input1, input2)
#
#     # 打印输入和输出张量的大小
#     print(f'Input1 size: {input1.size()}')
#     print(f'Input2 size: {input2.size()}')
#     print(f'Output size: {output.size()}')




class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP)
    """

    def __init__(self, channel, bias=True):
        super().__init__()
        self.w_1 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)
        self.w_2 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)

    def forward(self, x):
        return self.w_2(F.tanh(self.w_1(x)))


class MHRSA(nn.Module):
    """ Multi-Head Recurrent Spectral Attention
    """

    def __init__(self, channels, multi_head=True, ffn=True):
        super().__init__()
        self.channels = channels
        self.multi_head = multi_head
        self.ffn = ffn

        if ffn:
            self.ffn1 = MLP(channels)
            self.ffn2 = MLP(channels)

    def _conv_step(self, inputs):
        if self.ffn:
            Z = self.ffn1(inputs).tanh()
            F = self.ffn2(inputs).sigmoid()
        else:
            Z, F = inputs.split(split_size=self.channels, dim=1)
            Z, F = Z.tanh(), F.sigmoid()
        return Z, F

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        Z, F = self._conv_step(inputs)

        if self.multi_head:
            Z1, Z2 = Z.split(self.channels // 2, 1)
            Z2 = torch.flip(Z2, [2])
            Z = torch.cat([Z1, Z2], dim=1)

            F1, F2 = F.split(self.channels // 2, 1)
            F2 = torch.flip(F2, [2])
            F = torch.cat([F1, F2], dim=1)

        h = None
        h_time = []

        if not reverse:
            for _, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for _, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        y = torch.cat(h_time, dim=2)

        if self.multi_head:
            y1, y2 = y.split(self.channels // 2, 1)
            y2 = torch.flip(y2, [2])
            y = torch.cat([y1, y2], dim=1)

        return y


if __name__ == '__main__':
    batch_size = 1
    channels = 64
    depth = 10
    height = 32
    width = 32

    # 初始化 MHRSA 类的实例
    mhrsa = MHRSA(channels=channels)

    # 创建随机的3D输入数据
    inputs = torch.randn(batch_size, channels, depth, height, width)

    # 将输入数据传递给 MHRSA 网络
    outputs = mhrsa(inputs)

    # 打印输入和输出数据的形状
    print(f'Input shape: {inputs.shape}')
    print(f'Output shape: {outputs.shape}')