from torch import nn
import torch
import torch.nn.functional as F

"""
https://arxiv.org/pdf/1810.11579.pdf
DoubleAttentionLayer 类：该类定义了双重注意力层的架构。

构造函数 (__init__)：初始化双重注意力层的参数，
包括输入通道数 (in_channels)、查询向量 (c_m) 和键向量 (c_n) 的维度，
以及一个名为 reconstruct 的标志，指示是否重构输出。
它还初始化了三个卷积层 (convA、convB、convV)，用于计算注意力图和向量。
如果 reconstruct 设置为 True，则初始化另一个卷积层 (conv_reconstruct) 用于重构输出。

前向方法 (forward)：接受形状为 (batch_size, in_channels, height, width) 的输入张量 x，
并通过双重注意力层执行前向传递。

首先，将三个卷积操作 (convA、convB、convV) 应用于输入张量，以获得查询张量 A、键张量 B 和值张量 V。

然后，将 A、B 和 V 张量重塑以准备进行注意力计算。

通过在 B 张量的最后一个维度上应用 softmax，计算注意力图。

通过计算 tmpA 和注意力图的加权和，执行特征收集。

通过在 V 张量的第二个维度上应用 softmax，计算注意力向量。

通过计算全局描述符和注意力向量的加权和，执行特征分配。

如果设置了 reconstruct 标志，则重构并重塑输出。

返回最终的输出张量。
"""
class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """

    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct=False):
        """
        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super(DoubleAttentionLayer, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size=1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size=1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size=1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)
        Returns
        -------
        """
        batch_size, c, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1

        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)

        tmpA = A.view(batch_size, self.c_m, h * w)

        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)

        # softmax on the last dimension to create attention maps
        attention_maps = F.softmax(attention_maps, dim=-1)  # 对hxw维度进行softmax

        # step 1: feature gathering
        global_descriptors = torch.bmm(  # attention map(V)和tmpA进行
            tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)

        # step 2: feature distribution
        # (B, c_n, h * w) attention on c_n dimension - channel wise
        attention_vectors = F.softmax(attention_vectors, dim=1)

        tmpZ = global_descriptors.matmul(
            attention_vectors)  # B, self.c_m, h * w

        tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


if __name__ == "__main__":
    x = torch.zeros(3, 12, 8, 8)
    model = DoubleAttentionLayer(12, 24, 4)
    print(model(x).shape)