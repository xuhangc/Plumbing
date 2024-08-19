import torch
import torch.nn as nn
"""
SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation
https://arxiv.org/pdf/2401.13560
在`__init__`方法中，初始化了几个卷积层 (`nn.Conv3d`)，实例归一化层 (`nn.InstanceNorm3d`) 和ReLU激活层 (`nn.ReLU`)。
这些按照特定的顺序堆叠起来，形成了一个残差块，包括一个主路径（`self.proj`, `self.norm`, `self.nonliner`, `self.proj2`, `self.norm2`, `self.nonliner2`）
和一个快捷路径（`self.proj3`, `self.norm3`, `self.nonliner3`），然后它们被合并在一起（`x1 + x2`），
通过另一个层序列（`self.proj4`, `self.norm4`, `self.nonliner4`）然后加到原来的输入上（`x + x_residual`），这是在残差网络中找到的典型模式。
`forward`函数正确地实现了网络的向前传播，将每一层应用到输入张量 `x` 上，然后将最后投影的输出与原始输入张量相加，形成残差连接。
"""
class GSC3d(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual

if __name__ == '__main__':
    # 实例化模型，这里的假设输入通道数为32
    model = GSC3d(in_channles=32)

    # 创建一个随机输入张量，形状为(batch_size, channels, depth, height, width)
    input_tensor = torch.randn(1, 32, 64, 64, 64)

    # 计算模型的前向传播
    output_tensor = model(input_tensor)

    # 打印输入和输出张量的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
