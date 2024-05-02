import torch
from torch import nn
"""
`GRN`（全局响应归一化）层是继承自`nn.Module`的PyTorch模块，它引入了一个可学习的归一化处理，该处理在输入特征图的空间维度上全局操作。

以下是类的细节解析：

1. `__init__` 方法初始化 `GRN` 类，接受一个参数 `dim`，这个参数指定了将会被该层归一化的输入张量的特征数或通道数。

2. 在 `__init__` 方法内部，您定义了两个可学习的参数 `gamma` 和 `beta`，它们被初始化为零。这些是归一化过程中将被学习的缩放和偏移参数。
这些参数以 `(1, 1, 1, dim)` 的形状初始化，以允许正确地在输入张量上进行广播。

3. `forward` 方法定义了归一化层的前向传播过程：
   - `Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)` 计算输入张量 `x` 在其空间维度（维度1和2）上的L2范数。
   `keepdim=True` 确保得到的张量与输入具有相同数量的维度，归一化维度的大小为1。
   - `Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)` 通过将 `Gx` 除以其在最后一个维度上的均值来计算 `Gx` 的归一化版本，为了避免除以零的情况，
   在此加上一个很小的数 epsilon（1e-6）。
   - 最后，该方法返回 `self.gamma * (x * Nx) + self.beta + x`。该操作将输入 `x` 与归一化的张量 `Nx` 相乘，
   然后用 `gamma` 缩放，用 `beta` 偏移，并最终加上原始输入 `x`。这最后一步表明了一种残差连接的形式，意味着归一化的值不是替换而是加到输入上，
   这有助于在从归一化中受益的同时保持原始特征。

确保在训练期间，这些可学习的参数会被更新，以达到期望的归一化效果。同时，在神经网络中部署这一层时，要注意输入大小与定义的 `dim` 参数之间的关系。
"""

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x