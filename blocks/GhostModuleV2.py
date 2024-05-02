import torch.nn as nn
import math
import torch
import torch.nn.functional as F
"""
这段代码定义了一个名为 `GhostModuleV2` 的 PyTorch 模块，
它实现了 GhostNet 论文中介绍的 Ghost Module 的变体。
这个版本被标记为 "V2"，其中包括了一个注意力机制。

以下是关键组件和功能的详细说明：

1. **初始化 (`__init__`)**:
   - 构造函数使用各种参数初始化 Ghost Module，如输入通道数 (`inp`)、输出通道数 (`oup`)、
   卷积核大小、比率、深度可分离卷积大小 (`dw_size`)、步长以及是否应用 ReLU 激活 (`relu`)。
   - 支持两种模式：`'original'` 和 `'attn'`。根据模式的不同，在模块内执行不同的操作。

2. **前向传播 (`forward`)**:
   - 如果模式是 `'original'`，则执行原始 Ghost Module 操作：
     - 应用主要卷积，然后进行批归一化和 ReLU 激活。
     - 执行 "cheap" 操作，即深度可分离卷积，然后进行批归一化和 ReLU 激活。
     - 将主要卷积和 cheap 操作的输出连接起来。
   - 如果模式是 `'attn'`，则包括一个注意力机制：
     - 应用一系列卷积 (`short_conv`) 来降低输入的维度并提取注意力特征。
     - 然后，执行与原始模式相同的操作。
     - 最后，应用一个注意力门，根据注意力特征调节 Ghost Module 的输出。

3. **主函数 (`__main__`)**:
   - 它实例化了一个具有指定输入和输出通道以及模式 (`'attn'` 在本例中) 的 `GhostModuleV2` 实例。
   - 生成了一个随机输入张量 (`in_tensor`)，形状为 `(2, 128, 64, 64)`。
   - 将输入张量通过 `GhostModuleV2` 实例 (`model`)，得到输出张量。
   - 打印输出张量的形状。

4. **用途**:
   - `GhostModuleV2` 模块可以集成到更大的神经网络架构中，通常在卷积神经网络 (CNN) 中。
   它提供了一种减少计算和参数数量的方式，同时保持表示容量。

5. **输出形状**:
   - 输出张量的形状取决于输出通道数 (`oup`)。
   在这种情况下，输出张量的形状将是 `(2, 256, 64, 64)`，因为输出通道数被指定为 256。
"""

class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=out.shape[-1], mode='nearest')
if __name__ == "__main__":
    model = GhostModuleV2(128, 128, mode="attn")#attn=Ghost Module V2, original=Ghost Module

    in_tensor = torch.randn(2, 128, 64, 64)

    out_tensor = model(in_tensor)

    print(out_tensor.shape)