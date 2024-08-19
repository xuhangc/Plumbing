import torch
import torch.nn as nn
from torch import Tensor
#https://arxiv.org/abs/2303.03667
#Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
"""
这段代码定义了一个名为`Partial_conv3`的类，它继承自`torch.nn.Module`，用于创建一个具有部分卷积操作的模块。
这种模块特别适合处理输入特征图的一部分，而不是全部特征，这可以在一些特别的神经网络结构中找到应用，
比如需要分别处理特征图的不同区域时。以下是对代码每部分的详细解释：

1. **类定义与初始化 (`__init__` 方法)：**
   - `dim`: 输入特征图的通道数。
   - `n_div`: 用于决定如何分割通道的因子。`dim` 会被 `n_div` 整除，从而确定用于卷积的通道数(`dim_conv3`)和不变通道数(`dim_untouched`)。
   - `forward`: 指定前向传播的方式，支持`'slicing'`和`'split_cat'`两种方法。
   - `self.dim_conv3 = dim // n_div`: 计算进行卷积操作的通道数。
   - `self.dim_untouched = dim - self.dim_conv3`: 计算不参与卷积操作，保持不变的通道数。
   - `self.partial_conv3 = nn.Conv2d(...)`: 创建一个2D卷积层，仅适用于`dim_conv3`指定的部分特征图。
   
2. **前向传播方法 (`forward_slicing` 方法)：**
   - 这种方法采用“切片”，仅对输入特征图的前`dim_conv3`通道应用卷积，其余通道保持不变。
   - 使用`x.clone()`确保原输入`x`不被修改，这对于后续结构（如残差连接）可能很重要。
   - 通过`x[:, :self.dim_conv3, :, :]`选取前`dim_conv3`通道进行卷积处理。
   
3. **另一种前向传播方法 (`forward_split_cat` 方法)：**
   - 这种方法通过`torch.split`先将输入`x`在通道维度上分割为两部分，一部分用于卷积，另一部分保持不变。
   - 分割后的第一部分应用卷积操作，然后通过`torch.cat((x1, x2), 1)`与未变动的第二部分重新合并。
   - 这种方式适用于训练和推断，能够保持一部分特征图不受卷积层影响。
   
4. **特点与用途：**
   - 这个模块通过分割的方式，使得网络能够只对特征图的一部分应用卷积操作，而保留其他部分的原始信息。这在如需要专门针对特征图的一部分进行增强或抑制时非常有用。
   - 根据构造函数中`forward`参数的设定，可以选择`'slicing'`或`'split_cat'`作为前向传播的策略。这增加了模块的灵活性，允许用户根据需要选择合适的操作方式。

总之，这段代码提供了一个灵活的卷积模块，可以根据输入特征图的实际需求，有选择性地对其进行部分卷积处理，这在特定的应用场景中可能非常有用。
"""
class EnhancedPartialConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, n_div=2, bias=False,
                 method='split_cat'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_div = n_div
        self.method = method



        # 计算进行卷积处理部分的通道数
        self.dim_conv = in_channels // n_div
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.dim_untouched = in_channels - self.dim_conv

        # 创建卷积层
        self.conv = nn.Conv2d(self.dim_conv, out_channels, kernel_size, stride, padding, bias=bias)

        if method not in ['slicing', 'split_cat']:
            raise ValueError("Unsupported forward method. Use 'slicing' or 'split_cat'.")

    def forward(self, x: Tensor) -> Tensor:
        if self.method == 'slicing':
            return self.forward_slicing(x)
        elif self.method == 'split_cat':
            return self.forward_split_cat(x)

    def forward_slicing(self, x: Tensor) -> Tensor:
        untouched_conv = nn.Conv2d(self.dim_untouched, self.out_channels, 3, 1, 1, bias=False)

        untouched_output = untouched_conv(x[:, self.dim_conv:, :, :])

        conv_output = self.conv(x[:, :self.dim_conv, :, :])

        # 合并两个输出
        output = torch.add(conv_output, untouched_output)
        return output

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        Conv_1x1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        x = Conv_1x1(x)

        return x


if __name__ == '__main__':
    # 示例代码
    x = torch.rand((1, 16, 32, 32))
    model = EnhancedPartialConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, n_div=4, bias=False,
                                  method='split_cat')
    output = model(x)

    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output.shape}')