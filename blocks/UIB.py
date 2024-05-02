import torch.nn as nn
from typing import Optional
#https://arxiv.org/pdf/2404.10518.pdf
#MobileNetV4 - Universal Models for the Mobile Ecosystem
"""
这段代码是由几个部分构成的，主要用来构建深度学习模型中的卷积层，以及一个名为“UniversalInvertedBottleneckBlock”的模块。以下是对代码的详细解释：

1. `make_divisible` 函数：
   这个函数用来确保计算出来的值可以被选择的除数整除。在深度学习模型中，特别是在硬件效率要求较高的情况下，这一点非常重要。

   参数说明：
   - `value`: 浮点型，原始需要被整除的值。
   - `divisor`: 整型，需要能被该数整除。
   - `min_value`: 浮点型，阈值，用来确保计算出值不会小于这个最小值。
   - `round_down_protect`: 布尔型，保护机制，用来防止向下取整超过10%。

   功能：
   计算一个值，该值接近`value`但被`divisor`整除，并且尽量不减少超过10%的`value`。

2. `conv_2d` 函数：
   这个函数创建了一个二维卷积层，可能包含批量归一化和ReLU6激活函数。

   参数说明：
   - `inp`: 输入通道数。
   - `oup`: 输出通道数。
   - `kernel_size`: 卷积核的大小，默认为3。
   - `stride`: 卷积的步长，默认为1。
   - `groups`: 控制输入和输出之间的连接，分组卷积中的组数。
   - `bias`: 是否加上偏置项，默认为False。
   - `norm`: 是否添加批量归一化，默认为True。
   - `act`: 是否添加激活函数，默认为True。

3. `UniversalInvertedBottleneckBlock` 类：
   这是一个PyTorch模块，它使用了上述的方法来组建一个倒置的瓶颈块。一个典型的倒置瓶颈块包含一个扩展卷积（提高通道数），
   一个深度卷积（depthwise convolution），以及一个投影卷积（降低通道数）。

   参数说明：
   - `inp`: 输入通道数。
   - `oup`: 输出通道数。
   - `start_dw_kernel_size`: 开始层深度卷积的核大小。
   - `middle_dw_kernel_size`: 中间层深度卷积的核大小。
   - `middle_dw_downsample`: 是否在中间深度卷积降采样。
   - `stride`: 卷积的步长。
   - `expand_ratio`: 扩展比率，用于增加通道数。

   `forward` 方法：
   这个方法定义了输入数据通过该模块时的逻辑顺序，它通过可选的开始深度卷积，扩展卷积，中间深度卷积，和投影卷积来传递数据。

这段代码在机器学习和模型设计中是非常实用的，尤其是在设计适用于移动设备或需要高效计算的深度学习模型时。
`UniversalInvertedBottleneckBlock` 类中的三个卷积层分别承担不同的作用，它们是基于MobileNetV2中提出的倒置残差结构的概念。这些层分别是：

1. **扩展卷积层（Expansion Convolution）**:
   - **作用**：这一层主要是使用1x1卷积核来扩大（expand）特征图的通道数，使之增加。
   这是为了通过增加通道数来提升特征的表达能力，在接下来的深度卷积操作中可以捕获更多的信息。
   - **位置**：位于开始深度卷积（如果有的话）之后。对于输入通道`inp`和扩张比率`expand_ratio`，输出通道数为`inp * expand_ratio`。

2. **深度卷积层（Depthwise Convolution）**:
   - **作用**：深度卷积层通过对每个输入通道独立应用卷积来执行空间过滤，这种操作方式显著减少了参数量和计算成本。
   它不改变通道数，但是可以提取特征，因此增强了模型在捕捉空间特征上的能力。
   - **位置**：在扩展卷积层之后。如果有`middle_dw_downsample`，表示是否在这层降采样。

3. **投影卷积层（Projection Convolution）**:
   - **作用**：该层使用1x1卷积核将扩展后的通道数降低（project），通常用于将通道数降低至目标输出通道数`oup`。
   这种设计减少了模型的复杂性和计算需求，同时保持了关键特征信息，有助于提升网络的效率。
   - **位置**：在深度卷积层后。它基本上是完成了特征转换和通道压缩的角色。

综上，这三个层的搭配使用能够在减少了计算量和参数数量的同时，有效地提升了模型对特征的提取和表示能力。
这种设计思想特别适合于需要在计算资源受限的环境下运行的应用，如移动和边缘设备上的深度学习模型。
"""
def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"

    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio
                 ):
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


if __name__ == "__main__":
    model = UniversalInvertedBottleneckBlock(inp=32,
                                             oup=64,
                                             start_dw_kernel_size=3,
                                             middle_dw_kernel_size=3,
                                             middle_dw_downsample=True,
                                             stride=1,
                                             expand_ratio=6)
    print(model)

    # 在虚拟数据上测试模型
    import torch
    x = torch.randn(1, 32, 224, 224)  # 这里的1是batch_size，32是通道数，224x224是图像的高和宽
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)