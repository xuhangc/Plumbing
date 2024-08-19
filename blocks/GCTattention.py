import torch
from torch import nn
#title：Gated Channel Transformation for Visual Recognition
#https://arxiv.org/abs/1909.11519
"""
门控通道变换（Gated Channel Transformation）。
该类设计用于实现一种可学习的变换，通过计算得到的门控机制对输入张量进行缩放。这种门控机制可以以两种模式运行：l2 范数或 l1 范数。
有 alpha、gamma 和 beta 三个可学习参数，这些参数参与门控机制。
参数 alpha 用于缩放输入张量的计算范数，gamma 用于对这种缩放因子进行标准化，而 beta 是在计算门时的 tanh 激活函数中的偏移项。
在 forward 方法中，根据输入张量 x 的平方（l2）或绝对值（l1）来计算嵌入。
然后，使用这个嵌入来计算规范化的门控值 norm。gate 通过 tanh 激活函数计算，这是一个非线性函数，输出值在 -1 到 1 之间。该模块的输出是输入张量乘以这个门。
使用这种门控变换的目的可能是为了基于它们的范数，自适应地重新校准通道级特征响应，这对于对象检测或图像分类等任务可能很有益，这些任务中动态特征规模很重要。
存在 after_relu 标志表明，可以选择在 ReLU 激活函数之前或之后应用 l1 范数，ReLU 是一个非线性函数，将所有负值设为零
"""
# 定义 GCT 模块
class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x) if not self.after_relu else x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


if __name__ == '__main__':
    # 创建一个具有确定通道数的随机输入张量
    num_channels = 16  # 示例通道数
    input_tensor = torch.randn(1, num_channels, 32, 32)  # 示例批次大小为1，32x32的特征图

    # 打印输入张量的形状
    print(f'输入张量的形状: {input_tensor.shape}')

    # 实例化 GCT 模块
    gct = GCT(num_channels=num_channels)

    # 前向传播
    output_tensor = gct(input_tensor)

    # 打印输出张量的形状
    print(f'输出张量的形状: {output_tensor.shape}')