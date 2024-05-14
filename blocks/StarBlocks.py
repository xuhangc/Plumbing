import torch
import torch.nn as nn
from timm.models.layers import DropPath
#Rewrite the Stars
#https://arxiv.org/pdf/2403.19967
"""
ConvBN类：
这个类创建了一个序列块，其中包含一个卷积层和一个可选的批次正则化层。
in_planes和out_planes是输入和输出通道的数量。
kernel_size，stride，padding，dilation和groups是配置卷积操作的参数。
with_bn是一个标志，指示是否在卷积后添加一个批量归一化层。
批量归一化的权重和偏差被初始化为1和0。
Block类：
继承自torch.nn.Module。
这个块有一个深度可分离的卷积模式（dwconv和dwconv2）和点卷积（f1，f2和g）。
深度可分离的卷积包含一个深度卷积，它对每个输入通道应用一个滤波器（groups=dim），然后是一个点卷积，混合通道。
mlp_ratio是一个超参数，控制块中间卷积的扩展大小。
act是一个激活函数，特别是ReLU6，它应用了一个截断在6的整流线性单元。
DropPath用于正则化，它随机丢弃完整的卷积路径，丢弃的概率由drop_path确定。
这个块的前向传播首先进行深度卷积，然后并行进行两次点卷积。然后通过乘法和ReLU6激活将x1和x2的结果特征组合在一起。
这是另一个深度卷积，特性被添加到块的输入上，有残余连接，有可选的drop path正则化。
"""
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

if __name__ == '__main__':
    # 假设输入特征图的大小为(batch_size, channels, height, width)
    input_tensor = torch.randn(1, 32, 64, 64)  # batch_size为1, 通道数为32, 图像尺寸为64x64

    # Block的实例化
    block = Block(dim=32, mlp_ratio=3, drop_path=0.1)  # 假定drop_path为0.1

    # 前向传递，获取输出
    output_tensor = block(input_tensor)

    # 打印输入输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")