from torch import nn
import torch
import torch.nn.functional
import torch.nn.functional as F
# Channel prior convolutional attention for medical image segmentation
# https://arxiv.org/pdf/2306.05196
"""
这段代码定义了两个神经网络模块：ChannelAttention和CPCABlock。它们是基于通道注意力机制的，适用于像素级别的分类任务，如医学图像分割。
以下是每个类的主要部分：
ChannelAttention 类
目的：实现通道注意力机制，它挑选出输入特征的重要通道，并学习特征间的依赖关系。
初始化参数：
input_channels：输入特征的通道数。
internal_neurons：内部神经元的数量，这是进行通道注意力计算时的降维大小。
组成部分：
两个卷积层：第一个用于降低特征维度，第二个用于恢复到输入特征维度。
前向传播：
分别对输入特征进行全局平均池化和全局最大池化。
经过减少和恢复特征维度的卷积层后，应用ReLU和sigmoid激活函数。
计算平均池化和最大池化得到的注意力分布的和，并将其加权到输入特征。
CPCABlock 类
目的：实现了一个使用通道注意力的卷积处理块。
初始化参数：
in_channels和out_channels：输入和输出特征的通道数。需要注意的是，这两个数应该相等。
channelAttention_reduce：计算通道注意力时的降维比率。
组成部分：
一个ChannelAttention模块，用于加权输入特征的通道。
多个卷积层，包括深度卷积层和一个1x1卷积，实现了特征的交互和整合。
前向传播：
首先，使用一个1x1卷积和激活函数对输入特征进行处理。
接下来，用ChannelAttention模块对处理后的特征进行权重处理。
然后，通过多个不同核大小的卷积层对特征进行空间处理并相加。
最后，结合通道和空间的注意力权重，并使用1x1卷积生成输出。
为了适应不同大小的感受域，CPCABlock在空间注意力计算时使用了5x5、7x7、11x11和21x21四种大小的卷积。
这使得模块能够捕捉到从小范围到大范围的上下文信息。该模块可用于搭建各种针对医学图像区域分类或分割的深度网络模型。
"""

class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class CPCABlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


if __name__ == '__main__':
    # 假设输入数据
    # 定义批次大小为 4，通道数为 16，图像的高和宽分别为 64x64
    batch_size = 4
    channels = 16
    height = 64
    width = 64
    input_tensor = torch.randn(batch_size, channels, height, width)

    # 打印输入张量的形状
    print(f"Input shape: {input_tensor.shape}")

    # 初始化CPCABlock模块
    # 假定输入和输出通道数相同为16，并设定通道注意力降维比率为4
    cpca_block = CPCABlock(in_channels=16, out_channels=16, channelAttention_reduce=4)

    # 通过CPCABlock模块处理输入
    output_tensor = cpca_block(input_tensor)

    # 打印输出张量的形状
    print(f"Output shape: {output_tensor.shape}")