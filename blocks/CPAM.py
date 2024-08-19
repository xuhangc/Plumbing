import torch
import torch.nn as nn
import math
#ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation(IMAVIS)
#https://arxiv.org/abs/2312.06458
"""
channel_att（通道注意力机制）
这种机制专注于增强最具信息性的通道：
自适应平均池化（avg_pool）：将空间维度减少到1x1，同时保持通道数量不变。
1D卷积（conv）：在通道维度上应用卷积操作，以捕获通道间的依赖关系。
Sigmoid激活（sigmoid）：将输出标准化到0到1之间。
前向传播涉及：
将输入张量池化到大小为（N, C, 1, 1）。
压缩并转置张量以应用1D卷积。
扩展输出，并将其与输入张量逐元素相乘。

local_att
这种机制旨在捕捉高度和宽度维度上的空间依赖关系：
1x1卷积（conv_1x1）：将通道维度按reduction因子减少。
ReLU激活（relu）和批量归一化（bn）：在1x1卷积后应用以实现非线性和归一化。
卷积（F_h、F_w）：分别在高度和宽度维度上应用以学习空间注意力图。
Sigmoid激活（sigmoid_h、sigmoid_w）：将空间注意力图标准化到0到1之间。
前向传播涉及：
计算沿宽度和高度维度的均值。
将这些值拼接并通过1x1卷积、ReLU激活和批量归一化。
分割生成的张量，并分别应用高度和宽度卷积。
生成空间注意力图，并将其逐元素应用于输入张量。

CPAM（通道和位置注意力机制）
这种机制结合了通道和局部注意力机制：
初始化：初始化channel_att和local_att实例。
前向传播：将通道注意力机制应用于第一个输入张量，添加第二个输入张量，然后将局部注意力机制应用于结果。
"""
# 通道注意力机制
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # 自适应平均池化
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # 1D卷积
        y = self.sigmoid(y)  # Sigmoid激活
        return x * y.expand_as(x)  # 通道逐元素相乘


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

#Channel and Position Attention Mechanism (CPAM)
class CPAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)
    def forward(self, x):
        input1,input2 = x[0],x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


if __name__ == '__main__':
    # 定义CPAM的输入通道数
    ch = 128

    # 实例化CPAM
    cpam = CPAM(ch)

    # 创建两个示例输入张量
    input1 = torch.randn(1, ch, 32, 32)  # 假设批量大小为1，通道数为ch，空间尺寸为32x32
    input2 = torch.randn(1, ch, 32, 32)

    # 将输入张量打包成列表传递给CPAM
    inputs = [input1, input2]

    # 获取CPAM的输出
    output = cpam(inputs)


    # 打印输入和输出的形状
    print("Input 1 shape:", input1.shape)
    print("Input 2 shape:", input2.shape)
    print("Output shape:", output.shape)