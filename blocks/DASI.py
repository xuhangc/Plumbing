import torch
import torch.nn as nn
import torch.nn.functional as F
"""
https://arxiv.org/pdf/2403.10778v1.pdf
HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection
Bag 模块 (Bag 类)：该模块根据动态参数 d，将两个输入 p 和 i 进行组合。它对 p 进行 sigmoid 函数运算后与 i 进行元素级乘法，并对 i 的补运算。

卷积块 (conv_block 类)：这是一个可定制的卷积块，包括一个卷积层，后面跟着可选的归一化（批归一化或组归一化）和激活函数（ReLU 或 GELU）。它是神经网络的基本构建块。

DASI 模块 (DASI 类)：这是网络的主要模块。它包括一个 Bag 模块、多个卷积层和跳跃连接。它接收三个输入 x、x_low 和 x_high，对它们执行一些操作，然后产生输出。

它具有跳跃连接 (skips, skips_2, skips_3) 以保留空间信息并帮助训练更深层次的网络。
它利用 Bag 模块 (bag) 来动态地组合低级和高级特征。
forward 方法处理数据通过模块的流程，包括处理跳跃连接和应用卷积操作。
"""
class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class DASI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)
         self.skips_2 = conv_block(in_features=in_features * 2,
                                 out_features=out_features,
                                 kernel_size=(1, 1),
                                 padding=(0, 0),
                                 norm_type=None,
                                 activation=False)
         self.skips_3 = nn.Conv2d(in_features//2, out_features,
                                  kernel_size=3, stride=2, dilation=2, padding=2)
         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()
         self.fc = nn.Conv2d(out_features, in_features, kernel_size=1, bias=False)

         self.gelu = nn.GELU()
    def forward(self, x , x_low, x_high):
        if x_high != None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x_skip = self.skips(x)
        x = self.skips(x)
        x = torch.chunk(x, 4, dim=1)
        if x_high == None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low == None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.fc(x)
        x = self.relu(x)

        return x
if __name__ == '__main__':
    # 定义输入数据
    batch_size = 1
    channels = 3
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, height, width)
    x_low = torch.randn(batch_size, channels * 2, height // 2, width // 2)
    x_high = torch.randn(batch_size, channels // 2, height * 2, width * 2)

    # 实例化 DASI 模块
    dasinet = DASI(channels, channels * 4)

    # 打印输入和输出的形状
    output = dasinet(x, x_low, x_high)
    print("输入 x 的形状:", x.shape)
    print("输入 x_low 的形状:", x_low.shape)
    print("输入 x_high 的形状:", x_high.shape)
    print("输出的形状:", output.shape)