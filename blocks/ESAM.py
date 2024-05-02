import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""
ESAM被用作一个独立的网络模块，其主要功能是通过卷积层和梯度增强来处理输入特征图，目的是增强图像的边缘信息，从而更好地捕捉边缘细节。
通过这种方式，ESAM旨在将边缘信息集成到特征图中，帮助提高分割的精确度。
"""


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class ESAM(nn.Module):
    def __init__(self, in_channels):
        super(ESAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 保持通道数不变
        self.bn = nn.BatchNorm2d(in_channels)  # 用于conv1和conv2的输出
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, in_channels)  # 注意此处

    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = F.relu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        y = F.relu(self.bn(y))  # 使用self.bn而不是self.ban

        return y



if __name__ == '__main__':
    block = ESAM(in_channels=3)
    input = torch.rand(32, 3, 224, 224)
    output = block(input)
    print(input.size())
    print(output.size())