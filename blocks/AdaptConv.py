# https://github.com/hrzhou2/AdaptConv-master/blob/main/cls/model_cls.py

import torch
import torch.nn as nn

"""
标准卷积在处理3D点云数据时无法有效区分点之间的特征对应关系，从而限制了其在学习独特特征方面的能力。
为了解决这一问题，提出了AdaptConv，它通过为每个点生成基于动态学习特征的自适应内核，从而提高了点云卷积的灵活性和准确性。

通过两个卷积层conv0和conv1，其中conv0将辅助特征y转换成中间特征，然后conv1进一步将这些特征转换为与输入x的通道数相乘的输出，以生成自适应内核。
这种自适应性允许网络更有效地捕获来自不同语义部分的点之间的多样化关系，而不是简单地为相邻点分配不同的权重。
AdaptConv在点云分类和分割任务中突出了优越性。
"""

class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels * in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        # x: (bs, in_channels, num_points, k), y: (bs, feat_channels, num_points, k)
        batch_size, n_dims, num_points, k = x.size()

        y = self.conv0(y)  # (bs, out, num_points, k)
        y = self.leaky_relu(self.bn0(y))
        y = self.conv1(y)  # (bs, in*out, num_points, k)
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels,
                                       self.in_channels)  # (bs, num_points, k, out, in)

        x = x.permute(0, 2, 3, 1).unsqueeze(4)  # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(y, x).squeeze(4)  # (bs, num_points, k, out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()  # (bs, out_channels, num_points, k)

        x = self.bn1(x)
        x = self.leaky_relu(x)

        return x

if __name__ == '__main__':
    bs = 2
    in_channels = 16
    out_channels = 16
    feat_channels = 64  # 特征通道数，表示辅助输入数据y的特征或维度数。
    num_points = 1024
    k = 20

    block = AdaptiveConv(in_channels, out_channels, feat_channels)
    x = torch.rand(bs, in_channels, num_points, k)
    y = torch.rand(bs, feat_channels, num_points, k)

    output = block(x, y)

    print("Input x size:", x.size())
    print("Input y size:", y.size())
    print("Output size:", output.size())

