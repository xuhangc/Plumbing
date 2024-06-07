import torch
import torch.nn as nn
#SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation
#https://ieeexplore.ieee.org/document/9895210
#local pyramid attention (LPA) module
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention3D, self).__init__()
        # 使用3D版本的平均池和最大池
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention3D, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LPA3D(nn.Module):
    def __init__(self, in_channel):
        super(LPA3D, self).__init__()
        self.ca = ChannelAttention3D(in_channel)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [[self.ca(xx0) * xx0 for xx0 in x0_s.chunk(2, dim=4)] for x0_s in x0]
        x1 = [[self.ca(xx1) * xx1 for xx1 in x1_s.chunk(2, dim=4)] for x1_s in x1]
        x0 = [torch.cat(x0_s, dim=4) for x0_s in x0]
        x1 = [torch.cat(x1_s, dim=4) for x1_s in x1]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4
        x = x3 + x4
        return x


if __name__ == '__main__':
    # 输入数据的新维度，增加了深度维度
    batch_size = 1
    channels = 128
    depth = 32  # 新增的深度维度
    height = 64
    width = 64

    # 初始化一个符合这些维度的随机张量作为输入
    input_tensor = torch.rand(batch_size, channels, depth, height, width)

    # 实例化3D版本的LPA模块
    lpa3d = LPA3D(in_channel=channels)

    # 执行前向传播
    output_tensor = lpa3d(input_tensor)

    # 打印输入和输出的shape
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")