import torch
import torch.nn as nn
#CBAM: Convolutional Block Attention Module
#https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
"""
ChannelAttention3D 类: 类似于2D版本，通过自适应平均池化和最大池化来捕捉通道间的注意力。利用两个3D卷积层来学习注意力权重。
SpatialAttention3D 类: 类似于2D版本，通过平均池化和最大池化来捕捉空间注意力。利用一个3D卷积层来学习注意力权重。
CBAM3D 类: 结合 ChannelAttention3D 和 SpatialAttention3D 模块，先应用通道注意力，再应用空间注意力。
"""
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
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

class CBAM3D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_planes, ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x_out = self.channel_attention(x) * x
        x_out = self.spatial_attention(x_out) * x_out
        return x_out


if __name__ == '__main__':
    # 实例化CBAM模块
    cbam_3d = CBAM3D(in_planes=64)

    # 创建一个模拟输入的张量
    input_tensor_3d = torch.randn(1, 64, 16, 32, 32)  # Batch size = 1, Channels = 64, Depth = 16, Height = 32, Width = 32

    # 打印输入的shape
    print(f'输入的shape: {input_tensor_3d.shape}')

    # 通过CBAM模块
    output_tensor_3d = cbam_3d(input_tensor_3d)

    # 打印输出的shape
    print(f'输出的shape: {output_tensor_3d.shape}')
