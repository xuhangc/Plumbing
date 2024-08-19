import torch

# DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Indistinct-Boundary Object Segmentation
# https://arxiv.org/pdf/2311.00483
# Simplified 3D Spatial Attention(S3DSA)
class S3DSA(torch.nn.Module):
    def __init__(self, in_channels, spatial_dims=3):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels

        self.conv = torch.nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        out = x * attention
        return out


if __name__ == '__main__':
    # 实例化 S3DSA 模块
    in_channels = 16  # 输入通道数
    model = S3DSA(in_channels)

    # 创建一个示例输入张量
    input_tensor = torch.randn(1, in_channels, 32, 32, 32)  # (batch_size, channels, depth, height, width)

    # 打印输入的形状
    print("输入张量的形状:", input_tensor.shape)

    # 前向传播
    output_tensor = model(input_tensor)

    # 打印输出的形状
    print("输出张量的形状:", output_tensor.shape)