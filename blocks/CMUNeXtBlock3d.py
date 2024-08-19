import torch
import torch.nn as nn
#CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion
#https://arxiv.org/pdf/2308.01239
class conv_block_3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Residual3d(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class CMUNeXtBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock3D, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual3d(nn.Sequential(
                    # depth wise
                    nn.Conv3d(ch_in, ch_in, kernel_size=(k, k, k), groups=ch_in, padding=(k // 2, k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm3d(ch_in)
                )),
                nn.Conv3d(ch_in, ch_in * 4, kernel_size=(1, 1, 1)),
                nn.GELU(),
                nn.BatchNorm3d(ch_in * 4),
                nn.Conv3d(ch_in * 4, ch_in, kernel_size=(1, 1, 1)),
                nn.GELU(),
                nn.BatchNorm3d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block_3d(ch_in, ch_out)
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)#减半

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        x = self.Maxpool(x)
        return x


if __name__ == '__main__':
    # 实例化CMUNeXtBlock3D
    # 假设输入通道数为64，输出通道数为128，depth=2
    cmunext_block_3d = CMUNeXtBlock3D(ch_in=64, ch_out=128, depth=2, k=3)

    # 创建一个3D输入张量，例如，batch_size=1, 通道数=64, 体积尺寸为64x64x64
    input_tensor_3d = torch.rand(1, 64, 64, 64, 64)

    # 打印输入的shape
    print(f"3D输入的shape: {input_tensor_3d.shape}")

    # 使用定义好的CMUNeXtBlock3D处理3D输入tensor
    output_tensor_3d = cmunext_block_3d(input_tensor_3d)

    # 打印输出的shape
    print(f"3D输出的shape: {output_tensor_3d.shape}")