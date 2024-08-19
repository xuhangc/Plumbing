import torch
import torch.nn as nn
#MixDehazeNet : Mix Structure Block For Image Dehazing Network
#https://arxiv.org/abs/2305.17654
#Mix Structure Block contains multi-scale parallel large convolution kernel module and enhanced parallel attention module
"""
标准化层：它使用两个批量归一化层 (self.norm1, self.norm2) 来归一化特征，提高网络的训练稳定性和性能。

用于特征提取的卷积层：
self.conv1：1x1 卷积，用于通道间特征重校正。
self.conv2：5x5 卷积，带有填充和反射填充模式，用于捕获中等范围特征。
self.conv3_19、self.conv3_13、self.conv3_7：具有不同核大小（分别为 7x7、5x5、3x3）、填充和扩张率的分组卷积。这些旨在提取多尺度特征，同时通过分组卷积控制计算成本。

注意力模块：
**简单像素注意力 (self.Wv, self.Wg)**：这关注特征图中的重要空间位置。self.Wv 调整每个特征的重要性，而 self.Wg 使用 Sigmoid 激活提供全局门控机制。
**通道注意力 (self.ca)**：它使用全局平均池化后的两层网络（GELU 和 Sigmoid 激活）专注于有信息的通道。
**像素注意力 (self.pa)**：它在减少的通道维度上操作，以突出重要像素，同样使用 GELU 激活的二层网络。

用于特征整合的 MLP 层：
self.mlp 和 self.mlp2：这些顺序模型各包含两个卷积层和一个中间的 GELU 激活，用于整合提取的特征并细化输出。
在 forward 方法中，逐步将这些组件应用到输入上，包括在第一阶段处理后的初始身份跳过连接，接着是通过卷积和注意力进行特征提取，并进行另一组操作以产生最终输出。
模型采用跳过连接，增强了训练期间梯度的流动，允许网络在添加新信息的同时保留初始特征。
"""
class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


if __name__ == '__main__':
    # 实例化 MixStructureBlock 模块
    # 假设我们的输入通道数为 64
    dim = 64
    mix_structure_block = MixStructureBlock(dim=dim)

    # 创建一个随机的输入张量进行测试
    # 假设输入的尺寸为 (batch_size=1, channels=dim, height=128, width=128)
    input_tensor = torch.rand(1, dim, 128, 128)

    # 将输入张量传递给 MixStructureBlock 模块
    output_tensor = mix_structure_block(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)