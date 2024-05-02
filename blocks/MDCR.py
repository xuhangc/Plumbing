import torch
import torch.nn as nn
"""
https://arxiv.org/pdf/2403.10778v1.pdf
HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection
这段代码定义了一个多膨胀通道卷积模块（MDCR），适用于 PyTorch。
该模块通过使用不同的扩张率（dilation rate）的扩张卷积，在多个尺度上捕获特征，
特别适用于语义分割或物体检测等需要在多个尺度上捕获特征的任务。

conv_block：这是一个基本的卷积块，带有可选的归一化层（默认为 BatchNorm 或 GroupNorm）和激活函数（默认为 ReLU）。

MDCR（多孔卷积残差）：该模块包括四个卷积块，每个块具有不同的扩张率。这些块在输入张量的不同部分上操作，
允许模型在多个尺度上捕获特征。这些块的输出沿着通道维度进行串联。

MDCR 的 forward 方法：输入张量沿着通道维度分成四个部分。每个部分通过具有不同扩张率的相应卷积块。
然后，将这些输出沿着通道维度进行串联。最后，串联的张量通过另一个卷积块进行进一步处理。
"""
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

class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block2 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block3 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block4 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x


if __name__ == '__main__':
    # 创建一个随机输入张量作为示例
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入尺寸为 [batch_size, channels, height, width]

    # 实例化 MDCR 模块
    mdcr = MDCR(in_features=64, out_features=64)

    # 将输入张量传递给 MDCR 模块并获取输出
    output_tensor = mdcr(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
