import torch.nn as nn
import torch.nn.functional as F
#A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation
#https://ieeexplore.ieee.org/document/10458980
"""
SqueezeAndExcitation 类
目的：此类实现了SE块，一种设计用于通过显式模拟通道之间的相互依赖性来自适应地校准通道特征响应的机制。
初始化参数：
channel：输入特征图的通道数量。
reduction：用于控制瓶颈层的降维比率，以控制容量和计算成本。典型值为16。
activation：第一次卷积后使用的激活函数。默认为ReLU。
组成部分：
一个由两个卷积层组成的瓶颈结构：第一层通过降维比率减小维度后接激活函数，第二层恢复维度后接激活函数，接着使用sigmoid激活函数生成逐通道的权重。
前向传播：
执行全局平均池化来生成逐通道的统计数据。
将这些统计数据通过瓶颈（相当于全连接层）来生成一组逐通道的权重。
将这些权重应用于输入特征图，以增强重要特征并抑制不太有用的特征。
SqueezeAndExciteFusionAdd 类
目的：此类定义了一种融合机制，该机制采用两个SE块的输出，并使用按元素的加法将它们结合在一起。这对于整合来自不同来源或特征级别的信息非常有用。
初始化参数：
channels_in：每个输入特征图到 SE 块的通道数。
activation：每个 SE 块内使用的激活函数。
组成部分：
两个由 SqueezeAndExcitation 类定义的SE块。
前向传播：
将每个 SE 块应用于其各自的输入。
将这两个块的输出相加以产生最终输出。
这种设置通过专注于数据中更多的信息特征，增强模型性能的优势。它可以在各种CNN架构中用于改善图像分类、物体检测等任务的性能。
"""

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_1 = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_2 = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, se1, se2):
        se1 = self.se_1(se1)
        se2 = self.se_2(se2)
        out = se1 + se2
        return out


if __name__ == '__main__':
    import torch

    # 假设的输入数据
    input_1 = torch.randn(32, 64, 128, 128)
    input_2 = torch.randn(32, 64, 128, 128)

    # 打印输入数据的形状
    print("Input 1 shape:", input_1.shape)  # 输出: (32, 64, 128, 128)
    print("Input 2 shape:", input_2.shape)  # 输出: (32, 64, 128, 128)

    # 创建SqueezeAndExciteFusionAdd模块的实例
    se_fusion_module = SqueezeAndExciteFusionAdd(channels_in=64)

    # 将输入通过SqueezeAndExciteFusionAdd模块获得输出
    output = se_fusion_module(input_1, input_2)

    # 打印输出数据的形状
    print("Output shape:", output.shape)  # 输出应该和输入形状相同: (32, 64, 128, 128)