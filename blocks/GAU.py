import torch
import torch.nn as nn
"""
这段代码定义了两个PyTorch模块：TA（时间注意力）和SCA（空间通道注意力）。

时间注意力（TA）：
TA 旨在跨时间维度（沿着时间轴）对3D数据执行注意力。
它使用自适应平均池化和自适应最大池化来独立地汇集时间特征。
然后，对每个池化后的特征应用共享的MLP（多层感知器），然后是ReLU激活。
这些MLP的输出被逐元素求和在一起。
最后，应用Sigmoid激活函数以产生注意力权重。

空间通道注意力（SCA）：
SCA 在5D输入张量上操作，通常表示时空数据（批次、时间步长、通道数、高度、宽度）。
首先，它沿着批次和时间维度展平输入张量。
然后，它将共享的MLP应用于展平的张量，然后是ReLU激活。
最后，将输出重新塑形回原始形状。
"""
class TA(nn.Module):
    def __init__(self,  T,ratio=2):

        super(TA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(T, T // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(T // ratio, T, 1,bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        # B,T,C
        out1 = self.sharedMLP(avg)
        max = self.max_pool(x)
        # B,T,C
        out2 = self.sharedMLP(max)
        out = out1+out2

        return out

# task classifictaion or generation
class SCA(nn.Module):
    def __init__(self, in_planes, kerenel_size,ratio = 1):
        super(SCA, self).__init__()
        self.sharedMLP = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // ratio, kerenel_size, padding='same', bias=False),
                nn.ReLU(),
                nn.Conv2d(in_planes // ratio, in_planes, kerenel_size, padding='same', bias=False),)
    def forward(self, x):
        b,t, c, h, w = x.shape
        x = x.flatten(0,1)
        x = self.sharedMLP(x)
        out = x.reshape(b,t, c, h, w)
        return out
if __name__ == '__main__':
    # 创建TA模型
    ta_model = TA(T=10)  # 假设输入有10个时间步长
    print("TA模型结构：\n", ta_model)

    # 创建SCA模型
    sca_model = SCA(in_planes=64, kerenel_size=3)  # 假设输入通道数为64
    print("\nSCA模型结构：\n", sca_model)

    # 创建随机输入数据
    batch_size = 4
    time_steps = 10
    channels = 64
    height = 32
    width = 32
    input_data = torch.randn(batch_size, time_steps, channels, height, width)
    print("\n输入数据形状：", input_data.shape)

    # 测试TA模型
    output_ta = ta_model(input_data)
    print("TA模型输出形状：", output_ta.shape)

    # 测试SCA模型
    output_sca = sca_model(input_data)
    print("SCA模型输出形状：", output_sca.shape)