import torch
import torch.nn as nn
#MSCA：一种多尺度卷积注意力模块(NeurIPS 2022)，即插即用

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 定义多个卷积层
        # 这些卷积层用于提取不同范围的特征以构建注意力机制
        # 具体来说，使用了不同大小的卷积核来捕获不同范围的上下文信息
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # 最后使用一个卷积层来融合特征并输出注意力权重
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # 复制输入特征，以便后续计算
        u = x.clone()
        # 对输入特征进行一系列卷积操作
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        # 将不同范围的特征加权求和
        attn = attn + attn_0 + attn_1 + attn_2
        # 再次通过一个卷积层融合特征并输出注意力权重
        attn = self.conv3(attn)
        # 将输出的注意力权重与输入特征进行元素级别的相乘
        # 以加强或减弱输入特征的某些部分
        return attn * u

if __name__ == "__main__":
    # 创建一个AttentionModule实例
    attention_module = AttentionModule(64)
    # 生成一个随机输入张量
    input_tensor = torch.randn(8, 64, 32, 32)
    # 将输入张量输入到AttentionModule中进行前向传播
    output_tensor = attention_module(input_tensor)
    # 打印输入和输出张量的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
