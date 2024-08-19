import torch
import torch.nn as nn
import torch.nn.functional as F
#SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution( ECCV 2024 )
#https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ren_The_Ninth_NTIRE_2024_Efficient_Super-Resolution_Challenge_Report_CVPRW_2024_paper.pdf
"""
DMlp 类
DMlp（动态MLP）类实现了一个简单的多层感知机，使用卷积层：

初始化 (__init__ 方法)：

dim: 输入特征维度。
growth_rate: 增加隐藏层维度的因子。
hidden_dim: 计算为 dim * growth_rate。
conv_0: 一个顺序层，包含：
深度卷积（groups=dim）。
点卷积。
act: 激活函数（GELU）。
conv_1: 点卷积，将维度投射回 dim。
前向传播 (forward 方法)：

应用 conv_0，然后激活，再应用 conv_1。
PCFN 类
PCFN（部分卷积的前馈网络）类：

初始化：

dim: 输入特征维度。
growth_rate: 增加隐藏层维度的因子。
p_rate: 部分卷积的隐藏维度比例。
hidden_dim: 计算为 dim * growth_rate。
p_dim: 计算为 hidden_dim * p_rate。
conv_0: 点卷积。
conv_1: 对隐藏维度的一部分应用卷积。
act: 激活函数（GELU）。
conv_2: 点卷积，将维度投射回 dim。
前向传播：

对训练和评估模式有不同的行为。
在训练期间，将 conv_0 的输出分成两部分，对第一部分应用 conv_1，然后连接，再应用 conv_2。
在评估期间，直接对特征图的一部分应用 conv_1。
SMFA 类
SMFA（自调制特征聚合）模块：

初始化：

dim: 输入特征维度。
多个卷积层用于线性变换。
lde: 一个 DMlp 类的实例。
dw_conv: 深度卷积。
gelu: 激活函数。
down_scale: 下采样的缩放因子。
alpha 和 belt: 可学习的调制参数。
前向传播：

使用 linear_0 将输入分成两部分。
对第二部分（x）的下采样版本应用深度卷积。
计算特征图的方差，并与下采样特征图结合。
将调制后的特征图与 DMlp 的输出结合。
FMB 类
FMB（特征调制块）：

初始化：

dim: 输入特征维度。
ffn_scale: 前馈网络的缩放因子。
SMFA 和 PCFN 的实例。
前向传播：

对归一化的输入应用 SMFA，将结果加到输入上。
对归一化的输入应用 PCFN，将结果加到输入上。
"""
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

# partial convolution-based feed-forward network
class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

#self-modulation feature aggregation (SMFA) module
class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

#Feature modulation block(FMB)
class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.smfa = SMFA(dim)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x


if __name__ == '__main__':
    # 定义输入张量的形状 (batch_size, channels, height, width)
    input_shape = (1, 36, 64, 64)

    # 创建一个随机张量作为输入
    input_tensor = torch.randn(input_shape)

    # 实例化FMB类
    fmb = FMB(dim=36)

    # 将输入张量传入FMB实例
    output_tensor = fmb(input_tensor)

    # 打印输入和输出的形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")