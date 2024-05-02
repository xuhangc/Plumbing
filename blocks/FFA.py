import torch.nn as nn
import torch
#FFA(AAAI 2020)：用于单图像去雾的特征融合注意力模块,这个模块要求3通道输入，注意

# 定义默认的卷积操作
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


# 位置注意力层（Position Attention Layer）
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),  # 1x1卷积，降低通道数
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),  # 1x1卷积，输出单个注意力权重
            nn.Sigmoid()  # Sigmoid激活函数，将注意力权重限制在0到1之间
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y  # 将输入特征图与位置注意力权重相乘，以加强有意义的位置信息


# 通道注意力层（Channel Attention Layer）
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，将特征图大小调整为1x1
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),  # 1x1卷积，降低通道数
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),  # 1x1卷积，输出单个通道注意力权重
            nn.Sigmoid()  # Sigmoid激活函数，将注意力权重限制在0到1之间
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 对输入特征图进行平均池化
        y = self.ca(y)  # 计算通道注意力权重
        return x * y  # 将输入特征图与通道注意力权重相乘，以加强有意义的通道信息


# 基础块（Block）
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)  # 第一个卷积层
        self.act1 = nn.ReLU(inplace=True)  # 第一个ReLU激活函数
        self.conv2 = conv(dim, dim, kernel_size, bias=True)  # 第二个卷积层
        self.calayer = CALayer(dim)  # 通道注意力层
        self.palayer = PALayer(dim)  # 位置注意力层

    def forward(self, x):
        res = self.act1(self.conv1(x))  # 第一个卷积操作后的结果
        res = res + x  # 残差连接
        res = self.conv2(res)  # 第二个卷积操作
        res = self.calayer(res)  # 通道注意力操作
        res = self.palayer(res)  # 位置注意力操作
        res = res + x  # 残差连接
        return res


# 分组（Group）
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        # 构建多个基础块，并将它们放入一个序列中
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))  # 添加最后一个卷积层
        self.gp = nn.Sequential(*modules)  # 将所有块放入一个序列中

    def forward(self, x):
        res = self.gp(x)  # 对输入进行分组处理
        res = res + x  # 残差连接
        return res


# 特征融合注意力（Feature Fusion Attention）
class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps  # 分组数
        self.dim = 64  # 特征通道数
        kernel_size = 3  # 卷积核大小
        pre_process = [conv(3, self.dim, kernel_size)]  # 前处理卷积操作

        # 确保分组数为3
        assert self.gps == 3

        # 构建3个分组
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)

        # 通道注意力层
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # 位置注意力层
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]  # 后处理卷积操作

        self.pre = nn.Sequential(*pre_process)  # 前处理序列
        self.post = nn.Sequential(*post_process)  # 后处理序列

    def forward(self, x1):
        x = self.pre(x1)  # 对输入进行前处理
        res1 = self.g1(x)  # 第一个分组处理
        res2 = self.g2(res1)  # 第二个分组处理
        res3 = self.g3(res2)  # 第三个分组处理
        w = self.ca(torch.cat([res1, res2, res3], dim=1))  # 计算通道注意力权重
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]  # 调整权重形状
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3  # 根据权重进行特征融合
        out = self.palayer(out)  # 位置注意力操作
        x = self.post(out)  # 后处理卷积操作
        return x + x1  # 输出与输入特征图相加，以便保留原始信息


if __name__ == "__main__":
    # 创建一个随机输入张量，形状为(batch_size, channels, height, width)
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入大小为64x64，并且batch_size为1

    # 创建FFA网络实例
    net = FFA(gps=3, blocks=20)#这里gps只能是3，我就不演示具体操作了，之前的视频可以参考

    # 将输入张量传递给网络
    output_tensor = net(input_tensor)

    # 打印输入和输出张量的形状
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)