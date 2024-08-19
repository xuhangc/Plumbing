import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入 PyTorch 的函数库
# https://github.com/JustlfC03/SCKansformer
# SCKansformer: Fine-Grained Classification of Bone Marrow Cells via Kansformer Backbone and Hierarchical Attention Mechanisms
# https://arxiv.org/pdf/2406.09931

# 自定义 GroupBatchnorm2d 类，实现分组批量归一化
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num = group_num  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.view(N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.view(N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量


# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,  # 输出通道数
                 group_num: int = 16,  # 分组数，默认为16
                 gate_treshold: float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn: bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数

        # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x = self.gn(x)  # 应用分组批量归一化
        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights = self.sigomid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
        x_1 = info_mask * x  # 使用信息门控掩码
        x_2 = noninfo_mask * x  # 使用非信息门控掩码
        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接


# 自定义 CRU（Channel Reduction Unit）类
class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel = low_channel = op_channel - up_channel  # 计算下层通道数
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层

        # 上层特征转换
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)  # 创建卷积层
        self.advavg = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size,
                       group_kernel_size=group_kernel_size)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x


if __name__ == '__main__':
    model = ScConv(op_channel=64)#与c对应
    x= torch.randn(1, 64, 128, 128)#b, c, h, w
    output = model(x)
    print(x.shape)
    print(output.shape)