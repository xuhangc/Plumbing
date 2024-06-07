import torch
import torch.nn as nn
import torch.nn.functional as F
#Robust change detection for remote sensing images based on temporospatial interactive attention module
#https://www.sciencedirect.com/science/article/pii/S1569843224001213
"""
这几个注意力（SpatiotemporalAttentionFull, SpatiotemporalAttentionBase, SpatiotemporalAttentionFullNotWeightShared）模块主要有以下区别：

SpatiotemporalAttentionFull
在空间（space）和时间（time）上都应用了注意力机制。
对于特定的输入信号在时间和空间上的相关性进行建模。
使用了Softmax来标准化时间和空间能量向量（energy vectors）。
实现了交错的时间和空间注意力聚焦，通过先计算时间注意力跨越时间属性的相似性，然后在这个基础上计算空间注意力跨越空间结构的相似性。

SpatiotemporalAttentionBase
主要聚焦于空间注意力，适用于情况下只需要区分特定空间位置信息的重要性。
不涉及时间维度，仅对输入信号的空间相关性进行建模。
使用Softmax进行空间能量向量的标准化。
相比于SpatiotemporalAttentionFull，这个模块简化了计算，因为它没有包括时间关联的计算。

SpatiotemporalAttentionFullNotWeightShared
这个版本在实现空间和时间注意力时，不共享权重（在SpatiotemporalAttentionFull中，g和W序列被共享使用）。
为了处理两个输入其对应的特征运算，g1和g2以及W1和W2是独立的，这意味着网络能够学习到更加特定化的特征表示，可能使得网络针对不同输入信号学习到更加独特且有效的特征转换。
通过不共享权重，网络增加了参数量，这可能有助于提升模型的表达能力，但同时也增加了模型的复杂性以及对数据的要求。
总结来说，SpatiotemporalAttentionFull考虑了时间和空间的注意力机制，SpatiotemporalAttentionBase只关注空间注意力，
而SpatiotemporalAttentionFullNotWeightShared版本与SpatiotemporalAttentionFull相似，
但它在空间和时间注意力的实现上不共享权重，提供了灵活性以及可能的性能提升，但代价是增加了模型的参数量和复杂性。
"""
class SpatiotemporalAttentionFull(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionFull, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.energy_time_1_sf = nn.Softmax(dim=-1)
        self.energy_time_2_sf = nn.Softmax(dim=-1)
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)

    def forward(self, x1, x2):

        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = self.energy_time_1_sf(energy_time_1)
        energy_time_2s = self.energy_time_2_sf(energy_time_2)
        energy_space_2s = self.energy_space_2s_sf(energy_space_1)
        energy_space_1s = self.energy_space_1s_sf(energy_space_2)

        # energy_time_2s*g_x11*energy_space_2s = C2*S(C1) × C1*H1W1 × S(H1W1)*H2W2 = (C2*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()  # C2*H2W2
        # energy_time_1s*g_x12*energy_space_1s = C1*S(C2) × C2*H2W2 × S(H2W2)*H1W1 = (C1*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)


class SpatiotemporalAttentionBase(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionBase, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)

    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)

        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)
        energy_space_2s = self.energy_space_2s_sf(energy_space_1)  # S(H1W1)*H2W2
        energy_space_1s = self.energy_space_1s_sf(energy_space_2)  # S(H2W2)*H1W1

        # g_x11*energy_space_2s = C1*H1W1 × S(H1W1)*H2W2 = (C1*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(g_x11, energy_space_2s).contiguous()  # C2*H2W2
        # g_x21*energy_space_1s = C2*H2W2 × S(H2W2)*H1W1 = (C2*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(g_x21, energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)


class SpatiotemporalAttentionFullNotWeightShared(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionFullNotWeightShared, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )
        self.g2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.W2 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g1(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g2(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = F.softmax(energy_time_1, dim=-1)
        energy_time_2s = F.softmax(energy_time_2, dim=-1)
        energy_space_2s = F.softmax(energy_space_1, dim=-2)
        energy_space_1s = F.softmax(energy_space_2, dim=-2)
        #  C1*S(C2) energy_time_1s * C1*H1W1 g_x12 * energy_space_1s S(H2W2)*H1W1 -> C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()  # C2*H2W2
        #  C2*S(C1) energy_time_2s * C2*H2W2 g_x21 * energy_space_2s S(H1W1)*H2W2 -> C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()  # C1*H1W1
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W1(y1), x2 + self.W2(y2)


if __name__ == '__main__':
    # 输入特征图的假设参数
    batch_size, channels, height, width = 1, 64, 32, 32

    # 输入特征图
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)

    # 实例化SpatiotemporalAttentionFull
    sp_full = SpatiotemporalAttentionFull(in_channels=channels)
    output_full_x1, output_full_x2 = sp_full(x1, x2)

    print("SpatiotemporalAttentionFull 输入:", x1.shape, x2.shape)
    print("SpatiotemporalAttentionFull 输出:", output_full_x1.shape, output_full_x2.shape)

    # 实例化SpatiotemporalAttentionBase
    sp_base = SpatiotemporalAttentionBase(in_channels=channels)
    output_base_x1, output_base_x2 = sp_base(x1, x2)

    print("\nSpatiotemporalAttentionBase 输入:", x1.shape, x2.shape)
    print("SpatiotemporalAttentionBase 输出:", output_base_x1.shape, output_base_x2.shape)

    # 实例化SpatiotemporalAttentionFullNotWeightShared
    sp_full_not_shared = SpatiotemporalAttentionFullNotWeightShared(in_channels=channels)
    output_full_not_shared_x1, output_full_not_shared_x2 = sp_full_not_shared(x1, x2)

    print("\nSpatiotemporalAttentionFullNotWeightShared 输入:", x1.shape, x2.shape)
    print("SpatiotemporalAttentionFullNotWeightShared 输出:", output_full_not_shared_x1.shape,
          output_full_not_shared_x2.shape)