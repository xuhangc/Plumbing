import torch
import torch.nn as nn

"""
深度卷积神经网络 (CNN) 的最新进展促使研究人员采用 CNN 来直接对 3D 点云中的点进行建模。
局部结构建模已被证明对于卷积架构的成功非常重要，研究人员利用了特征提取层次结构中局部点集的建模。
然而，对局部区域中点之间的几何结构的显式建模的关注有限。为了解决这个问题，我们提出了 Geo-CNN，它将一种称为 GeoConv 的通用类卷积运算应用于每个点及其局部邻域。
在提取中心与其相邻点之间的边缘特征时，捕获点之间的局部几何关系。我们首先将边缘特征提取过程分解到三个正交基上，然后根据边缘向量与基之间的角度聚合提取的特征。
这鼓励网络在整个特征提取层次结构中保留欧几里得空间中的几何结构。GeoConv 是一种通用且高效的操作，可以轻松集成到多个应用程序的 3D 点云分析管道中。
我们在 ModelNet40 和 KITTI 上评估 Geo-CNN 并实现了最先进的性能。
"""


def Norm(name, c, channels_per_group=16, momentum=0.1, md=1):
    if name == 'bn':
        return eval(f'nn.BatchNorm{md}d')(c, momentum=momentum)
    elif name == 'gn':
        num_group = c // channels_per_group
        if num_group * channels_per_group != c:
            num_group = 1
        return nn.GroupNorm(num_group, c)

class GeoConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, norm='bn'):
        super(GeoConv, self).__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.lins = nn.ModuleList([nn.Linear(in_channels, hidden_channels) for _ in range(6)])
        self.norm1 = Norm(norm, hidden_channels, md=1)
        self.acti1 = nn.ReLU(inplace=True)
        self.norm2 = Norm(norm, out_channels, md=1)
        self.acti2 = nn.ReLU(inplace=True)

    def forward(self, x, p, B, n, id_euc):
        # x[B*N, C] p[B*N, 3] sid/tid[B*n*k]
        sid_euc, tid_euc = id_euc
        k = int(len(sid_euc) / B / n)
        dev = x.device

        euc_i, euc_j = x[tid_euc], x[sid_euc]  # [B*n*k, C]
        edge = euc_j - euc_i

        p_diff = p[sid_euc] - p[tid_euc]  # [B*n*k, 3]
        p_dis = p_diff.norm(dim=-1, keepdim=True).clamp(min=1e-16)  # [B*n*k, 1]
        p_cos = (p_diff / p_dis).cos() ** 2  # [B*n*k, 3]
        p_cos = p_cos.transpose(0, 1).reshape(-1, B, n, k, 1)  # [3, B, n, k, 1]
        bid = (p_diff > 0).long()  # [B*n*k, 3]
        bid += torch.tensor([0, 2, 4], device=dev, dtype=torch.long).view(1, 3)
        edge = torch.stack([lin(edge) for lin in self.lins])  # [bases, B*n*k, C]
        edge = torch.stack([edge[bid[:, i], range(B * n * k)] for i in range(3)])  # [3, B*n*k, C]
        edge = edge.view(3, B, n, k, -1)
        edge = edge * p_cos  # [3, B, n, k, C]
        edge = edge.sum(dim=0)  # [B, n, k, C]
        p_dis = p_dis.view(B, n, k, 1)
        p_r = p_dis.max(dim=2, keepdim=True)[0] * 1.1  # [B, n, 1, 1]
        p_d = (p_r - p_dis) ** 2  # [B, n, k, 1]
        edge = edge * p_d / p_d.sum(dim=2, keepdim=True)  # [B, n, k, C]
        y = edge.sum(dim=2).transpose(1, -1)  # [B, C, n]
        y = self.acti1(self.norm1(y)).transpose(1, -1)  # [B, n, C]
        x = self.lin1(x[tid_euc[::k]]).view(B, n, -1)  # [B, n, C]
        y = x + self.lin2(y)  # [B, n, C]
        y = y.transpose(1, -1)  # [B, C, n]
        y = self.acti2(self.norm2(y))
        y = y.transpose(1, -1)  # [B, n, C]
        y = y.flatten(0, 1)  # [B*n, C]

        return y


if __name__ == '__main__':
    in_channels = 3             # 输入特征的通道数
    hidden_channels = 64        # 隐藏层的通道数
    out_channels = 3            # 输出特征的通道数
    kernel_size = 3             # 卷积核大小
    B = 2  # 假设批量大小
    n = 1000  # 假设每个批量中的点的数量
    id_euc = (torch.randint(0, n, (B*n*kernel_size,)), torch.randint(0, n, (B*n*kernel_size,)))  # 模拟的欧几里得距离的索引数组

    block = GeoConv(in_channels, hidden_channels, out_channels, norm='bn')
    x = torch.rand(B*n, in_channels)  # 模拟输入特征
    p = torch.rand(B*n, 3)  # 模拟点云的坐标
    output = block(x, p, B, n, id_euc)
    print('Input size:', x.size())
    print('Output size:', output.size())