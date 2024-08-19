import torch
import torch.nn as nn
import math
#SUnet: A multi-organ segmentation network based on multiple attention
#https://www.sciencedirect.com/science/article/abs/pii/S0010482523010612
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Efficient_Attention_Gate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=32):
        super(Efficient_Attention_Gate3D, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm3d(F_int),
            nn.ReLU(inplace=True)
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm3d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(g1 + x1))
        out = x * psi
        out += x

        return out


class EfficientChannelAttention3D(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention3D, self).__init__()

        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return v


class EFF3D(nn.Module):
    def __init__(self, in_dim, out_dim, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.EAG = Efficient_Attention_Gate3D(in_dim, in_dim, out_dim)
        else:
            self.EAG = nn.Identity()
        self.ECA = EfficientChannelAttention3D(in_dim*2)
        self.SA = SpatialAttention3D()

    def forward(self, x, skip):
        if not self.is_bottom:
            EAG_skip = self.EAG(x, skip)
            x = EAG_skip + x
        else:
            x = self.EAG(x)
        x = self.ECA(x) * x
        x = self.SA(x) * x
        return x


if __name__ == '__main__':
    # 实例化 EFF3D 类
    eff_module = EFF3D(in_dim=512, out_dim=512, is_bottom=False)

    batch_size = 1
    depth = 32  # 新增体积深度维度
    height = 64
    width = 64
    in_dim = 512

    x1 = torch.randn(batch_size, in_dim, depth, height, width)
    x2 = torch.randn(batch_size, in_dim, depth, height, width)

    output = eff_module(x1, x2)

    print(f"Input shape1: {x1.shape}")
    print(f"Input shape2: {x2.shape}")
    print(f"Output shape: {output.shape}")