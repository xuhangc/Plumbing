import torch
import torch.nn as nn
import math
import torch.nn.functional as F
"""
LocalGlobalAttention（局部全局注意力）： 这个模块似乎执行一种注意力机制。它计算输入特征图的局部注意力，
然后将其与全局注意力机制结合起来。它利用可学习参数（prompt 和 top_down_transform）调节注意力。

SpatialAttentionModule（空间注意力模块）： 这个模块执行空间注意力，计算跨通道的平均和最大激活，并
将它们连接起来。然后应用一个卷积层，然后是 Sigmoid 激活函数，以获得应用于输入特征图的元素级注意权重。

ECA： 这个模块实现了高效的通道注意力（ECA），增强了通道间的特征交互。
它沿着通道维度应用一个 1D 卷积操作，然后是 Sigmoid 激活，生成注意权重，然后与输入特征图进行元素级乘法。

conv_block（卷积块）： 这是一个通用的卷积块，包括一个卷积层，后跟归一化（GroupNorm 或 BatchNorm）和激活函数（ReLU）。用于网络内的特征提取。

PPA： 这似乎是主要模块，整合了上述所有组件。它包括几个卷积块（c1，c2，c3），跳跃连接，局部全局注意力模块（lga2，lga4），
空间注意力，ECA 和批归一化。它似乎被设计用于特征的细化和整合，带有注意力机制和跳跃连接，以便更好地传递信息。
"""

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

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

class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type='bn',
                                activation=False)
         self.c1 = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c2 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c3 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.bn1 = nn.BatchNorm2d(filters)
         self.drop = nn.Dropout2d(0.1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
if __name__ == '__main__':
    # 创建 PPA 实例
    ppa = PPA(in_features=64, filters=128)  # 假设输入特征维度为 64，输出特征维度为 128

    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 64, 32, 32)  # （batch_size, channels, height, width）

    # 前向传播
    output_tensor = ppa(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)