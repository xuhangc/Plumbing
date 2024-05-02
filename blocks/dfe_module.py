import torch
import torch.nn as nn
import torch.nn.functional as F

"""深度感知特征增强（DFE）模块
dfe_module 模块是 Depth-aware Feature Enhancement (DFE) 模块中的一个子模块，用于增强深度感知的特征。让我来逐步解释其组成部分和功能：

1. 初始化函数 (__init__):
in_channels: 输入特征的通道数。
out_channels: 输出特征的通道数。
2. 初始化层:
Softmax 激活函数: 用于计算特征之间的注意力权重。
卷积层1 (conv1):
使用1x1卷积核将输入特征通道数从 in_channels 调整为 out_channels。
紧接着是批标准化、ReLU激活函数和Dropout层。
卷积层2 (conv2):
用于将增强的特征进一步处理。
3. 前向传播函数 (forward):
接受两个输入参数: feat_ffm 和 coarse_x。
feat_ffm: 来自主干网络的特征，用于生成注意力分布。
coarse_x: 来自深度估计网络的粗糙特征，用作注意力的查询（query）。
将 feat_ffm 通过 conv1 进行处理，得到特征映射。
根据 coarse_x 和处理后的 feat_ffm 计算注意力权重。
使用注意力权重对 coarse_x 进行加权，以得到增强的特征。
最后，通过 conv2 对增强的特征进行进一步处理，并返回输出。
这个模块的主要功能是通过注意力机制，利用深度信息来增强特征表示。它将主干网络提取的特征与深度估计网络生成的粗糙特征结合起来，通过学习的注意力权重，将更多关注放在与深度信息相关的区域上，从而提高模型对深度感知的表征能力。
"""

class DepthAwareFE(nn.Module):
    def __init__(self, output_channel_num):
        super(DepthAwareFE, self).__init__()
        self.output_channel_num = output_channel_num
        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num / 2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num / 2)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num / 2), 96, 1),
        )
        self.depth_down = nn.Conv2d(96, 12, 3, stride=1, padding=1, groups=12)
        self.acf = dfe_module(256, 256)

    def forward(self, x):
        depth = self.depth_output(x)
        N, C, H, W = x.shape
        depth_guide = F.interpolate(depth, size=x.size()[2:], mode='bilinear', align_corners=False)
        depth_guide = self.depth_down(depth_guide)
        x = x + self.acf(x, depth_guide)

        return depth, depth_guide, x


class dfe_module(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(dfe_module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, feat_ffm, coarse_x):
        N, D, H, W = coarse_x.size()

        # depth prototype
        feat_ffm = self.conv1(feat_ffm)
        _, C, _, _ = feat_ffm.size()

        proj_query = coarse_x.view(N, D, -1)
        proj_key = feat_ffm.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # depth enhancement
        attention = attention.permute(0, 2, 1)
        proj_value = coarse_x.view(N, D, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)
        out = self.conv2(out)

        return out

if __name__ == '__main__':

    # 假定输入特征图的尺寸为 [N, C, H, W] = [1, 256, 64, 64]
    # 假定粗糙深度图的尺寸为 [N, D, H, W] = [1, 12, 64, 64]

    N, C, H, W = 1, 256, 64, 64
    D = 12

    # 初始化输入特征图和粗糙深度图
    feat_ffm = torch.rand(N, C, H, W)  # 输入特征图
    coarse_x = torch.rand(N, D, H, W)  # 粗糙深度图

    # 初始化dfe_module
    dfe = dfe_module(in_channels=C, out_channels=C)  # 使用相同的通道数作为示例

    # 前向传播
    output = dfe(feat_ffm, coarse_x)

    # 打印输入和输出尺寸
    print("Input feat_ffm size:", feat_ffm.size())
    print("        Output size:", output.size())