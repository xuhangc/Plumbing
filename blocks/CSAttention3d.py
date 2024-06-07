import torch.nn as nn
import torch.nn.functional as F
import torch
#Towards Cross-Scale Attention and Surface Supervision for Fractured Bone Segmentation in CT
#https://arxiv.org/pdf/2405.01204
"""
初始化（__init__ 方法）：

W_x1：一个 3D 卷积层，用于处理局部特征图 x_l。
W_x2：一个 3D 卷积层，用于下采样并处理局部特征图 x_l，使其与全局特征图 x_g 的尺寸匹配。
W_g1：一个 3D 卷积层，用于处理第一层的特征图 first_layer_f。
W_g2：一个 3D 卷积层，用于处理全局特征图 x_g。
relu：ReLU 激活函数。
psi1 和 psi2：生成注意力图的 3D 卷积层。
sig：Sigmoid 激活函数，用于对注意力图进行归一化。
前向传播方法（forward 方法）：

第一次注意力操作：

first_layer_afterconv 使用 W_g1 处理 first_layer_f。
xl_afterconv 使用 W_x1 处理 x_l。
通过结合 first_layer_afterconv 和 xl_afterconv，并经过 ReLU、psi1 和 Sigmoid 生成注意力图 (att_map_first)。
该注意力图用于重新加权局部特征图 x_l，得到 xl_after_first_att。

第二次注意力操作：

xg_afterconv 使用 W_g2 处理 x_g。
xl_after_first_att_and_conv 使用 W_x2 处理 xl_after_first_att，将其下采样至 x_g 的尺寸。
通过结合 xg_afterconv 和 xl_after_first_att_and_conv，并经过 ReLU、psi2 和 Sigmoid 生成注意力图 (att_map_second)。
使用 F.interpolate 将该注意力图上采样至 x_l 的尺寸。
该上采样后的注意力图重新加权 xl_after_first_att，得到最终输出 out。
"""
class CSA(nn.Module):
    def __init__(self, channel_l, channel_g, init_channel=64):
        super(CSA, self).__init__()
        self.W_x1 = nn.Conv3d(channel_l, channel_l, kernel_size=1)
        self.W_x2 = nn.Conv3d(channel_l, channel_g, kernel_size=3, stride=2, padding=1)
        self.W_g1 = nn.Conv3d(init_channel, channel_l, kernel_size=3, stride=1, padding=1)
        self.W_g2 = nn.Conv3d(channel_g, channel_g, kernel_size=1)
        self.relu = nn.ReLU()
        self.psi1 = nn.Conv3d(channel_l, out_channels=1, kernel_size=1)
        self.psi2 = nn.Conv3d(channel_g, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x_l, x_g, first_layer_f):
        # First Attention Operation
        first_layer_afterconv = self.W_g1(first_layer_f)
        xl_afterconv = self.W_x1(x_l)

        if first_layer_afterconv.size() != xl_afterconv.size():
            raise ValueError(f"Shape mismatch: {first_layer_afterconv.size()} vs {xl_afterconv.size()}")

        att_map_first = self.sig(self.psi1(self.relu(first_layer_afterconv + xl_afterconv)))
        xl_after_first_att = x_l * att_map_first

        # Second Attention Operation
        xg_afterconv = self.W_g2(x_g)
        xl_after_first_att_and_conv = self.W_x2(xl_after_first_att)

        if xg_afterconv.size() != xl_after_first_att_and_conv.size():
            raise ValueError(f"Shape mismatch: {xg_afterconv.size()} vs {xl_after_first_att_and_conv.size()}")

        att_map_second = self.sig(self.psi2(self.relu(xg_afterconv + xl_after_first_att_and_conv)))
        att_map_second_upsample = F.interpolate(att_map_second, size=x_l.size()[2:], mode='trilinear')
        out = xl_after_first_att * att_map_second_upsample
        return out


if __name__ == '__main__':
    # 实例化 CSA
    csa_module = CSA(channel_l=128, channel_g=256, init_channel=64)

    # 模拟输入，这里仅作为示例，使用具体的数值
    batch_size = 1
    depth, height, width = 64, 128, 128  # x_l 的空间尺寸
    reduced_depth, reduced_height, reduced_width = 32, 64, 64  # x_g 的空间尺寸，假设为 x_l 的一半

    x_l = torch.rand(batch_size, 128, depth, height, width)  # 局部特征图输入，注意通道数与 csa_module 的 channel_l 匹配
    x_g = torch.rand(batch_size, 256, reduced_depth, reduced_height, reduced_width)  # 全局特征图输入，注意通道数与 csa_module 的 channel_g 匹配
    first_layer_f = torch.rand(batch_size, 64, depth, height, width)  # 第一层的输入特征图，注意通道数与 csa_module 的 init_channel 匹配

    # 前向传播
    output = csa_module(x_l, x_g, first_layer_f)

    # 打印输入输出形状
    print("Input shape of x_l:", x_l.shape)
    print("Input shape of x_g:", x_g.shape)
    print("Input shape of first_layer_f:", first_layer_f.shape)
    print("Output shape:", output.shape)