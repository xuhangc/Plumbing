import torch.nn as nn
import torch.nn.functional as F
import torch
#Towards Cross-Scale Attention and Surface Supervision for Fractured Bone Segmentation in CT
#https://arxiv.org/pdf/2405.01204
class CSA2D(nn.Module):
    def __init__(self, channel_l, channel_g, init_channel=64):
        super(CSA2D, self).__init__()
        self.W_x1 = nn.Conv2d(channel_l, channel_l, kernel_size=1)
        self.W_x2 = nn.Conv2d(channel_l, channel_g, kernel_size=3, stride=2, padding=1)
        self.W_g1 = nn.Conv2d(init_channel, channel_l, kernel_size=3, stride=1, padding=1)
        self.W_g2 = nn.Conv2d(channel_g, channel_g, kernel_size=1)
        self.relu = nn.ReLU()
        self.psi1 = nn.Conv2d(channel_l, out_channels=1, kernel_size=1)
        self.psi2 = nn.Conv2d(channel_g, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x_l, x_g, first_layer_f):
        # 第一次注意力操作
        first_layer_afterconv = self.W_g1(first_layer_f)
        xl_afterconv = self.W_x1(x_l)

        if first_layer_afterconv.size() != xl_afterconv.size():
            raise ValueError(f"形状不匹配: {first_layer_afterconv.size()} vs {xl_afterconv.size()}")

        att_map_first = self.sig(self.psi1(self.relu(first_layer_afterconv + xl_afterconv)))
        xl_after_first_att = x_l * att_map_first

        # 第二次注意力操作
        xg_afterconv = self.W_g2(x_g)
        xl_after_first_att_and_conv = self.W_x2(xl_after_first_att)

        if xg_afterconv.size() != xl_after_first_att_and_conv.size():
            raise ValueError(f"形状不匹配: {xg_afterconv.size()} vs {xl_after_first_att_and_conv.size()}")

        att_map_second = self.sig(self.psi2(self.relu(xg_afterconv + xl_after_first_att_and_conv)))
        att_map_second_upsample = F.interpolate(att_map_second, size=x_l.size()[2:], mode='bilinear',
                                                align_corners=True)
        out = xl_after_first_att * att_map_second_upsample
        return out


if __name__ == '__main__':
    # 实例化 CSA2D
    csa_module = CSA2D(channel_l=128, channel_g=256, init_channel=64)

    # 模拟输入，这里仅作为示例，使用具体的数值
    batch_size = 1
    height, width = 128, 128  # x_l 的空间尺寸
    reduced_height, reduced_width = 64, 64  # x_g 的空间尺寸，假设为 x_l 的一半

    x_l = torch.rand(batch_size, 128, height, width)  # 局部特征图输入
    x_g = torch.rand(batch_size, 256, reduced_height, reduced_width)  # 全局特征图输入
    first_layer_f = torch.rand(batch_size, 64, height, width)  # 第一层输入特征图

    # 前向传播
    output = csa_module(x_l, x_g, first_layer_f)

    # 打印输入输出形状
    print("x_l 的输入形状:", x_l.shape)
    print("x_g 的输入形状:", x_g.shape)
    print("first_layer_f 的输入形状:", first_layer_f.shape)
    print("输出形状:", output.shape)
