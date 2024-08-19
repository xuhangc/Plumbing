import torch
import torch.nn as nn
import torch.nn.functional as F
"""
RSANet: Recurrent Slice-wise Attention Network for Multiple Sclerosis Lesion
https://arxiv.org/pdf/2002.12470
每一个空间注意力块（saAxialBlock、saCoronalBlock、saSagittalBlock）都是通过首先交换和重塑输入张量来强调其专注的特定空间维度，
计算注意力权重，然后应用这些权重来指导特征的混合。在这些块中的操作流程非常相似：交换，重塑，计算注意力，应用注意力，然后重塑并交换回原始维度。
rsaBlock通过首先应用三种不同的卷积处理输入（theta_x、phi_x、g_x）为每个注意力机制做准备。
然后，它迭代通过每一个空间注意力块，应用它，并且将结果加回输入，并使用一个缩放因子（beta）。这些beta是学习到的参数，可以让网络动态地调整每个空间注意力分量的重要性。
"""

class saAxialBlock(nn.Module):

    def __init__(self):

        super(saAxialBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0] # original size: n * c * d * h * w
        depth_size = x_.size()[2]

        x_ = x_.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        x_ = x_.contiguous().view(batch_size, depth_size, -1)
        # n * d * chw

        g_x = g_x.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, depth_size, -1)
        # n * d * chw

        x_t = x_t.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        x_t = x_t.contiguous().view(batch_size, depth_size, -1)
        # n * d * chw
        x_t = x_t.permute(0, 2, 1)
        # n * chw * d

        attention = torch.matmul(x_, x_t)
        # n * d * d
        attention = F.softmax(attention, dim = -1)
        # n * d * d
        del x_, x_t

        # n * d * d X n * d * chw
        out = torch.matmul(attention, g_x)
        # n * d * chw
        del attention, g_x
        out = out.view(batch_size, depth_size, *ori_size[2:])
        # n * d * c * h * w
        out = out.permute(0, 2, 1, 3, 4)
        # n * c * d * h * w

        return out


class saCoronalBlock(nn.Module):

    def __init__(self):

        super(saCoronalBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0]
        coronal_size = x_.size()[3]

        # n * c * d * h * w
        x_ = x_.permute(0, 3, 2, 1, 4)
        x_ = x_.contiguous().view(batch_size, coronal_size, -1)

        g_x = g_x.permute(0, 3, 2, 1, 4)
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, coronal_size, -1)

        x_t = x_t.permute(0, 3, 2, 1, 4)
        x_t = x_t.contiguous().view(batch_size, coronal_size, -1)
        x_t = x_t.permute(0, 2, 1)

        attention = torch.matmul(x_, x_t)
        attention = F.softmax(attention, dim = -1)
        del x_, x_t

        out = torch.matmul(attention, g_x)
        del attention, g_x
        out = out.view(batch_size, coronal_size, *ori_size[2:])
        out = out.permute(0, 3, 2, 1, 4)

        return out

class saSagittalBlock(nn.Module):

    def __init__(self):

        super(saSagittalBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0]
        sagittal_size = x_.size()[4]

        # n * c * d * h * w
        x_ = x_.permute(0, 4, 2, 3, 1)
        x_ = x_.contiguous().view(batch_size, sagittal_size, -1)

        g_x = g_x.permute(0, 4, 2, 3, 1)
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, sagittal_size, -1)

        x_t = x_t.permute(0, 4, 2, 3, 1)
        x_t = x_t.contiguous().view(batch_size, sagittal_size, -1)
        x_t = x_t.permute(0, 2, 1)

        attention = torch.matmul(x_, x_t)
        attention = F.softmax(attention, dim = -1)
        del x_, x_t

        out = torch.matmul(attention, g_x)
        del attention, g_x
        out = out.view(batch_size, sagittal_size, *ori_size[2:])
        out = out.permute(0, 4, 2, 3, 1)

        return out

class rsaBlock(nn.Module):

    def __init__(self, in_channels):

        super(rsaBlock, self).__init__()

        self.theta_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        coronal_sa  = saCoronalBlock()
        sagittal_sa = saSagittalBlock()
        axial_sa    = saAxialBlock()

        coronal_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        sagittal_beta = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        axial_beta    = nn.Parameter(torch.tensor(0.0, requires_grad = True))

        self.saBlocks = nn.ModuleList([coronal_sa, sagittal_sa, axial_sa])
        self.betas = nn.ParameterList([coronal_beta, sagittal_beta, axial_beta])

    def forward(self, x):

        for idx, saBlock in enumerate(self.saBlocks):

            theta_x = self.theta_conv(x)
            phi_x = self.phi_conv(x)
            g_x = self.g_conv(x)

            x = x + saBlock(theta_x, phi_x, g_x) * self.betas[idx]
            del theta_x, phi_x, g_x

        return x

if __name__ == '__main__':
    # 实例化 rsaBlock，这里假设输入的特征通道数为in_channels
    in_channels = 64  # 这个值可以是任意正数，这里只是一个例子
    rsa_block = rsaBlock(in_channels=in_channels)

    # 创建输入张量，假设张量形状为 [batch_size, channels, depth, height, width]
    # 这里的形状值也是示例，可以根据实际情况进行调整
    batch_size, depth, height, width = 2, 32, 64, 64  # 这些值也是示例，可以自定义
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # 通过 rsaBlock 传递输入张量，获得输出
    output = rsa_block(x)

    # 打印输入和输出的形状
    print('输入形状:', x.shape)
    print('输出形状:', output.shape)