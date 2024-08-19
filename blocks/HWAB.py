import torch
import torch.nn as nn
#HALF WAVELET ATTENTION ON M-NET+ FOR LOW-LIGHT IMAGE ENHANCEMENT
#https://arxiv.org/abs/2203.01296
"""
卷积层（conv）：此函数返回一个2D卷积层，为图片的基本和复杂特征提取奠定了基础。使用填充确保根据内核大小管理输出尺寸，使网络架构灵活。

离散小波变换（DWT）和逆离散小波变换（IWT）：这些函数（dwt_init, iwt_init）及其相应的类包装器（DWT, IWT）代表了基于小波的方法的核心。
它们通过将图像分解为高频和低频部分，然后分别处理这些部分。这种分离使模型能够更好地捕获和增强低光条件下的细节。

空间注意力层（SALayer）：该组件聚焦于图像的空间特征。通过对处理过的通道信息应用sigmoid函数，它衡量了图像不同区域内的重要性，帮助网络确定其增强努力的焦点。

通道注意力层（CALayer）：针对通道方面的特征，此层通过降低机制和sigmoid激活过程中的特征缩放，进一步精细化图像细节，帮助强调更重要的通道。

半小波注意力块（HWAB）：HWAB是模型将小波组件与注意机制整合在一起的地方。
它处理从DWT获得的低频和高频成分，应用空间和通道注意力，然后使用IWT重建增强后的图像。此块在管理不同规模提取的特征时至关重要，确保保留和增强全局和局部细节。
"""
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width])

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Half Wavelet Attention Block (HWAB)
class HWAB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
        super(HWAB, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        modules_body = \
            [
                conv(n_feat*2, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat*2, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer()
        self.WCA = CALayer(n_feat*2, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x

        # Split 2 part
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path_in)
        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt
        wavelet_path = self.iwt(res)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out += self.conv1x1_final(residual)

        return out


if __name__ == '__main__':

    # 实例化HWAB
    hwab = HWAB(n_feat=64, o_feat=64)

    # 创建一个假的输入张量，例如形状为 [batch_size, channels, height, width]
    # 假设批次大小(batch_size)为1，通道数(channels)为64
    fake_input = torch.randn(1, 64, 128, 128)

    # 通过HWAB
    output = hwab(fake_input)

    # 打印输入输出的shape
    print("Input shape:", fake_input.shape)
    print("Output shape:", output.shape)