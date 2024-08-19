import torch.nn as nn
import torch
# EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# https://arxiv.org/pdf/2405.06880

def channel_shuffle(x, groups):
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, depth, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

#   Efficient up-convolution block (EUCB) for 3D data
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm3d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

if __name__ == '__main__':
    # 示例输入
    x = torch.randn(1, 32, 32, 64, 64)  # 输入特征图

    # 实例化EUCB
    eucb = EUCB(in_channels=32, out_channels=64)

    # 打印输入的shape
    print("EUCB输入的 shape:", x.shape)

    # 前向传播并打印输出的shape
    output = eucb(x)
    print("EUCB输出的 shape:", output.shape)
