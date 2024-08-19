import torch
import torch.nn as nn
#DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection
#https://ieeexplore.ieee.org/abstract/document/10504297
"""
MSFF（多尺度特征融合）：
使用多个卷积路径（conv1到conv4），将不同尺度的特征进行融合。
将各输出串联起来，并通过convmix混合它们，得到融合的特征表示。

MDFM（多尺度差异融合模块）：
使用conv_sub增强输入特征（x1和x2）之间的差异。
通过注意力（x_att）和特征融合（MPFL）调整x1和x2。
使用convmix将调整后的特征（x1和x2）进行最终的特征融合。
使用conv_dr生成最终的输出特征。
"""
# multiscale feature fusion (MSFF)
class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.convmix = nn.Sequential(nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.convmix(x_f)

        return out

#Multiscale Difference Fusion Module (MDFM)
class MDFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.MPFL = MSFF(inchannel=in_d, mid_channel=64)   ##64

        self.conv_diff_enh = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, 3,  padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        self.convmix = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, groups=self.in_d, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # difference enhance
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))
        x1 = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        x2 = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))


        # fusion
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)

        # after ca
        x_f = x_f * x_att
        out = self.conv_dr(x_f)

        return out

if __name__ == '__main__':
    x1 = torch.randn((32, 512, 8, 8))
    x2 = torch.randn((32, 512, 8, 8))
    model = MDFM(512, 64)
    out = model(x1, x2)
    print(out.shape)