#B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection
#https://ieeexplore.ieee.org/document/10547405
import torch
import torch.nn as nn

class simam_module_3d(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module_3d, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, d, h, w = x.size()  # 注意这里多了一个维度d代表深度

        n = d * w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class diff_module_3d(nn.Module):
    def __init__(self, in_channel):
        super(diff_module_3d, self).__init__()
        self.avg_pool = nn.AvgPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module_3d()

    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        out = self.simam(out)
        return out

# Change Boundary-Aware Module(CBM) for 3D data
class CBM_3d(nn.Module):
    def __init__(self, in_channel):
        super(CBM_3d, self).__init__()
        self.diff_1 = diff_module_3d(in_channel)
        self.diff_2 = diff_module_3d(in_channel)
        self.simam = simam_module_3d()

    def forward(self, x1, x2):
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1 - d2)
        d = self.simam(d)
        return d

if __name__ == '__main__':
    # 实例化CBM_3d，这里我们以输入通道数为512为例，并假设深度 (depth) 为16
    cbm_module_3d = CBM_3d(in_channel=512)

    # 创建两个形状为[1, 512, 8, 16, 16]的随机tensor，代表batch size为1，通道数为512，体积深度为8，图像尺寸为16x16的输入
    x1 = torch.rand(1, 512, 16, 16, 16)
    x2 = torch.rand(1, 512, 16, 16, 16)

    # 通过CBM_3d模块
    output = cbm_module_3d(x1, x2)

    # 打印输入输出的shape
    print("Input shape1:", x1.shape)
    print("Input shape2:", x2.shape)
    print("Output shape:", output.shape)