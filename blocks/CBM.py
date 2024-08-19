#B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection
#https://ieeexplore.ieee.org/document/10547405
import torch
import torch.nn as nn
"""
SIMAM模块： 一个简单的、无参数的注意力模块，通过强调信息丰富的区域和抑制不太有用的区域来提高特征的质量。
它通过调整输入与均值的偏差，并乘以一个包含小lambda（e_lambda）的因子来实现，这个因子是为了防止除以零的情况发生。

Diff模块： 该模块旨在增加网络对特征变化的敏感性。它通过计算输入与其平滑后（平均）的版本之间的差异，然后应用SIMAM模块来精细地关注重要变化。

变化边界感知模块（CBM）： 该模块整合了Diff模块和SIMAM的能力，有效捕捉和突出两个输入特征之间发生变化的边界。
这似乎特别适用于需要高度敏感边界的任务，如多时相数据集的变化检测，在这个任务中准确识别变化区域至关重要。
"""
#Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021)
class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class diff_moudel(nn.Module):
    def __init__(self,in_channel):
        super(diff_moudel, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module()
    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # weight = self.conv_1(edge)
        out = weight * x + x
        out = self.simam(out)
        return out
# Change Boundary-Aware Module(CBM)
class CBM(nn.Module):
    def __init__(self,in_channel):
        super(CBM, self).__init__()
        self.diff_1 = diff_moudel(in_channel)
        self.diff_2 = diff_moudel(in_channel)
        self.simam = simam_module()
    def forward(self,x1,x2):
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1-d2)
        d = self.simam(d)
        return d


if __name__ == '__main__':
    # 实例化CBM，这里我们以输入通道数为3为例
    cbm_module = CBM(in_channel=512)

    # 创建两个形状为[1, 512, 16, 16]的随机tensor，代表batch size为1，通道数为512，图像尺寸为16x16的输入
    x1 = torch.rand(1, 512, 16, 16)
    x2 = torch.rand(1, 512, 16, 16)

    # 通过CBM模块
    output = cbm_module(x1, x2)

    # 打印输入输出的shape
    print("Input shape1:", x1.shape)
    print("Input shape2:", x2.shape)
    print("Output shape:", output.shape)