import torch
import torch.nn as nn

'''
https://github.com/qiulinzhang/SPConv.pytorch
Split to Be Slim: An Overlooked Redundancy in Vanilla Convolution
初始化：
通过指定的参数（如输入和输出通道数、步长、比例和减少因子），对SPConv_3x3层进行初始化。
定义了卷积层，包括深度可分离卷积（gwc和pwc）、1x1卷积（conv1x1）和池化层。

前向方法：
将输入张量根据指定的比例分成两部分。
对第一部分应用3x3卷积路径（gwc和pwc）。
对第二部分应用1x1卷积路径（conv1x1）。
基于每个路径输出的全局平均池化计算空间注意分数。
根据计算的注意力分数融合来自两个路径的输出。
'''

class SPConv_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, ratio=0.5, reduction=16):
        super(SPConv_3x3, self).__init__()
        self.inplanes_3x3 = int(inplanes*ratio)
        self.inplanes_1x1 = inplanes - self.inplanes_3x3
        self.outplanes_3x3 = int(outplanes*ratio)
        self.outplanes_1x1 = outplanes - self.outplanes_3x3
        self.outplanes = outplanes
        self.stride = stride

        self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
                             padding=1, groups=2, bias=False)
        self.pwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=1, bias=False)

        self.conv1x1 = nn.Conv2d(self.inplanes_1x1, self.outplanes,kernel_size=1)
        self.avgpool_s2_1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.avgpool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_add_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)
        self.ratio = ratio
        self.groups = int(1/self.ratio)
    def forward(self, x):
        b, c, _, _ = x.size()


        x_3x3 = x[:,:int(c*self.ratio),:,:]
        x_1x1 = x[:,int(c*self.ratio):,:,:]
        out_3x3_gwc = self.gwc(x_3x3)
        if self.stride ==2:
            x_3x3 = self.avgpool_s2_3(x_3x3)
        out_3x3_pwc = self.pwc(x_3x3)
        out_3x3 = out_3x3_gwc + out_3x3_pwc
        out_3x3 = self.bn1(out_3x3)
        out_3x3_ratio = self.avgpool_add_3(out_3x3).squeeze()

        # use avgpool first to reduce information lost
        if self.stride == 2:
            x_1x1 = self.avgpool_s2_1(x_1x1)

        out_1x1 = self.conv1x1(x_1x1)
        out_1x1 = self.bn2(out_1x1)
        out_1x1_ratio = self.avgpool_add_1(out_1x1).squeeze()

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
        out = out_1x1 * (out_31_ratio[:,:,1].view(b, self.outplanes, 1, 1).expand_as(out_1x1))\
              + out_3x3 * (out_31_ratio[:,:,0].view(b, self.outplanes, 1, 1).expand_as(out_3x3))

        return out
if __name__ == '__main__':
    # 定义测试输入张量,需要确保inplanes_3x3是groups的倍数,groups=2
    batch_size = 3
    channels = 64
    height, width = 64, 64
    input_tensor = torch.randn(batch_size, channels, height, width)

    # 创建SPConv_3x3的实例
    spconv_layer = SPConv_3x3(64, 32)

    # 前向传播
    output_tensor = spconv_layer(input_tensor)

    # 打印输入和输出的形状
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)