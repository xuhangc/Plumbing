#B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection
#https://ieeexplore.ieee.org/document/10547405
import torch
import torch.nn as nn
"""
“双时相特征聚合模块”（BFAM）。这个模块以一种创新的方式融合了两个时间点的输入特征，并通过使用SIMAM（一个简单的、无参数的注意力模块）以及不同膨胀率的卷积层来增强和细化特征。
具体来说，BFAM首先将来自两个不同时间点（inp1和inp2）的输入特征进行拼接。
随后，通过多个具有不同膨胀率的卷积层（conv_1到conv_4）来处理这些拼接的特征，以捕捉不同尺度的上下文信息。
这些特征再次被合并，并通过一个卷积块（fuse模块）来融合，生成更加丰富和细化的特征表示。
此外，BFAM利用了SIMAM模块（pre_siam和lat_siam）来分别增强inp1和inp2中的特征，通过自适应地调整特征重要性来提升模型注意力机制的性能。
接着，这些增强的特征与通过fuse模块细化后的特征进行组合，并再次通过一个SIMAM模块(fuse_siam)进行处理，以进一步聚焦于重要的变化区域。
最终，所有这些特征都会被合并，并通过另一卷积序列(out)进行最终的融合和输出。
如果存在last_feature（即前一个阶段的特征），它也会被加入到最终的特征融合中，以利用先前阶段的信息进行更加细致的变化检测。
整个BFAM模块通过结合多尺度上下文信息、注意力机制以及特征增强，旨在为多时相遥感影像提供一种高效和准确的变化检测能力。
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

#bitemporal feature aggregation module (BFAM)
class BFAM(nn.Module):
    def __init__(self,inp,out):
        super(BFAM, self).__init__()

        self.pre_siam = simam_module()
        self.lat_siam = simam_module()


        out_1 = int(inp/2)

        self.conv_1 = nn.Conv2d(inp, out_1 , padding=1, kernel_size=3,groups=out_1,
                                   dilation=1)
        self.conv_2 = nn.Conv2d(inp, out_1, padding=2, kernel_size=3,groups=out_1,
                                   dilation=2)
        self.conv_3 = nn.Conv2d(inp, out_1, padding=3, kernel_size=3,groups=out_1,
                                   dilation=3)
        self.conv_4 = nn.Conv2d(inp, out_1, padding=4, kernel_size=3,groups=out_1,
                                   dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )

        self.fuse_siam = simam_module()

        self.out = nn.Sequential(
            nn.Conv2d(out_1, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self,inp1,inp2,last_feature=None):
        x = torch.cat([inp1,inp2],dim=1)
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        cat = torch.cat([c1,c2,c3,c4],dim=1)
        fuse = self.fuse(cat)
        inp1_siam = self.pre_siam(inp1)
        inp2_siam = self.lat_siam(inp2)


        inp1_mul = torch.mul(inp1_siam,fuse)
        inp2_mul = torch.mul(inp2_siam,fuse)
        fuse = self.fuse_siam(fuse)
        if last_feature is None:
            out = self.out(fuse + inp1 + inp2 + inp2_mul + inp1_mul)
        else:
            out = self.out(fuse+inp2_mul+inp1_mul+last_feature+inp1+inp2)
        out = self.fuse_siam(out)

        return out


if __name__ == '__main__':
    # 假设的输入通道数和输出通道数
    input_channels = 128
    output_channels = 256

    # 输入数据的大小
    batch_size = 1
    height, width = 16, 16

    # 实例化BFAM
    bfam = BFAM(inp=input_channels, out=output_channels)

    # 创建两个模拟输入
    inp1 = torch.rand(batch_size, input_channels // 2, height, width)
    inp2 = torch.rand(batch_size, input_channels // 2, height, width)
    last_feature = torch.rand(batch_size, input_channels // 2, height, width)

    # 通过BFAM模块，这里没有提供last_feature的话，可以为None
    output = bfam(inp1, inp2, last_feature)
    # output = bfam(inp1, inp2)

    # 打印输入和输出的shape
    print("inp1 shape:", inp1.shape)
    print("inp2 shape:", inp2.shape)
    print("Output shape:", output.shape)