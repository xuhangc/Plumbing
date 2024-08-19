import math
import torch
from torch import nn
# from torchstat import stat  # 查看网络参数
#Unsupervised Bidirectional Contrastive Reconstruction and Adaptive Fine-Grained Channel Attention Networks for image dehazing
#https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387
"""
Mix 类
实现了带有可学习因素的混合机制：
定义了一个可学习参数 w。
使用 Sigmoid 激活函数确保混合因素在 0 和 1 之间。
在前向传播过程中，使用混合因素对两个输入特征进行加权混合。

FCAttention 类
实现了细粒度通道注意力机制：
全局平均池化（GAP）： 定义了 avg_pool 层进行全局平均池化。
卷积操作： 使用 1D 卷积层 conv1 和全连接层 fc 处理池化后的特征。
注意力权重计算： 通过矩阵乘法和 Sigmoid 激活函数计算注意力权重：
将池化后的特征图通过 1D 卷积和全连接层分别得到两个中间特征。
使用矩阵乘法计算两个中间特征的乘积，得到注意力权重。
使用 Mix 类对两个注意力权重进行加权混合。
将最终的注意力权重应用到原始特征图上。
"""
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

#Adaptive Fine-Grained Channel Attention (FCA)
class FCAttention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out

if __name__ == '__main__':
    input = torch.rand(1,64,256,256)

    A = FCAttention(channel=64)
    #stat(A, input_size=[64, 1, 1])
    y = A(input)
    print(y.size())

