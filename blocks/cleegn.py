import torch
import torch.nn as nn
#CLEEGN: A Convolutional Neural Network for Plug-and-Play Automatic EEG Reconstruction
#https://arxiv.org/pdf/2210.05988v2.pdf

class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)


class CLEEGN(nn.Module):
    def __init__(self, n_chan, fs, N_F=20, tem_kernelLen=0.1):
        super(CLEEGN, self).__init__()
        self.n_chan = n_chan
        self.N_F = N_F
        self.fs = fs
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_chan, (n_chan, 1), padding="valid", bias=True),
            Permute2d((0, 2, 1, 3)),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.99)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(N_F, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(N_F, n_chan, (n_chan, 1), padding="same", bias=True),
            nn.BatchNorm2d(n_chan, eps=1e-3, momentum=0.99)
        )
        self.conv5 = nn.Conv2d(n_chan, 1, (n_chan, 1), padding="same", bias=True)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.conv2(x)
        # decoder
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        return x

"""
这个模型 CLEEGN 是为处理 EEG 数据而设计的，其输入应该是具有特定形状的张量，符合模型的期待输入。以下是输入和输出的一个示例以及它们的形状。
我们可以假设输入数据是一个时间序列信号，其中
batch_size 是批次大小
channel 是 EEG 信号的通道数
height 对应于通道数，因为第一个卷积层期望看到的是 (batch_size, 1, channel, width)
width 是信号的时间长度
为了保持简单，我们可以使用一个具有 1 个样本（批次大小为 1）、56 个通道以及 128 个时间点步长的张量来模拟输入 EEG 数据。
"""
if __name__ == '__main__':
    # 假设有以下参数
    batch_size = 1
    n_chan = 56  # 通道数，根据模型初始化参数
    fs = 128.0  # 采样频率，根据模型初始化参数
    width = int(fs)  # 例如，我们取一个秒的数据

    # 初始化模型
    model = CLEEGN(n_chan=56, fs=128.0, N_F=20, tem_kernelLen=0.1)

    # 创建一个随机的输入张量
    x = torch.randn(batch_size, 1, n_chan, width)  # (batch_size, channels, height, width)

    # 前向传播
    y = model(x)

    # 打印输入输出的shape
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
