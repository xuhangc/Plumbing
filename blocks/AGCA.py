import torch
import torch.nn as nn
from torch.nn import init

"""
表面缺陷检测是钢铁生产过程中的重要组成部分。近年来，注意力机制已广泛应用于钢材表面缺陷检测，以保证产品质量。
现有的注意力模块无法区分钢材表面图像和自然图像之间的差异。因此，我们提出了一种自适应图通道注意力（adaptive graph channel attention）模块，它将图卷积理论引入通道注意力中。 
AGCA模块将每个通道作为一个特征顶点，它们的关系用邻接矩阵来表示。我们通过分析 AGCA 中构建的图来对特征执行非局部（NL）操作。该操作显着提高了特征表示能力。
与其他注意力模块类似，AGCA模块具有轻量级和即插即用的特点。它使模块能够轻松嵌入到缺陷检测网络中。
"""


class AGCA(nn.Module):  # AGCA模块本质上是一种自注意力机制，其设计灵感来自于图卷积网络中的邻域聚合和信息传播的思想。因此，可以将AGCA模块视为一种特定形式的图卷积层。
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # Choose to deploy A0 on GPU or CPU according to your needs
        self.A0 = torch.eye(hide_channel).to('cuda')
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y

if __name__ == '__main__':
    block = AGCA(in_channel=64, ratio=4).to('cuda')
    input = torch.rand(1, 64, 32, 32).to('cuda')
    output = block(input)
    print(input.size())
    print(output.size())