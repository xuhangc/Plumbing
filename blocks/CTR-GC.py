# https://github.com/Uason-Chen/CTR-GCN/blob/main/model/ctrgcn.py
import torch
import torch.nn as nn

"""
图卷积网络（GCNs）在基于骨架的动作识别中得到了广泛的应用，并取得了显著的成果。
在GCN中，图拓扑在特征聚合中占主导地位，因此是提取代表性特征的关键。
在这项工作中，我们提出了一种新颖的通道拓扑细化图卷积（CTR-GC），以动态学习不同的拓扑结构，并有效地聚合不同通道中的联合特征，以进行基于骨架的动作识别。
所提出的CTR-GC通过学习共享拓扑作为所有通道的通用先验，并使用每个通道的特定于通道的相关性对其进行优化，从而对通道拓扑进行建模。
我们的细化方法引入了很少的额外参数，并显着降低了通道拓扑建模的难度。

虽然GCN处理的是图数据（非规则数据），但在实现时
为了充分利用深度学习框架（如PyTorch、TensorFlow）和硬件加速（特别是GPU）的优势
通常会采用类似于处理图像的方法来组织和传递数据。
"""

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1



if __name__ == '__main__':
    block = CTRGC(in_channels=64, out_channels=64)
    input = torch.rand(32, 64, 9, 9)
    output = block(input)
    print(input.size())
    print(output.size())