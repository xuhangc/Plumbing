# ---------------------------------------
# 论文: Squeeze-and-Excitation Networks (CVPR 2018)
# ---------------------------------------
import torch
from torch import nn
from torch.nn import init


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


#   输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)
    se = SEAttention(channel=32, reduction=8)
    output = se(input)
    print(output.shape)
