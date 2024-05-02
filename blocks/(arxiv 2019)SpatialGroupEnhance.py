# ---------------------------------------
# 论文: “Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks.” ArXiv abs/1905.09646
# Github地址: https://github.com/implus/PytorchInsight
# ---------------------------------------
import torch
from torch import nn


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


#   输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)
    sge = SpatialGroupEnhance(groups=8)
    output = sge(input)
    print(output.shape)
