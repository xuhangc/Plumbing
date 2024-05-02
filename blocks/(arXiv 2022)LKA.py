# ---------------------------------------
# 论文: Visual Attention Network (arXiv 2022)
# Github地址: https://github.com/Visual-Attention-Network/VAN-Classification
# ---------------------------------------
import torch
from torch import nn


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = LKA(64)
    input = torch.rand(3, 64, 64, 64)
    output = block(input)
    print(input.size(), output.size())
