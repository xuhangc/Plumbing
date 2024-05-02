# ---------------------------------------
# 论文:ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks(CVPR 2020)
# Github地址: https://github.com/BangguWu/ECANet
# ---------------------------------------


import torch
from torch import nn


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = eca_layer()
    input = torch.rand(3, 32, 64, 64)
    output = block(input)
    print(input.size(), output.size())
