import numpy as np
import torch
from torch import nn
from torch.nn import init


class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):  # torch.Size([32, 784, 128])
        attn = self.mk(queries)  # torch.Size([32, 784, 8])

        attn = self.softmax(attn)  # torch.Size([32, 784, 8])
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # torch.Size([32, 784, 8])
        out = self.mv(attn)         # torch.Size([32, 784, 128])

        return out


# 输入 B C N,  输出 B C N
if __name__ == '__main__':
    block = ExternalAttention(d_model=128, S=8).cuda()
    input = torch.rand(32, 784, 128).cuda()
    output = block(input)
    print(input.size())
    print(output.size())
