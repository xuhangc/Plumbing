import torch
from torch import nn

"""
这个模块是一个自定义的神经网络层，用于在输入数据中选择具有显著性的位置或特征。

该模块的作用如下：

选择显著位置：通过使用Salient Positions Selection (SPS) 算法，从输入数据中选择具有显著性的位置或特征。这些显著位置可能对于后续的任务具有重要的影响。

自适应选择特征数目：根据参数中设置的自适应选项，可以动态地选择特征的数量。当设置为自适应时，特征的数量会根据输入数据的通道数目进行调整。

学习特征数目：如果学习参数设置为True，模块将学习选择的特征数目。这意味着网络可以根据训练数据动态地调整特征的数量，以优化任务的性能。

选择特征的计算方法：可以根据参数中设置的不同模式来选择特征。目前，支持的模式是'pow'，它使用了一种幂的计算方法来确定特征的显著性。

总的来说，这个模块可以用于在神经网络中自动选择具有显著性的特征或位置，以提高网络的性能和效率。
"""


class SPABlock(nn.Module):
    def __init__(self, in_channels, k=784, adaptive = False, reduction=16, learning=False, mode='pow'):
        """
        Salient Positions Selection (SPS) algorithm
        :param in_channels: 待处理数据的通道数目
        :param k=5, 默认的选择通道数目
        :param kadaptive = False: k是否需要根据通道数进行自适应选择
        :param learning=False: k是否需要学习
        :param mode='power':挑选k个位置的计算方法
        :return out, [batchsize, self.k, channels]
        """
        super(SPABlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.k = k
        self.adptive = adaptive
        self.reduction = reduction
        self.learing = learning
        if self.learing is True:
            self.k = nn.Parameter(torch.tensor(self.k))

        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, return_info=False):
        input_shape = x.shape
        if len(input_shape)==4:
            x = x.view(x.size(0), self.in_channels, -1)
            x = x.permute(0, 2, 1)
        batch_size,N = x.size(0),x.size(1)

        #（B, H*W，C）
        if self.mode == 'pow':
            x_pow = torch.pow(x,2)# （batchsize，H*W，channel）
            x_powsum = torch.sum(x_pow,dim=2)# （batchsize，H*W）

        if self.adptive is True:
            self.k = N//self.reduction
            if self.k == 0:
                self.k = 1

        outvalue, outindices = x_powsum.topk(k=self.k, dim=-1, largest=True, sorted=True)

        outindices = outindices.unsqueeze(2).expand(batch_size, self.k, x.size(2))
        out = x.gather(dim=1, index=outindices).to(self.device)

        if return_info is True:
            return out, outindices, outvalue
        else:
            return out

if __name__ == '__main__':
    block = SPABlock(in_channels=128)
    input = torch.rand(32, 784, 128)
    output = block(input)
    print(input.size())
    print(output.size())