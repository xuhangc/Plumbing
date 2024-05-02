import torch
import torch.nn as nn
from einops import rearrange

# RFAConv：创新空间注意力和标准卷积操作
"""
空间注意力已被证明可以使卷积神经网络专注于关键信息以提高网络性能，但它仍然有局限性。
我们从一个新的角度解释了空间注意力的有效性，即空间注意力机制本质上解决了卷积核参数共享的问题。
然而，对于大尺寸卷积核来说，空间注意力生成的注意力图中包含的信息仍然缺乏。
因此，我们提出了一种新的注意力机制，称为感受场注意力（RFA）。
RFA设计的感受场注意力卷积运算（RFAConv）可以被认为是一种替代标准卷积的新方法，它带来的计算成本和许多参数几乎可以忽略不计。
在Imagenet-1k、MS COCO和VOC上的大量实验表明，我们的方法在分类、目标检测和语义分割任务中具有卓越的性能。
重要的是，我们认为，对于目前一些只关注空间特征的空间注意力机制，是时候通过关注感受野空间特征来提高网络的性能了。
"""

class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
        self.get_weights = nn.Sequential(
            nn.Conv2d(in_channel * (kernel_size ** 2), in_channel * (kernel_size ** 2), kernel_size=1,
                      groups=in_channel),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)))

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=kernel_size)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        unfold_feature = self.unfold(x)  # 获得感受野空间特征  b c*kernel**2,h*w
        x = unfold_feature
        data = unfold_feature.unsqueeze(-1)
        weight = self.get_weights(data).view(b, c, self.kernel_size ** 2, h, w).permute(0, 1, 3, 4, 2).softmax(-1)
        weight_out = rearrange(weight, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size, n2=self.kernel_size) # b c h w k**2 -> b c h*k w*k
        receptive_field_data = rearrange(x, 'b (c n1) l -> b c n1 l', n1=self.kernel_size ** 2).permute(0, 1, 3, 2).reshape(b, c, h, w, self.kernel_size ** 2) # b c*kernel**2,h*w ->  b c h w k**2
        data_out = rearrange(receptive_field_data, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size,n2=self.kernel_size) # b c h w k**2 -> b c h*k w*k
        conv_data = data_out * weight_out
        conv_out = self.conv(conv_data)
        return self.act(self.bn(conv_out))


if __name__ == '__main__':
    block = RFAConv(in_channel=64, out_channel=64)
    input = torch.rand(32, 64, 9, 9)
    output = block(input)
    print(input.size())
    print(output.size())
