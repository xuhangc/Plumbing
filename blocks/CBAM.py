import torch
from einops import rearrange
from torch import nn
#今天我想给大家介绍一下，怎么替换原有模块中的注意力，换成其他模块的注意力，从而获得一个新的模块
#我这里就用CBAM举个例子，我用Multi-DConv Head Transposed Self-Attention (MDTA)这个注意力来替代原模块中的channel attention model
#先看原来的模块模型图
#再看MDTA
#先运行一下，直接copy过了，看看需要改哪些参数
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape


        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # [B, head, C/head, HW] * [B, head, HW, C/head] * [head, 1, 1] ==> [B, head, C/head, C/head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # [B, head, C/head, C/head] * [B, head, C/head, HW] ==> [B, head, C/head, HW]
        out = (attn @ v)

        # [B, head, C/head, HW] ==> [B, head, C/head, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = True, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = Attention(dim, num_heads, bias)#改成copy过来的
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


if __name__ == '__main__':
    block = CBAM(64)
    input_tensor = torch.rand(3, 64, 128, 128)
    output_tensor = block(input_tensor)
    print(block)#打印模型
    # 打印输入和输出张量的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    #截个图备用2,来对比一下网络结构，可以看到原来的channel attention 已经替换成了MDTA，谢谢观看