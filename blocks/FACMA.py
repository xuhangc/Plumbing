from torch import nn
import math
#FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
#https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848
"""
get_1d_dct 函数：
这个函数用于计算单维度的离散余弦变换（DCT），给定索引i、频率freq和长度L。在信号和图像处理中广泛用于将数据转换到频率空间。
通过结合不同轴的输出来为2D DCT权重的计算。

get_dct_weights 函数：
根据给定的宽度、高度、通道以及特定频率fidx_u和fidx_v生成2D DCT权重。这些权重预先计算了神经网络层中所有通道的频率分量，便于将频率域信息纳入学习过程。
这对于后续利用频率感知操作的注意力块来说至关重要。

FCABlock 类（频率通道注意力块）:
实现了参考FcaNet论文中提出的频率通道注意力机制。它使用预先计算的DCT权重来调制输入的特征图，然后是通道级注意力机制。
其目的是通过强调重要的频率分量来加强显著性特征学习。

SFCA 类（空间频率通道注意力）:
通过引入空间注意力到频率增强特征来扩展FCABlock。它结合了频率通道注意力和空间上下文注意力，以进一步精炼显著性检测。
结合的设计旨在利用全局频率信息和局部空间细节。

FACMA 类（频率感知交叉模态注意力）:
通过分别对每种模态使用之前定义的SFCA块，代表了结合RGB和深度模式信息的主要机制。
首先通过各自的SFCA块处理深度输入和RGB输入。然后，将经过处理的深度特征与RGB特征逐元素相乘，反之亦然。这种跨模态交互旨在通过结合两个源的互补信息来丰富特征表示。
"""
def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i+0.5)/L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)
def get_dct_weights(width,height,channel,fidx_u,fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i*c_part: (i+1)*c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    return dct_weights
class FCABlock(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel,width,height,fidx_u, fidx_v, reduction=16):
        super(FCABlock, self).__init__()
        mid_channel = channel // reduction
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(width,height,channel,fidx_u,fidx_v))
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)
class SFCA(nn.Module):
    def __init__(self, in_channel,width,height,fidx_u,fidx_v):
        super(SFCA, self).__init__()

        fidx_u = [temp_u * (width // 8) for temp_u in fidx_u]
        fidx_v = [temp_v * (width // 8) for temp_v in fidx_v]
        self.FCA = FCABlock(in_channel, width, height, fidx_u, fidx_v)
        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, x):
        # FCA
        F_fca = self.FCA(x)
        #context attention
        con = self.conv1(x) # c,h,w -> 1,h,w
        con = self.norm(con)
        F_con = x * con
        return F_fca + F_con
class FACMA(nn.Module):
    def __init__(self,in_channel,width,height,fidx_u,fidx_v):
        super(FACMA, self).__init__()
        self.sfca_depth = SFCA(in_channel, width, height, fidx_u, fidx_v)
        self.sfca_rgb   = SFCA(in_channel, width, height, fidx_u, fidx_v)
    def forward(self, rgb, depth):
        out_d = self.sfca_depth(depth)
        out_d = rgb * out_d

        out_rgb = self.sfca_rgb(rgb)
        out_rgb = depth * out_rgb
        return out_rgb, out_d

if __name__ == '__main__':
    import torch

    # 定义输入参数
    in_channel = 64
    width = 224
    height = 224
    fidx_u = [0, 1]
    fidx_v = [0, 1]

    # 实例化FACMA
    facma = FACMA(in_channel, width, height, fidx_u, fidx_v)

    # 假设的RGB和深度输入
    rgb_input = torch.randn(1, in_channel, width, height)  # Batch size为1
    depth_input = torch.randn(1, in_channel, width, height)  # Batch size为1

    # 通过FACMA
    out_rgb, out_d = facma(rgb_input, depth_input)

    # 打印输入输出形状
    print("RGB 输入形状:", rgb_input.shape)
    print("深度 输入形状:", depth_input.shape)
    print("RGB 输出形状:", out_rgb.shape)
    print("深度 输出形状:", out_d.shape)