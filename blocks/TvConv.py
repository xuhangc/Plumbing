import torch
import torch.nn as nn
#TVConv：用于布局感知视觉处理的高效平移变体卷积
from MSCA import AttentionModule


class _ConvBlock(nn.Sequential):
    """
    _ConvBlock类定义了一个简单的卷积块，包含卷积层、层归一化和ReLU激活函数。
    """
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.LayerNorm([out_planes, h, w]),  # 层归一化
            nn.ReLU(inplace=True)
        )

class TVConv(nn.Module):
    """
    TVConv类定义了一个基于位置映射的空间变体卷积模块。
    """
    def __init__(self,
                 channels,
                 TVConv_k=3,
                 stride=1,
                 TVConv_posi_chans=4,
                 TVConv_inter_chans=64,
                 TVConv_inter_layers=3,
                 TVConv_Bias=False,
                 h=3,
                 w=3,
                 **kwargs):
        super(TVConv, self).__init__()

        # 注册缓冲区变量
        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k**2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))

        self.bias_layers = None

        out_chans = self.TVConv_k_square * self.channels

        # 初始化位置映射参数
        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)

        # 创建权重层和偏置层
        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)

        # 初始化 Unfold 模块
        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k-1)//2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        """
        创建卷积层序列。
        """
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(
            in_channels=inter_chans,
            out_channels=out_chans,
            kernel_size=3,
            padding=1,
            bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        使用位置图生成空间变体卷积。
        :param x: 输入张量
        :return: 输出张量
        """
        # 计算卷积权重
        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w)
        # 利用 Unfold 模块获取局部区域，并按照权重进行加权求和
        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w)
        out = (weight * out).sum(dim=2)

        if self.bias_layers is not None:
            # 如果使用偏置，则加上偏置
            bias = self.bias_layers(self.posi_map)
            out = out + bias

        return out

class ConvBNReLU(nn.Sequential):
    """
    ConvBNReLU类定义了一个卷积、批归一化和ReLU激活函数的组合模块。
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class TVConvBNReLU(nn.Sequential):
    """
    TVConvBNReLU类定义了包含TVConv、批归一化和ReLU激活函数的组合模块。
    """
    def __init__(self, planes, h_w, stride=1, norm_layer=None, **kwargs):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(TVConvBNReLU, self).__init__(
            TVConv(planes, h=h_w, w=h_w, stride=stride, **kwargs),
            norm_layer(planes),
            nn.ReLU6(inplace=True)
        )

class TVConvInvertedResidual(nn.Module):
    """
    TVConvInvertedResidual类定义了基于TVConv的倒残差模块。
    """
    def __init__(self, inp, oup, stride, h_w, expand_ratio=6, norm_layer=None, **kwargs):
        super(TVConvInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            TVConvBNReLU(hidden_dim, h_w, stride=stride, norm_layer=norm_layer, **kwargs),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    """
    InvertedResidual类定义了一个标准的倒残差模块，使用标准的卷积、批归一化和ReLU激活函数。
    """
    def __init__(self, inp, oup, stride, h_w, expand_ratio=6, norm_layer=None, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

if __name__ == "__main__":
    # 生成随机的输入和位置映射
    input_tensor = torch.rand(1, 64, 32, 32)  # 输入张量为NCHW格式

    # 创建TVConv模块
    tvconv = TVConv(64,h=32, w=32)

    # 运行TVConv模块
    output_tensor = tvconv(input_tensor)

    # 创建一个AttentionModule实例（MSCA模块）
    attention_module = AttentionModule(64)
    output_tensor2 = attention_module(input_tensor)

    #直接相加就完事儿了
    output = output_tensor + output_tensor2

    # 打印输出张量的形状
    print("Output shape:", output.shape)
    #该模块可直接缝，适用于人脸识别和医学图像分割
