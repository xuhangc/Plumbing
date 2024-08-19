import torch
from torch import nn
#UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation
#https://arxiv.org/abs/2107.02368
"""
conv类：这是一个可根据输入参数（如in_channels、out_channels、kernel_size等）进行调整的自定义卷积层。
提供了指定参数的灵活性，比如填充（‘same’、‘valid’或自定义填充）、批量归一化以及是否包含ReLU激活函数。
为了实现‘same’填充而计算填充大小的方法值得注意，因为它确保了输出特征图保持了原始输入尺寸。

self_attn类：该类实现了一个自注意力机制，这是关注输入特征图中对任务更相关部分的关键组件。
这种注意力机制使用了与Transformer模型中相似的查询（query）、键（key）和值（value）概念，
尽管它是应用在卷积上下文中的。mode参数允许注意力机制沿着高度（h）、宽度（w）或两个维度进行聚焦。

PAA类：平行轴向注意力类通过结合卷积和注意力机制来处理输入特征图，通过沿高度和宽度分别应用注意力机制，然后组合关注特征，增强了模型关注相关空间信息的能力。
这是通过一系列具有不同核大小的卷积和应用轴（高度和宽度）自注意力机制来实现的。

PAA_e类： 该模块作为利用PAA机制的编码器。它包含多个分支，每个分支都应用不同感受野大小的PAA操作，然后结合这些分支的输出。
结合后的特征图进一步处理以整合额外的空间信息，然后通过残差连接加回原始输入特征。
"""
class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
#Parallel Axial Attention(PAA)
class PAA(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(PAA, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x
#Parallel Axial Attention encoder(PAA_e)
class PAA_e(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PAA_e, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = PAA(in_channel, out_channel, 3)
        self.branch2 = PAA(in_channel, out_channel, 5)
        self.branch3 = PAA(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)#h和w减半

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        # x = self.Maxpool(x)


        return x


if __name__ == '__main__':
    # 实例化PAA_e，例如，假设输入特征图的通道数为64，我们想要输出通道数为128的特征图
    paa_encoder = PAA_e(in_channel=64, out_channel=128)

    # 创建一个模拟输入的张量，假设我们的输入尺寸为[批次大小, 通道数, 高, 宽]，例如：[1, 64, 256, 256]
    input_tensor = torch.randn(1, 64, 256, 256)

    # 将输入传递给实例化的PAA_e对象
    output_tensor = paa_encoder(input_tensor)

    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")