# @Time    : 2023/3/17 15:56
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : Module.py
# @Software: PyCharm
import torch.nn as nn
import torch.utils.data
import torch
#ABC: Attention with Bilinear Correlation for Infrared Small Target Detection ICME2023
#https://arxiv.org/pdf/2303.10321
"""
摘要的翻译：
红外小目标检测（ISTD）在预警、救援和引导等方面有着广泛的应用。
然而，基于CNN的深度学习方法对缺乏清晰轮廓和纹理特征的红外小目标（IRST）分割效果不佳，而基于Transformer的方法由于缺乏卷积诱导偏差也难以取得显著的效果。
为了解决这些问题，我们提出了一种新的模型，称为双线性相关注意模型（ABC），该模型基于Transformer架构，包括一个卷积线性融合Transformer（CLFT）模块，
该模块具有用于特征提取和融合的新型注意机制，可有效增强目标特征并抑制噪声。此外，我们的模型包括一个位于网络较深层的U形卷积-扩张卷积（UCDC）模块，
它利用较深层特征的较小分辨率来获得更精细的语义信息。在公共数据集上的实验结果表明，我们的方法达到了最先进的性能。

本文的主要贡献如下：
1）基于 Transformer 结构设计的 CLFT 模块可以有效增强目标特征并抑制噪声。
2）UCDC 模块充分利用了深度特征的特点，可以更精细地处理网络的深度特征。
3）所提出的方法在所有现有的公开数据集上都达到了最佳性能。
"""
def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


#u-shaped convolution-dilated convolution (UCDC)
class UCDC(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UCDC, self).__init__()
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        self.dconv1 = conv_relu_bn(out_ch, out_ch // 2, 2)
        self.dconv2 = conv_relu_bn(out_ch // 2, out_ch // 2, 4)
        self.dconv3 = conv_relu_bn(out_ch, out_ch, 2)
        self.conv2 = conv_relu_bn(out_ch * 2, out_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))
        out = self.conv2(torch.cat((x1, dx3), dim=1))
        return out


if __name__ == '__main__':
    # Define input dimensions
    in_ch = 64  # Input channels
    out_ch = 64  # Output channels

    # Instantiate the UCDC model
    ucdc = UCDC(in_ch, out_ch)

    # Create a sample input tensor with shape (batch_size, in_channels, height, width)
    batch_size = 1
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, in_ch, height, width)

    # Print input shape
    print("Input shape:", input_tensor.shape)

    # Pass the input tensor through the model
    output_tensor = ucdc(input_tensor)

    # Print output shape
    print("Output shape:", output_tensor.shape)
