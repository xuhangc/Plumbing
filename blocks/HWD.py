import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
#Haar wavelet downsampling: A simple but effective downsampling module for semantic segmentation
#https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174
"""
__init__ 函数：
in_ch：输入通道的数量。
out_ch：卷积操作后的输出通道数量。
self.wt：创建了一个DWTForward实例，设定分解级数为1（J=1），填充模式为zero，选择haar作为小波变换使用的波形。
self.conv_bn_relu：一个顺序容器，先通过一个2D卷积（核大小为1，步幅为1）将通道维度从in_ch * 4增加到out_ch，然后进行批量标准化，接着应用ReLU激活函数。
forward 函数：
forward方法定义了每次调用该模块时的计算过程。
yL, yH：将前向小波变换应用于输入x后的结果，其中yL是低频系数，yH是高频系数。
y_HL、y_LH、y_HH：从元组yH中提取在水平（HL）、垂直（LH）和对角（HH）方向上的高频分量。
将高频分量和低频分量yL沿着通道维度拼接成一个新的张量。
这个拼接后的张量随后通过conv_bn_relu来产生带有out_ch通道的输出。
"""
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x


if __name__ == '__main__':
    # 实例化Down_wt模块
    down_wt = Down_wt(in_ch=64, out_ch=128)

    # 创建一个4维张量来模拟输入数据，形状为(batch_size, channels, height, width)
    # 例如：batch_size = 1, channels = 3, height = 64, width = 64
    input_tensor = torch.randn(1, 64, 64, 64)

    # 通过Down_wt模块运行输入张量
    output_tensor = down_wt(input_tensor)

    # 打印输入和输出的形状
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")