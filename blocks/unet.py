import torch
import torch.nn as nn
import torch.nn.functional as F
from axial import AxialAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        """
        定义卷积块，用于构建U-Net中的卷积层部分。

        参数：
            ch_in (int)：输入通道数。
            ch_out (int)：输出通道数。
        """
        super(conv_block, self).__init__()
        # 定义卷积块，包括卷积、批归一化和ReLU激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x (tensor)：输入张量。

        返回：
            tensor：经过卷积块处理后的张量。
        """
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        """
        定义上采样卷积块，用于构建U-Net中的上采样部分。

        参数：
            ch_in (int)：输入通道数。
            ch_out (int)：输出通道数。
        """
        super(up_conv, self).__init__()
        # 定义上采样卷积块，包括上采样、卷积、批归一化和ReLU激活函数
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x (tensor)：输入张量。

        返回：
            tensor：经过上采样卷积块处理后的张量。
        """
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        """
        定义U-Net模型。

        参数：
            img_ch (int)：输入图像的通道数，默认为3（RGB图像）。
            output_ch (int)：输出图像的通道数，默认为1。
        """
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        # 定义U-Net的编码部分
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # 定义U-Net的解码部分
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        # 最后的1x1卷积层用于输出
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.avial1 = AxialAttention(in_planes=64, out_planes=64, groups=1, kernel_size=224, stride=1, bias=False, width=False)


#在forward里用模块
    def forward(self, x):
        """
        前向传播函数。

        参数：
            x (tensor)：输入张量。

        返回：
            tensor：经过U-Net模型处理后的张量。
        """
        x1 = self.Conv1(x)
        print(x1.shape)
        x1 = self.avial1(x1)#加在这里
        print(x1.shape)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)



        # 解码路径
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)


        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # 使用softmax函数进行多分类任务中的输出
        d1 = F.softmax(d1, dim=1)

        return d1


if __name__ == '__main__':
    # 实例化U-Net模型并放置在GPU上
    net = U_Net(1, 2).cuda()
    # 创建输入张量
    in1 = torch.randn(1, 1, 224, 224).cuda()
    # 前向传播
    out = net(in1)
    # 输出张量大小
    print(out.size())