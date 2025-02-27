import torch.nn as nn
import torch.nn.functional as F
#PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion
#2D图像分割即插即用特征融合模块


class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv2dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        # 或者使用以下代码，根据需要选择
        # self.conv1 = Conv2dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0] = -1
        kernel[:, :, 0, 2] = 1
        kernel[:, :, 2, 0] = 1
        kernel[:, :, 2, 2] = -1

        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        guidance = self.conv1(guidance)

        x_diff = F.conv2d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv2d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out


class SDM(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDM, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature

        return boundary_enhanced


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Conv2dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dbn, self).__init__(conv, bn)


class Conv2dGNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()

        gn = nn.GroupNorm(4, out_channels)

        super(Conv2dGNReLU, self).__init__(conv, gn, gelu)


class Conv2dGN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        gn = nn.GroupNorm(4, out_channels)

        super(Conv2dGN, self).__init__(conv, gn)


if __name__ == '__main__':
    import torch

    # 定义输入张量的形状
    input_shape = (1, 3, 32, 32)  # (batch_size, channels, height, width)

    # 创建输入张量
    input_tensor = torch.randn(input_shape)

    # 创建引导张量
    guidance_tensor = torch.randn((1, 3, 32, 32))  # 假设引导张量与输入张量大小相同

    # 创建模型
    model = SDM(in_channel=3, guidance_channels=3)

    # 将模型设置为评估模式
    model.eval()

    # 打印输入张量的形状
    print("输入张量1的形状:", input_tensor.shape)
    print("输入张量2的形状:", guidance_tensor.shape)

    # 执行前向传播
    output_tensor = model(input_tensor, guidance_tensor)

    # 打印输出张量的形状
    print("输出张量的形状:", output_tensor.shape)
