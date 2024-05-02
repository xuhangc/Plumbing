import torch.nn as nn
import torch.nn.functional as F
#PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion
#3D图像分割即插即用模块
"""
这段代码定义了几个用于三维卷积神经网络（CNN）的类，特别专注于体积数据的语义分割任务。让我们逐个组件来了解：

1. **SDC（选择性核卷积）**：
   - 这个类实现了一层执行选择性核卷积的操作，增强了卷积操作的适应性。它接受两个输入：输入张量 `x` 和引导张量 `guidance`。选择性核卷积是通过根据引导信息调节卷积核来实现的。
   - 参数：
     - `in_channels`：输入通道数。
     - `guidance_channels`：引导张量中的通道数。
     - `kernel_size`：卷积核的大小。
     - `stride`、`padding`、`dilation`、`groups`、`bias`：卷积操作的参数。
     - `theta`：一个超参数，控制根据引导张量进行调节的程度。
   - `forward` 方法通过根据引导张量调节卷积核来应用选择性核卷积。

2. **SDM（选择性扩散模块）**：
   - 这个类定义了一个模块，使用 SDC 层来增强特征表示。它接受两个输入：特征张量和引导张量。它首先应用 SDC 操作，然后将增强的边界添加到原始特征中。
   - 参数：
     - `in_channel`：输入通道数。
     - `guidance_channels`：引导张量中的通道数。
   - `forward` 方法应用 SDC 操作，然后将增强的边界添加到原始特征中。

3. **Conv3dReLU、Conv3dbn、Conv3dGNReLU、Conv3dGN**：
   - 这些类定义了具有不同操作组合的卷积块，包括卷积、批归一化和激活函数（如 ReLU 或 GELU）。这些块在 SDC 和 SDM 模块中用于特征提取和处理。

总的来说，这些类提供了一个框架，用于构建具有选择性核卷积和选择性扩散的三维 CNN 架构，专门用于语义分割任务，特别是通过选择性核卷积和选择性扩散来增强特征表示。
"""


class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        # self.conv1 = Conv3dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 0, 2].detach()
        self.x_kernel_diff[:, :, 0, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2, 2].detach()
        self.x_kernel_diff[:, :, 2, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0, 0] = -1

        kernel[:, :, 0, 0, 2] = 1
        kernel[:, :, 0, 2, 0] = 1
        kernel[:, :, 2, 0, 0] = 1

        kernel[:, :, 0, 2, 2] = -1
        kernel[:, :, 2, 0, 2] = -1
        kernel[:, :, 2, 2, 0] = -1

        kernel[:, :, 2, 2, 2] = 1

        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        guidance = self.conv1(guidance)

        x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out


class SDM(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDM, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature

        return boundary_enhanced


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)


class Conv3dGNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGNReLU, self).__init__(conv, gn, gelu)


class Conv3dGN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGN, self).__init__(conv, gn)
if __name__ == '__main__':
    import torch

    # 定义输入张量的形状
    input_shape = (1, 3, 32, 32, 32)  # (batch_size, channels, depth, height, width)

    # 创建输入张量
    input_tensor = torch.randn(input_shape)

    # 创建引导张量
    guidance_tensor = torch.randn((1, 2, 32, 32, 32))  # 假设引导张量与输入张量大小相同

    # 创建模型
    model = SDM(in_channel=3, guidance_channels=2)

    # 将模型设置为评估模式
    model.eval()

    # 打印输入张量的形状
    print("输入张量的形状:", input_tensor.shape)

    # 执行前向传播
    output_tensor = model(input_tensor, guidance_tensor)

    # 打印输出张量的形状
    print("输出张量的形状:", output_tensor.shape)
