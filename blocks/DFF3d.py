import torch
import torch.nn as nn
#D-Net: Dynamic Large Kernel with Dynamic Feature Fusion for Volumetric Medical Image Segmentation
#https://arxiv.org/abs/2403.10674
"""
__init__ 方法初始化了在前向传播中使用的层。
self.avg_pool 执行自适应平均池化，将空间维度减小到 1x1x1。
self.conv_atten 是一个卷积层，后面跟着一个 sigmoid 激活函数。它根据输入的拼接（x 和 skip）计算注意力权重，并降低它们的维度。这些注意力权重表示每个特征图的重要性。
self.conv_redu 基于计算出的注意力权重降低了拼接输入的维度。
self.conv1 和 self.conv2 是卷积层，后面跟着一个 sigmoid 激活函数。它们独立地计算了来自 x 和 skip 的注意力权重。
self.nonlin 是一个 sigmoid 激活函数。
forward 方法接受两个输入，x 和 skip，并通过网络执行前向传播。它将 x 和 skip 进行拼接，使用全局和局部信息计算注意力权重，并将这些注意力权重应用于降维后的特征。
总的来说，这个模块通过注意力机制将全局和局部信息相结合，从而执行特征融合，这在需要捕捉局部细节和全局上下文都很重要的任务中非常有用。
"""
class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output
if __name__ == '__main__':
    # 定义测试输入
    x = torch.randn(1, 48, 128, 128, 128)
    skip = torch.randn(1, 48, 128, 128, 128)

    # 创建 DFF 模块
    model = DFF(48)

    # 前向传播
    output = model(x, skip)

    # 打印输入和输出的形状
    print("Input shape (x):", x.shape)
    print("Input shape (skip):", skip.shape)
    print("Output shape:", output.shape)