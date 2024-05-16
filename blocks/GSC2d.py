import torch.nn as nn
import torch

class GSC2d(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm2d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm2d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


if __name__ == '__main__':
    # 实例化模型，这里的假设输入通道数为32
    model = GSC2d(in_channles=32)

    # 创建一个随机输入张量，形状为(batch_size, channels, height, width)
    input_tensor = torch.randn(1, 32, 64, 64)

    # 计算模型的前向传播
    output_tensor = model(input_tensor)

    # 打印输入和输出张量的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)