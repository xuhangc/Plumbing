import torch
import torch.nn as nn

#简单的多尺度卷积+CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#一个简单的多尺度卷积
class MultiScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleModule, self).__init__()
        self.conv_1x1_init = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv_7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv_5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.conv_1x1_final1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

        self.CA = ChannelAttention(in_channels)

        self.SA = SpatialAttention()

        self.conv_1x1_final2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        #原始输入
        x_copy = x
        #1x1卷积
        x = self.conv_1x1_init(x)

        #并行卷积，不同的卷积核 + concat操作
        concatenated = torch.cat([self.conv_7x7(x), self.conv_5x5(x), self.conv_3x3(x)], dim=1)

        # 1x1卷积
        out = self.conv_1x1_final1(concatenated)

        # 残差连接
        out = out + x_copy

        # 原始输入
        out_copy = out

        # 并行注意力 + concat操作
        out = torch.cat([self.CA(out) * out, self.SA(out) * out], dim=1)

        # 1x1卷积
        out = self.conv_1x1_final2(out)

        # 残差连接
        return out + out_copy




if __name__ == '__main__':
    # Example usage
    input_tensor = torch.randn(1, 64, 128, 128)  # Example input tensor
    model = MultiScaleModule(in_channels=64, out_channels=64)#in_channels == out_channels
    print(model)
    output = model(input_tensor)
    print(output.shape)
