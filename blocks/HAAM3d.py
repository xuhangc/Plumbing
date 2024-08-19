import torch
import torch.nn as nn

def expend_as(tensor, rep):
    return tensor.unsqueeze(2).repeat(1, 1, rep, 1, 1, 1)

class Channelblock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channelblock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=3, dilation=3)
        self.batch1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(out_channels * 2, out_channels)
        self.batch3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv3 = nn.Conv3d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, padding=0)
        self.batch4 = nn.BatchNorm3d(out_channels)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)

        conv2 = self.conv2(x)
        batch2 = self.batch2(conv2)
        relu2 = self.relu2(batch2)

        combined = torch.cat([relu1, relu2], dim=1)
        pooled = self.global_avg_pool(combined)
        pooled = pooled.view(pooled.size(0), -1)  # flatten the tensor

        fc1 = self.fc1(pooled)
        batch3 = self.batch3(fc1)
        relu3 = self.relu3(batch3)
        fc2 = self.fc2(relu3)
        sigm = self.sigmoid(fc2)

        a = sigm.view(-1, sigm.size(1), 1, 1, 1)
        a1 = 1 - sigm
        a1 = a1.view(-1, a1.size(1), 1, 1, 1)

        y = relu1 * a
        y1 = relu2 * a1

        combined = torch.cat([y, y1], dim=1)

        conv3 = self.conv3(combined)
        batch4 = self.batch4(conv3)
        relu4 = self.relu4(batch4)

        return relu4

class Spatialblock3D(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Spatialblock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=size, padding=(size//2))
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x, channel_data):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)

        conv2 = self.conv2(relu1)
        batch2 = self.batch2(conv2)
        spatial_data = self.relu2(batch2)

        combined = channel_data + spatial_data
        combined = nn.ReLU()(combined)

        return combined

class HAAM3D(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(HAAM3D, self).__init__()
        self.channel_block = Channelblock3D(in_channels, out_channels)
        self.spatial_block = Spatialblock3D(out_channels, out_channels, size)

    def forward(self, x):
        channel_data = self.channel_block(x)
        haam_data = self.spatial_block(x, channel_data)
        return haam_data

# 示例
if __name__ == '__main__':
    # 创建示例输入张量
    batch_size = 2
    in_channels = 64  # 输入通道数
    depth, height, width = 64, 224, 224  # 输入图像的深度、高度和宽度
    input_tensor = torch.randn(batch_size, in_channels, depth, height, width).cuda()

    # 实例化 HAAM 模型
    out_channels = 64  # 输出通道数
    haam_model_3d = HAAM3D(in_channels, out_channels).cuda()

    # 前向传播
    output_tensor = haam_model_3d(input_tensor).cuda()

    # 打印输入输出的形状
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)