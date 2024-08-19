import torch.nn as nn
import torch

# https://arxiv.org/pdf/1409.4842
class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

        self.branch3x3_1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.branch3x3_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)

        self.branch5x5_1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.branch5x5_2 = nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=2, bias=False)

        self.branch_pool = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(4 * out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(in_channels=out_channel * 4, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        out = torch.cat(outputs, 1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv1x1(out)

        return out


if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64)
    model = InceptionBlock(in_channel=32, out_channel=32)
    output = model(input)
    print(output.shape)
