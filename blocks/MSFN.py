import torch
import torch.nn as nn
import torch.nn.functional as F
# Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (IEEE GRSL 2024)
# https://ieeexplore.ieee.org/document/10445289
# https://github.com/summitgao/HCANet
## Multi-Scale Feed-Forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2.66, bias = False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*3, kernel_size=(1,1,1), bias=bias)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)
        # self.dwconv2 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        # self.dwconv3 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)


        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1,1,1), bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        # x1 = self.dwconv1(x1)
        # x2 = self.dwconv2(x2)
        # x3 = self.dwconv3(x3)
        x = F.gelu(x1)*x2*x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        x = x.squeeze(2)
        return x


if __name__ == '__main__':
    # 假设输入张量的形状
    input_tensor = torch.randn(1, 64, 128, 128)  # [batch_size, channels, height, width]

    # 实例化并打印 MSFN 的输入输出
    msfn = FeedForward(dim=64)
    msfn_output = msfn(input_tensor)
    print("MSFN Input Shape: ", input_tensor.shape)
    print("MSFN Output Shape: ", msfn_output.shape)