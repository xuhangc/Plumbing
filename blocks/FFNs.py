from torch import nn
#这是修改后的FFN层，这下不需要x_size这个参数了

class SeparableConv1d(nn.Module):#深度可分离卷积
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            SeparableConv1d(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        B, N, C = x.shape  # x: (B, num_patches, embed_dim)
        x = x.permute(0, 2, 1)  # Convert to (B, embed_dim, num_patches) for Conv1d
        x = self.net(x)
        x = x.permute(0, 2, 1)  # Convert back to (B, num_patches, embed_dim)
        return x