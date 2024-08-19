import torch
import torch.nn as nn
#DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation(PRCV)
#https://arxiv.org/pdf/2212.11677
# 基本的3D卷积块
class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 用于上下文聚合的上下文块
class ContextBlock3d(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_mul', )):
        super(ContextBlock3d, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, '至少应使用一种融合方式'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, depth, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x.view(batch, channel, depth * height * width).unsqueeze(1)
            context_mask = self.conv_mask(x).view(batch, 1, depth * height * width)
            context_mask = self.softmax(context_mask).unsqueeze(-1)
            context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

# 带有空间注意力的卷积分支
class ConvBranch3d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1

# 全局到局部空间聚合（GLSA）
class GLSA3d(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.conv1_1 = BasicConv3d(embed_dim * 2, embed_dim, 1)
        self.conv1_1_1 = BasicConv3d(input_dim // 2, embed_dim, 1)
        self.local_11conv = nn.Conv3d(input_dim // 2, embed_dim, 1)
        self.global_11conv = nn.Conv3d(input_dim // 2, embed_dim, 1)
        self.GlobelBlock = ContextBlock3d(inplanes=embed_dim, ratio=2)
        self.local = ConvBranch3d(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        local = self.local(self.local_11conv(x_0))
        Globel = self.GlobelBlock(self.global_11conv(x_1))
        x = torch.cat([local, Globel], dim=1)
        x = self.conv1_1(x)
        return x


if __name__ == '__main__':
    # 测试代码
    input_tensor = torch.randn(1, 32, 16, 64, 64)
    glsa3d = GLSA3d(input_dim=32, embed_dim=32)
    output_tensor = glsa3d(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")
