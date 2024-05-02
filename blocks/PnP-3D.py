import torch
import torch.nn as nn
import torch.nn.functional as F

# PnP-3D: A Plug-and-Play for 3D Point Clouds   增强现有点云分析网络的有效性

"""
在深度学习范式的帮助下，许多点云网络已被发明用于视觉分析。然而，这些网络的发展潜力巨大，因为点云数据的给定信息尚未被充分利用。
为了提高现有网络在分析点云数据时的有效性，我们提出了一种即插即用的模块，PnP-3D，旨在通过涉及更多本地上下文和来自显式三维空间和隐式特征空间的全局双线性响应，优化基本的点云特征表示。
在三个标准的点云分析任务上进行实验，包括分类、语义分割和物体检测，我们选择了每个任务中的三个最先进的网络进行评估。作为即插即用的模块，PnP-3D可以显著提升已建立网络的性能。
"""


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_neighbors(x, feature, k=20, idx=None):
    '''
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    neighbor_x = torch.cat((neighbor_x - x, x), dim=3).permute(0, 3, 1, 2)

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2,
                                1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
    feature = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    neighbor_feat = torch.cat((neighbor_feat - feature, feature), dim=3).permute(0, 3, 1, 2)

    return neighbor_x, neighbor_feat


class Mish(nn.Module):
    '''new activation function'''

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx):
        ctx = ctx * (torch.tanh(F.softplus(ctx)))
        return ctx

    @staticmethod
    def backward(ctx, grad_output):
        input_grad = (torch.exp(ctx) * (4 * (ctx + 1) + 4 * torch.exp(2 * ctx) + torch.exp(3 * ctx) +
                                        torch.exp(ctx) * (4 * ctx + 6))) / (2 * torch.exp(ctx) + torch.exp(2 * ctx) + 2)
        return input_grad


class PnP3D(nn.Module):
    def __init__(self, input_features_dim):
        super(PnP3D, self).__init__()

        self.mish = Mish()

        self.conv_mlp1 = nn.Conv2d(6, input_features_dim // 2, 1)
        self.bn_mlp1 = nn.BatchNorm2d(input_features_dim // 2)

        self.conv_mlp2 = nn.Conv2d(input_features_dim * 2, input_features_dim // 2, 1)
        self.bn_mlp2 = nn.BatchNorm2d(input_features_dim // 2)

        self.conv_down1 = nn.Conv1d(input_features_dim, input_features_dim // 8, 1, bias=False)
        self.conv_down2 = nn.Conv1d(input_features_dim, input_features_dim // 8, 1, bias=False)

        self.conv_up = nn.Conv1d(input_features_dim // 8, input_features_dim, 1)
        self.bn_up = nn.BatchNorm1d(input_features_dim)

    def forward(self, xyz, features, k):
        # Local Context fusion
        neighbor_xyz, neighbor_feat = get_neighbors(xyz, features, k=k)

        neighbor_xyz = F.relu(self.bn_mlp1(self.conv_mlp1(neighbor_xyz)))  # B,C/2,N,k
        neighbor_feat = F.relu(self.bn_mlp2(self.conv_mlp2(neighbor_feat)))  # B,C/2,N,k

        f_encoding = torch.cat((neighbor_xyz, neighbor_feat), dim=1)  # B,C,N,k
        f_encoding = f_encoding.max(dim=-1, keepdim=False)[0]  # B,C,N

        # Global Bilinear Regularization
        f_encoding_1 = F.relu(self.conv_down1(f_encoding))  # B,C/8,N
        f_encoding_2 = F.relu(self.conv_down2(f_encoding))  # B,C/8,N

        f_encoding_channel = f_encoding_1.mean(dim=-1, keepdim=True)[0]  # B,C/8,1
        f_encoding_space = f_encoding_2.mean(dim=1, keepdim=True)[0]  # B,1,N
        final_encoding = torch.matmul(f_encoding_channel, f_encoding_space)  # B,C/8,N
        final_encoding = torch.sqrt(final_encoding + 1e-12)  # B,C/8,N
        final_encoding = final_encoding + f_encoding_1 + f_encoding_2  # B,C/8,N
        final_encoding = F.relu(self.bn_up(self.conv_up(final_encoding)))  # B,C,N

        f_out = f_encoding - final_encoding

        # Mish Activation
        f_out = self.mish(f_out)

        return f_out


if __name__ == "__main__":
    # 创建一个PnP3D模块，输入特征维度为64
    pnp3d_module = PnP3D(64).cuda()
    # 随机生成32个样本的三维坐标，每个样本有1024个点 [B,3,N]
    coords = torch.rand(32, 3, 1024).cuda()
    # 随机生成32个样本的特征，每个样本有1024个点，特征维度为64 [B,C,N]
    in_feat = torch.rand(32, 64, 1024).cuda()
    # 对PnP3D模块进行前向传播，传入坐标、特征和k=20的邻居数量
    out_feat = pnp3d_module(coords, in_feat, 20)
    # 打印输出特征的形状
    print(out_feat.shape)