import torch
import torch.nn as nn
from torch_geometric.nn import fps


"""
我们提出了一个用于 3D 点云分析的非参数网络 Point-NN，它由纯粹的不可学习组件组成：最远点采样 （FPS）、k 最近邻 （k-NN） 和池化运算，具有三角函数。令人惊讶的是，它在各种 3D 任务中表现良好，不需要参数或训练
甚至超过了现有的完全训练的模型。从这个基本的非参数模型开始，我们提出了两个扩展。首先，Point-NN 可以作为构建参数网络的基本架构框架，只需在顶部插入线性层即可。
其次，Point-NN可以看作是推理过程中已经训练好的3D模型的即插即用模块。Point-NN 捕获了互补的几何知识，并增强了不同 3D 基准测试的现有方法，而无需重新训练。
"""


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# # FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = fps(xyz.reshape(-1, 3), ratio=self.group_num / N, batch=None).long()  # 使用torch_geometric的fps
        fps_idx = fps_idx[:self.group_num]  # 选择前group_num个点作为采样结果

        # 适应batch处理的索引调整
        batch_indices = torch.arange(B, dtype=torch.long).unsqueeze(1).repeat(1, self.group_num).reshape(-1)
        fps_idx = fps_idx.unsqueeze(0).repeat(B, 1) + batch_indices * N
        fps_idx = fps_idx.reshape(-1)

        lc_xyz = xyz.reshape(-1, 3)[fps_idx].reshape(B, self.group_num, 3)
        lc_x = x.reshape(-1, x.size(-1))[fps_idx].reshape(B, self.group_num, x.size(-1))

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta

    def forward(self, xyz):
        B, _, N = xyz.shape
        device = xyz.device  # 获取xyz的设备
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().to(device)  # 确保在正确的设备上
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed



# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        device = knn_xyz.device  # 获取knn_xyz的设备
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().to(device)  # 确保在正确的设备上
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim).to(device)  # 确保dim_embed也在正确的设备上
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w



# Non-Parametric Encoder
class EncNP(nn.Module):
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.LGA_list = nn.ModuleList()  # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))

    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)
        return x


# Non-Parametric Network
class Point_NN(nn.Module):
    def __init__(self, input_points=1024, num_stages=4, embed_dim=72, k_neighbors=90, beta=1000, alpha=100):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta)

    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        x = self.EncNP(xyz, x)
        return x


if __name__ == '__main__':
    batch_size = 1
    num_features = 3  # 假设输入特征为点的x, y, z坐标
    input_points = 1024  # 与Point_NN类中定义的点数相匹配

    # 创建一个具有随机数据的输入张量
    input_tensor = torch.rand(batch_size, num_features, input_points)

    # 初始化Point_NN模型
    model = Point_NN(input_points=input_points, num_stages=4, embed_dim=72, k_neighbors=90, beta=1000, alpha=100)

    # 将输入张量传递给模型
    output_tensor = model(input_tensor)

    # 打印输入和输出张量的形状
    print(f"Input tensor shape: {input_tensor.size()}")
    print(f"Output tensor shape: {output_tensor.size()}")