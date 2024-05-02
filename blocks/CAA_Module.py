import torch
import torch.nn as nn

"""
CAA_Module是Channel-wise Affinity Attention模块，用于增强特征表示的通道关联性和语义表达能力。下面对该模块的功能进行解释：

输入与输出：

输入：尺寸为 (B, C, N) 的特征图，其中 B 是 batch size，C 是通道数，N 是特征点数。
输出：经过通道注意力增强的特征图，尺寸与输入相同。
结构组成：

query_conv：对输入特征图进行一维卷积，将通道数降低到输入通道数的 1/8，然后经过 Batch Normalization 和 ReLU 激活函数。
key_conv：与 query_conv 相同，用于计算通道之间的相似度。
value_conv：对输入特征图进行一维卷积，用于生成输出特征图的基础特征。
softmax：用于将相似度矩阵转换为权重矩阵，以便对每个通道进行加权平均。
alpha：可学习的参数，用于调节残差连接中新生成特征与原始特征的权重。
功能：

Compact Channel-wise Comparator block：通过将输入特征图转置，然后分别对转置后的特征图进行卷积操作，计算通道之间的相似度矩阵。
Channel Affinity Estimator block：利用相似度矩阵计算通道之间的亲和力矩阵，通过 softmax 函数将相似度转换为权重，以便对每个通道进行加权平均。
Residual connection：最后，将加权的特征与原始特征相加，以引入残差连接，并通过可学习的权重 alpha 控制两者的比例。
总体来说，CAA_Module旨在通过通道间的相似性计算和加权平均来增强特征表示的语义相关性，从而提高模型在任务中的性能表现。
"""

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
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
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1]))  # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2]))  # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st - org_x
    edge2 = neighbor_2nd - org_x
    normals = torch.cross(edge1, edge2, dim=1)  # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True)  # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True)  # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1)  # B,14,N

    return new_pts


def get_graph_feature(x, k=20, idx=None):
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
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""

    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(1024 // 8)
        self.bn2 = nn.BatchNorm1d(1024 // 8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024 // 8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024 // 8, kernel_size=1, bias=False),
                                      self.bn2,
                                      nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat) - similarity_mat
        affinity_mat = self.softmax(affinity_mat)

        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha * out + x
        return out

if __name__ == '__main__':
    block = CAA_Module(in_dim=64)  # 假设输入特征的通道数为 64
    input = torch.rand(2, 64, 1024)  # (batch_size, channels, num_points)
    output = block(input)
    print("输入大小:", input.size())
    print("输出大小:", output.size())

