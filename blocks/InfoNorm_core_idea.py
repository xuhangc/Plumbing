import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# [ECCV 2024] InfoNorm: Mutual Information Shaping of Normals for Sparse-View Reconstruction.
# https://arxiv.org/abs/2407.12661
# 自定义线性层，支持批量参数梯度
class MyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    # 前向传播，调用父类的forward方法
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    # 批量前向传播方法
    def batch_forward(self, input: Tensor) -> Tensor:
        bs = input.shape[0]
        # 扩展weight矩阵以适应批量输入
        self.weight_matrix = self.weight[None, :, :].expand(bs, self.out_features, self.in_features)
        # 使用einsum进行矩阵乘法并加上偏置
        output = torch.add(torch.einsum('ab,abc->ac', input, self.weight_matrix.transpose(1, 2)), self.bias)
        return output


# 一个小型的MLP模型
class TinySDFMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TinySDFMLP, self).__init__()
        # 第一层线性层
        self.fc1 = nn.Linear(in_dim, 256)
        # 使用自定义的线性层
        self.sdf_linear = MyLinear(256, out_dim)

    def forward(self, x):
        # 使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 调用自定义线性层的批量前向传播方法
        x = self.sdf_linear.batch_forward(x)
        return x

    # 计算SDF值
    def sdf(self, x):
        return self.forward(x)

    # 计算法线
    def normal(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        normals = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True
        )[0]
        return normals

    # 计算法线对权重的梯度
    def dnormal_dw(self, x):
        normal = self.normal(x)
        dnormal_dw_list = []
        for i in range(normal.shape[1]):
            normal_slice = normal[:, i:i + 1]
            d_output_slice = torch.ones_like(normal_slice, requires_grad=False, device=normal.device)
            dnormal_dw_slice = torch.autograd.grad(
                outputs=normal_slice,
                inputs=self.sdf_linear.weight_matrix,
                grad_outputs=d_output_slice,
                create_graph=True,
            )[0]
            dnormal_dw_list.append(dnormal_dw_slice)
        return torch.stack(dnormal_dw_list, dim=1)


# 互信息损失函数
def mi_loss(pos_condition, feature):
    # 计算特征的余弦相似度
    feature = feature.reshape(feature.shape[0], -1)
    feature_dot = (feature @ torch.transpose(feature, 0, 1))
    feature_norm = torch.linalg.norm(feature, dim=-1)
    feature_norm_square = feature_norm * (feature_norm.unsqueeze(1)) + 1e-7
    mi_square = torch.exp(torch.abs(feature_dot / feature_norm_square))

    # 计算正样本和负样本的相似度总和
    pos_sim_total = (pos_condition.float() * mi_square).sum(dim=-1)
    neg_sim_total = ((~pos_condition).float() * mi_square).sum(dim=-1)
    contrastive = torch.log(pos_sim_total / (pos_sim_total + neg_sim_total) + 1e-7)

    # 计算损失
    contrastive_no_nan_count = torch.sum(~torch.isnan(contrastive))
    mi_contrastive_loss = -torch.nansum(contrastive) / contrastive_no_nan_count

    return mi_contrastive_loss


if __name__ == "__main__":
    threshold = 0.9

    # 初始化模型
    SDFNet = TinySDFMLP(3, 1)

    # 生成随机点和特征
    pts = torch.randn(512, 256, 3)  # 512条光线，每条光线256个点，3D点
    feats = torch.randn(512, 256)  # 512个像素，每个像素256维特征
    dots = feats @ feats.t()  # 计算相似度矩阵

    # 计算相似度并确定正样本对
    norms = torch.linalg.norm(feats, axis=1)
    similarity = dots / (norms[:, None] * norms[None, :])
    positive = similarity > threshold  # 正样本对
    print(f'Positive Shape : ', positive.shape)

    # 计算法线对权重的梯度
    pts_flattern = pts.view(-1, 3)
    dnormal_dw = SDFNet.dnormal_dw(pts_flattern).reshape(pts.shape[0], pts.shape[1], -1)
    print(f'dnormal_dw Shape : ', dnormal_dw.shape)

    # 聚合特征
    feats_bar = torch.sum(dnormal_dw, dim=1)
    print(f'Pred Feats Shape : ', feats_bar.shape)

    # 计算互信息损失
    print(f'Mi loss term : {mi_loss(positive, feats)}')
