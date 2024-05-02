import torch
import torch.nn as nn
from torch_scatter import scatter


""" 自注意力层的 Lipschitz 归一化及其在图神经网络中的应用
基于注意力的神经网络在广泛的应用中是最先进的。但是，当层数增加时，它们的性能往往会下降。
在这项工作中，我们表明，通过归一化注意力分数来加强 Lipschitz 连续性可以显着提高深度注意力模型的性能。首先，我们表明，对于深度图注意力网络（GAT），在训练过程中会出现梯度爆炸，导致基于梯度的训练算法性能不佳。
为了解决这个问题，我们推导了注意力模块的 Lipschitz 连续性的理论分析，并引入了 LipschitzNorm，这是一种简单且无参数的自我注意力机制归一化，它强制模型是 Lipschitz 连续的。
然后，我们将 LipschitzNorm 应用于 GAT 和 Graph Transformers，并表明它们的性能在深度设置（10 到 30 层）中得到了显着改善。
更具体地说，我们表明，使用 LipschitzNorm 的深度 GAT 模型在表现出长期依赖性的节点标签预测任务中实现了最先进的结果，同时在基准节点分类任务中显示出与非规范化对应物的一致改进。
"""


class LipschitzNorm(nn.Module):
    def __init__(self, att_norm=4, recenter=False, scale_individually=True, eps=1e-12):
        super(LipschitzNorm, self).__init__()
        self.att_norm = att_norm
        self.eps = eps
        self.recenter = recenter
        self.scale_individually = scale_individually

    def forward(self, x, att, alpha, index):
        att_l, att_r = att

        if self.recenter:
            mean = scatter(src=x, index=index, dim=0, reduce='mean')
            x = x - mean.index_add(0, index, mean)  # 重新定位到原始形状

        norm_x = torch.norm(x, dim=-1, keepdim=True) ** 2
        max_norm = scatter(src=norm_x, index=index, dim=0, reduce='max')
        max_norm = torch.sqrt(max_norm.index_select(0, index) + norm_x)  # simulation of max_j ||x_j||^2 + ||x_i||^2

        if not self.scale_individually:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim=-1), dim=-1, keepdim=True)
        else:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim=-1), dim=-1, keepdim=True)

        alpha = alpha / (norm_att * max_norm + self.eps)
        return alpha


if __name__ == '__main__':
    N, D = 10, 5  # 例如，10个样本，每个样本5个特征
    x = torch.randn(N, D)
    att_l = torch.randn(N, D)  # 注意力权重
    att_r = torch.randn(N, D)  # 注意力权重
    alpha = torch.randn(N, D)  # 权重系数
    index = torch.arange(N)  # 索引，每个样本唯一标识

    block = LipschitzNorm()
    alpha_updated = block(x, (att_l, att_r), alpha, index)

    print(f'Input alpha size: {alpha.size()}')
    print(f'Updated alpha size: {alpha_updated.size()}')