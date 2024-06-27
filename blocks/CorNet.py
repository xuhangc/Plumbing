import torch
import torch.nn as nn
import torch.nn.functional as F

"""
开发了用于极端多标签文本分类 （XMTC） 任务的相关网络 （CorNet） 架构，其目标是使用来自超大标签集的最相关标签子集标记输入文本序列。
XMTC 可以在许多实际应用中找到，例如文档标记和产品注释。最近，深度学习模型在XMTC任务中取得了出色的表现。然而，这些深度XMTC模型忽略了不同标签之间的有用关联信息。
CorNet 通过在深度模型的预测层添加一个额外的 CorNet模块 来解决这一限制，该模块能够学习标签相关性，利用相关性知识增强原始标签预测并输出增强标签预测。
我们表明，CorNet可以很容易地与深度XMTC模型集成，并在不同的数据集上有效地泛化。我们进一步证明，CorNet可以在性能和收敛率方面为现有的深度XMTC模型带来显著的改进。
"""

ACT2FN = {'elu': F.elu, 'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}


class CorNetBlock(nn.Module):
    def __init__(self, context_size, output_size, cornet_act='sigmoid', **kwargs):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = ACT2FN[cornet_act]

    def forward(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn


class CorNet(nn.Module):
    def __init__(self, output_size, cornet_dim=100, n_cornet_blocks=2, **kwargs):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [CorNetBlock(cornet_dim, output_size, **kwargs) for _ in range(n_cornet_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits


if __name__ == '__main__':
    output_size = 10
    cornet_dim = 100
    n_cornet_blocks = 2
    cornet_act = 'relu'

    model = CorNet(output_size=output_size, cornet_dim=cornet_dim, n_cornet_blocks=n_cornet_blocks)

    input_tensor = torch.rand(4, output_size)

    output = model(input_tensor)

    # 打印输入和输出的尺寸
    print("Input size :", input_tensor.size())
    print("Output size:", output.size())