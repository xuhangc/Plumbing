import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#LMFLOSS: A HYBRID LOSS FOR IMBALANCED MEDICAL IMAGE CLASSIFICATION
"""
这段代码定义了三种自定义损失函数：Focal Loss（聚焦损失）、
LDAM Loss（基于标签驱动的边界损失）和LMF Loss（标签驱动的边界和聚焦损失）。
这些损失函数在数据集不平衡或者需要在训练过程中专注于特定类别时非常有用。

以下是每个损失函数的简要概述：

Focal Loss（聚焦损失）：最初用于密集目标检测，聚焦损失有助于解决类别不平衡问题，
通过降低分类正确的样本的权重。它使训练集中的难例得到关注，防止简单样本主导训练过程。

LDAM Loss（基于标签驱动的边界损失）：该损失函数将类别感知的边界融入损失计算中。
它根据每个类别的样本数量动态调整边界。在类别高度不平衡时特别有用，因为它有助于更好地区分类别。

LMF Loss（标签驱动的边界和聚焦损失）：该损失函数结合了聚焦损失和LDAM损失。
它提供了两种损失的加权组合，使模型能够同时受益于聚焦损失的难例关注和LDAM损失的类别感知边界调整。

每个损失函数接受模型输出（output）和目标标签（target）作为输入。
forward 方法根据这些输入计算损失值，并返回总损失。

这些损失函数可以像其他内置损失函数（如交叉熵损失）一样，在 PyTorch 训练循环中使用。
您可以实例化这些损失函数，并在训练过程中与优化器一起使用，将模型预测和目标标签传递给计算梯度并相应地更新模型参数。
"""

class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1 - p) ** self.gamma * logp

        return torch.mean(focal_loss)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance.
        You can start with a small value and gradually increase it to observe the impact on the model's performance.
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.

        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem.
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies
        the impact of the logits and can be useful when dealing with highly imbalanced datasets.
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, weight, alpha=1, beta=1, gamma=2, max_m=0.5, s=30):
        super().__init__()
        self.focal_loss = FocalLoss(weight, gamma)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss