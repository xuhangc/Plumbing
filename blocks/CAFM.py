import torch
import torch.nn as nn
import torch.nn.functional as F

"""
这个CAFM（Cross Attention Fusion Module）是一个特征融合模块，设计用于深度学习中的图像处理任务，特别是那些需要融合来自两个不同输入源的特征信息的任务。
通过使用交叉注意力机制和卷积运算，该模块能够有效地结合两组特征图（f1和f2），以增强模型对关键信息的捕获能力。
以下是该模块主要步骤的详细解释：

初始化卷积层：定义了几个卷积层，既用于空间信息的提取（conv1_spatial和conv2_spatial），也用于特征降维和升维（avg1、avg2、max1、max2以及对应的avg11、avg22、max11、max22）。

特征转换：输入的特征图f1和f2首先被reshape成三维张量（batch, channel, -1），以便进行接下来的操作。

特征池化和激活：对每组特征分别进行平均池化和最大池化，然后通过卷积层对池化后的特征进行降维、ReLU激活和升维，生成两组加权特征a1和a2。

交叉注意力机制：利用加权特征a1和a2，通过矩阵乘法和softmax函数计算交叉注意力，这一步骤使模块能够捕捉两个输入之间的相互关系。

特征融合：将计算得到的注意力权重分别应用于原始特征f1和f2，然后将加权特征与原始特征相加，得到加权融合后的特征。

空间注意力处理：对融合后的特征进行空间注意力处理，包括通过卷积层提取空间特征，再次应用softmax函数获取空间注意力权重，并通过这些权重进一步精细化特征表示。

自定义融合和重塑：你添加的部分首先将两个特征图f1和f2进行转置和相加，以融合两组特征。然后，使用reshape将结果转换回原始的四维形状（1, 192, 256, 256），使其适用于后续的处理或层。

这个模块的设计理念体现了利用深度学习进行特征融合的先进策略，即通过注意力机制强化模型对关键信息的捕捉能力，并通过卷积层进行空间信息的提取和融合。这种方法在处理需要综合不同信息源的复杂场景（如多模态学习、图像融合等）时尤其有效。
"""

class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(128, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, 128, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        f1 = f1.squeeze(0)
        f2 = f2.squeeze(0)


        f = f1.transpose(0, 1) + f2.transpose(0, 1)  # add it myself

        f = f.reshape(1, 128, 256, 256)   # add it myself



        # return f1.transpose(0, 1), f2.transpose(0, 1)
        # return f1.transpose(0, 1) + f2.transpose(0, 1)

        return f

if __name__ == '__main__':
    block = CAFM()
    input1 = torch.rand(1, 128, 256, 256)  # 模拟输入特征图1
    input2 = torch.rand(1, 128, 256, 256)  # 模拟输入特征图2
    output = block(input1, input2)  # 获取两个输入的输出
    print(output.shape)


