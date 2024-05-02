import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
gated axial-attention
1x1卷积（conv1x1）：这个函数定义了一个使用nn.Conv2d的1x1卷积层。
qkv_transform类：这个类旨在处理与自注意力机制中的查询（query）、键（key）和值（value）向量相关的变换。
AxialAttention类：
这个类定义了轴向注意力模块。
它继承自nn.Module。
它接受多个参数，如输入和输出维度、组数、核大小等。
在初始化方法（__init__）中，它设置了轴向注意力模块的各种组件：
qkv_transform：一个变换层（可能与查询、键和值的计算相关）。
用于归一化各种操作输出的批归一化层（bn_qkv、bn_similarity、bn_output）。
位置嵌入：这是作为可学习参数随机初始化的。
forward方法实现了轴向注意力模块的前向传播：
它重塑输入张量并执行变换（qkv_transform）。
计算查询、键和值张量。
计算位置嵌入并将其合并到注意力计算中。
使用softmax计算注意力分数。
将注意力分数应用于值张量以获得关注的值。
将关注的值与位置嵌入组合。
重塑并返回最终输出张量。
"""
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # print(x.shape)
        # pdb.set_trace()
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
if __name__ == '__main__':
    # Define input tensor
    input_tensor = torch.randn(1, 64, 224, 224)  # (batch_size, width, channels, height)

    # Initialize AxialAttention module
    axial_attn = AxialAttention(in_planes=64, out_planes=64, groups=1, kernel_size=224, stride=1, bias=False, width=False)
    #in_planes、out_planes和channel一致， kernel_size和h,w一致

    # Forward pass
    output_tensor = axial_attn(input_tensor)

    # Print input and output shapes
    print("Input shape:", input_tensor.shape)
    print("OUTput shape:", output_tensor.shape)

