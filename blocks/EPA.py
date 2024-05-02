import torch.nn as nn
import torch


"""
由于Transformer模型的成功，最近的工作研究了它们在三维医疗分割任务中的适用性。在 transformer 模型中，自注意力机制是努力捕获长期依赖关系的主要构建块之一。
然而，自注意力操作具有二次复杂度，这被证明是一个计算瓶颈，尤其是在体积医学成像中，其中输入是带有大量切片的 3D 的。
在本文中，我们提出了一种高效的3D医学图像分割方法，该方法既能提供高质量的分割掩码，又能在参数、计算成本和推理速度方面提高效率。
我们设计的核心是引入一种新型的高效配对注意力 （EPA） 模块，该模块使用一对基于空间和通道注意力的相互依赖的分支有效地学习空间和通道方面的判别特征。
我们的空间注意力公式是有效的，具有相对于输入序列长度的线性复杂性。为了实现空间和以通道为中心的分支之间的通信，我们共享了查询和键映射函数的权重，这些函数提供了互补的好处（配对注意力），同时也减少了整体网络参数。
"""


class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module to project keys and values from HWD-dimension to P-dimension
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape
        #print("The shape in EPA ", self.E.shape)

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

if __name__ == '__main__':
    # 直接在创建实例时提供必需的参数  input_size与N对应，hidden_size与C对应
    block = EPA(input_size=32, hidden_size=128, proj_size=16)
    input = torch.rand(4, 32, 128)  # (B,N,C)
    output = block(input)
    print(input.size())
    print(output.size())