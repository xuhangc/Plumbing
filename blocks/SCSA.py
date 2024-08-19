import typing as t
import torch
import torch.nn as nn
from einops import rearrange
#SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
#https://arxiv.org/pdf/2407.05128
#Spatial and Channel Synergistic Attention module (SCSA)
"""
初始化：

参数：
dim：输入特征的维度。
head_num：注意力头的数量。
window_size：用于下采样的窗口大小。
group_kernel_sizes：不同组的内核大小列表。
qkv_bias：是否在Q、K、V投影中使用偏置。
fuse_bn：是否融合批量归一化。
norm_cfg：归一化配置。
act_cfg：激活函数配置。
down_sample_mode：下采样的方法（'avg_pool'、'max_pool'或'recombination'）。
attn_drop_ratio：注意力的丢弃率。
gate_layer：门控层的类型（'sigmoid'或'softmax'）。

属性：
dim、head_num、head_dim、scaler、group_kernel_sizes、window_size、qkv_bias、fuse_bn、down_sample_mode、group_chans。
用于本地和全局深度卷积的不同卷积层。
不同的归一化层（GroupNorm）。
用于Q、K和V投影的卷积层。
用于注意力的丢弃层。
门控层（Softmax或Sigmoid）。
基于window_size和down_sample_mode的下采样函数。

前向方法：

空间注意力：
通过沿宽度和高度维度对输入张量进行平均计算水平和垂直注意力图。
根据通道数量将张量分成组，并对每个组应用不同的深度卷积。
连接结果并应用门控函数以获得空间注意力图。
重塑并将输入张量与空间注意力图相乘。

通道注意力：
使用指定的下采样方法对输入张量进行下采样。
归一化下采样的张量并生成Q、K和V投影。
重塑并计算缩放点积注意力。
应用丢弃并将结果重塑为原始维度。
计算注意力图的平均值并应用门控函数以获得最终的通道注意力图。
将输入张量与通道注意力图相乘。
"""
class SCSA(nn.Module):

    def __init__(
            self,
            dim: int,
            head_num: int = 8,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='ReLU'),
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, '输入特征的维度应能被4整除。'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # 维度降维
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量的维度为 (B, C, H, W)
        """
        # 空间注意力优先级计算
        b, c, h_, w_ = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # 基于自注意力的通道注意力
        # 降低计算量
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W) 并生成 q, k 和 v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


if __name__ == '__main__':
    # 实例化 SCSA 模块
    dim = 64
    scsa = SCSA(dim=dim)

    # 打印模型结构
    # print(scsa)

    # 创建一个示例输入张量 (B, C, H, W)
    input_tensor = torch.randn(1, 64, 32, 32)  # batch size = 1, channels = 64, height = 32, width = 32

    # 计算输出
    output_tensor = scsa(input_tensor)

    # 打印输入和输出张量的形状
    print(f'输入张量的形状: {input_tensor.shape}')
    print(f'输出张量的形状: {output_tensor.shape}')
