import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops import rearrange
from tqdm import tqdm
# 系统相关的库
import math
import os
import urllib.request
from zipfile import ZipFile
from transformers import AutoTokenizer
from utils import RMSNorm
from torchsummary import summary

torch.autograd.set_detect_anomaly(True)

# 配置标识和超参数
USE_MAMBA =1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM =0
# 设定所用设备
device = torch.device('cuda'if torch.cuda.is_available() else'cpu')

d_model =8 # 即embedding长度
state_size =128  # 状态大小
seq_len =100  # 序列长度
batch_size =256  # 批次大小
last_batch_size =81  # 最后一个批次大小
current_batch_size = batch_size
different_batch_size =False
h_new =None
temp_buffer =None

#ssm核心模块
class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()
        # 一系列线性变换
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)
        # 设定一些超参数
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # 定义内部参数h和y
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)

        # 参数初始化

    # 离散化函数

    def discretization(self):
        # 离散化函数定义介绍在Mamba论文中的28页
        # dA = torch.matrix_exp(A * delta)
        # matrix_exp() only supports square matrix

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB


    # 前行传播
    def forward(self, x):
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))
        # 离散化
        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            # 如果不使用'h_new'，将触发本地允许错误
            global current_batch_size
            current_batch_size = x.shape[0]
            if self.h.shape[0] != current_batch_size:
                different_batch_size = True
                # 缩放h的维度匹配当前的批次
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                               "b l d -> b l d 1") * self.dB
            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

                # 改变y的维度
                self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

                # 基于h_new更新h的信息
                global temp_buffer
                temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
                return self.y
        else:
            # 将会触发错误
            # 设置h的维度
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # 设置y的维度
            y = torch.einsum('bln,bldn->bld', self.C, h)
            return y


class MambaBlock(nn.Module):

    def __init__(self, seq_len, d_model, state_size, device):
       super(MambaBlock, self).__init__()


       self.inp_proj = nn.Linear(d_model, 2*d_model, device=device)
       self.out_proj = nn.Linear(2*d_model, d_model, device=device)


       # 残差连接
       self.D = nn.Linear(d_model, 2*d_model, device=device)


       # 设置偏差属性
       self.out_proj.bias._no_weight_decay =True


       # 初始化偏差
       nn.init.constant_(self.out_proj.bias, 1.0)
                                      # 初始化S6模块
       self.S6 = S6(seq_len, 2*d_model, state_size, device)


       # 添加1D卷积
       self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)


       # 添加线性层
       self.conv_linear = nn.Linear(2*d_model, 2*d_model, device=device)


       # 正则化
       self.norm = RMSNorm(d_model, device=device)
                    # 前向传播
    def forward(self, x):
        # 参考Mamba论文中的图3
        x =self.norm(x)


        x_proj =self.inp_proj(x)


        # 1D卷积操作
        x_conv =self.conv(x_proj)
        x_conv_act = F.silu(x_conv) # Swish激活


        # 线性操作
        x_conv_out =self.conv_linear(x_conv_act)
                                     # S6模块操作
        x_ssm =self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish激活


        # 残差连接
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out =self.out_proj(x_combined)


        return x_out


# vim源代码里使用两个block交替实现vimencoder，参考论文图片算法和源代码，直接手撸vimencoder，这样仅需要12个vimencoder，
# 原图中残差这条线由block类实现，本模块只输出上半路
class VimEncoder(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
       super(VimEncoder, self).__init__()

       self.norm = RMSNorm(d_model, device=device)

       #投射层,一个用于投射x，一个用于投射z
       self.x_in_proj = nn.Linear(in_features=d_model, out_features=2 * d_model, device=device)
       self.z_in_proj = nn.Linear(in_features=d_model, out_features=2 * d_model, device=device)

       #一维卷积层
       self.for_conv = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=3, padding=1, device=device)
       self.back_conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)

       # 添加线性层
       self.for_conv_linear = nn.Linear(2 * d_model, 2 * d_model, device=device)
       self.back_conv_linear = nn.Linear(2 * d_model, 2 * d_model, device=device)

       #ssm模块
       self.forward_ssm = S6(seq_len, 2 * d_model, state_size, device)
       self.backward_ssm = S6(seq_len, 2 * d_model, state_size, device)

       self.out_proj = nn.Linear(2 * d_model, d_model, device=device)

       self.out_proj.bias._no_weight_decay = True

       # 初始化偏差
       nn.init.constant_(self.out_proj.bias, 1.0)

    def forward(self, x):
        x = self.norm(x)

        #先进行上半部
        #线性投射
        x_proj = self.x_in_proj(x)
        x_forw = x_proj
        x_back = x_proj.flip([1])

        #卷积操作
        x_for_conv = self.for_conv(x_forw)
        x_for_act = F.silu(x_for_conv)

        x_back_conv = self.back_conv(x_back)
        x_back_act = F.silu(x_back_conv)

        #线性操作和ssm
        x_for_conv_out = self.for_conv_linear(x_for_act)
        x_back_conv_out = self.for_conv_linear(x_back_act)

        x_for_ssm = F.silu(self.forward_ssm(x_for_conv_out))
        x_back_ssm = F.silu(self.backward_ssm(x_back_conv_out))

        #再执行下半部
        z = F.silu(self.z_in_proj(x))
        x_z_for = x_for_ssm * z
        x_z_back = x_back_ssm * z
        x_combined = x_z_for + x_z_back
        x_out = self.out_proj(x_combined)

        return x_out










# 定义Mamba模型
class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, device)


    def forward(self, x):
        x =self.mamba_block1(x)
        x =self.mamba_block2(x)
        x =self.mamba_block3(x)
        return x




if __name__ == "__main__":
    test_output = torch.rand(1, 16, 768).to('cuda')
    vim_enc = VimEncoder(seq_len=16,d_model=768,device='cuda',state_size=8)
    out_put = vim_enc(test_output)
    summary(vim_enc, input_size=(1, 224, 224), batch_size=1, device='cuda')
