import torch
import torch.nn as nn
#RevIN: Reversible Instance Normalization For Accurate Time-series Forecasting Against Distribution Shift(ICLR2022)
#https://openreview.net/pdf?id=cGDAkQo1C0p

# 定义一个名为RevIN的自定义归一化层类继承自nn.Module
class RevIN(nn.Module):
    # 类初始化函数
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        构造函数初始化该层的参数

        :param num_features: 该层处理的特征或通道的数量
        :param eps: 为了数值稳定性而加上的小数值
        :param affine: 如果为True，则RevIN层将具有可学习的仿射参数
        """
        super(RevIN, self).__init__()  # 调用父类nn.Module的构造函数
        self.num_features = num_features  # 设置通道数
        self.eps = eps  # 设置数值稳定参数
        self.affine = affine  # 设置是否有仿射参数
        if self.affine:
            self._init_params()  # 如果使用仿射参数，则初始化这些参数

    # 定义前向传播函数
    def forward(self, x, mode: str):
        # 根据mode参数选择进行规范化或者反规范化操作
        if mode == 'norm':
            self._get_statistics(x)  # 计算统计量，即平均值和标准偏差
            x = self._normalize(x)  # 规范化操作
        elif mode == 'denorm':
            x = self._denormalize(x)  # 反规范化操作
        else:
            raise NotImplementedError  # 如果模式不是norm或denorm，则不执行
        return x  # 返回处理后的x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


if __name__ == '__main__':
    # 创建一个张量x，维度为(4, 3, 2)，并进行归一化处理
    x = torch.reshape(torch.arange(0, 24), shape=(4, 3, 2)) / 24  # 创建张量并做初步归一化
    layer = RevIN(2)  # 创建RevIN层的实例，假定处理的特征数量为2
    x_in = layer(x, mode='norm')  # 使用RevIN层进行规范化
    #x_in = blocks(x_in) # your model or subnetwork within the model
    x_out = layer(x_in, mode='denorm')  # 使用RevIN层进行反规范化

    # 打印原始张量x、规范化后的张量y、反规范化后的张量z的形状
    # 期望它们的形状保持不变
    print(x.shape)  # 输出原始张量x的形状
    print(x_in.shape)  # 输出规范化后的张量y的形状
    print(x_out.shape)  # 输出反规范化后的张量z的形状
"""
from RevIN import RevIN
revin_layer = RevIN(num_features)
x_in = revin_layer(x_in, 'norm')
x_out = blocks(x_in) # your model or subnetwork within the model
x_out = revin_layer(x_out, 'denorm')
"""