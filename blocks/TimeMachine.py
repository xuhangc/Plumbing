from mamba_ssm import Mamba
import torch
import torch.nn as nn

"""
由于捕获长期依赖性、实现线性可扩展性和保持计算效率的困难，长期时间序列预测仍然具有挑战性。
推出了TimeMachine，这是一种创新模型，它利用 Mamba（一种状态空间模型）来捕获多元时间序列数据中的长期依赖性，同时保持线性可扩展性和较小的内存占用。 
TimeMachine 利用时间序列数据的独特属性来产生多尺度的显着上下文线索，并利用创新的集成四曼巴架构来统一对通道混合和通道独立情况的处理，从而能够有效地选择内容进行预测不同尺度的全球和地方背景。
在实验上，TimeMachine 在预测准确性、可扩展性和内存效率方面实现了卓越的性能，
"""


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

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


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        if self.configs.revin == 1:
            self.revin_layer = RevIN(self.configs.enc_in)

        self.lin1 = torch.nn.Linear(self.configs.seq_len, self.configs.n1)
        self.dropout1 = torch.nn.Dropout(self.configs.dropout)

        self.lin2 = torch.nn.Linear(self.configs.n1, self.configs.n2)
        self.dropout2 = torch.nn.Dropout(self.configs.dropout)
        if self.configs.ch_ind == 1:
            self.d_model_param1 = 1
            self.d_model_param2 = 1

        else:
            self.d_model_param1 = self.configs.n2
            self.d_model_param2 = self.configs.n1

        self.mamba1 = Mamba(d_model=self.d_model_param1, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                            expand=self.configs.e_fact)
        self.mamba2 = Mamba(d_model=self.configs.n2, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                            expand=self.configs.e_fact)
        self.mamba3 = Mamba(d_model=self.configs.n1, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                            expand=self.configs.e_fact)
        self.mamba4 = Mamba(d_model=self.d_model_param2, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                            expand=self.configs.e_fact)

        self.lin3 = torch.nn.Linear(self.configs.n2, self.configs.n1)
        self.lin4 = torch.nn.Linear(2 * self.configs.n1, self.configs.pred_len)

    def forward(self, x):
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = torch.permute(x, (0, 2, 1))
        if self.configs.ch_ind == 1:
            x = torch.reshape(x, (x.shape[0] * x.shape[1], 1, x.shape[2]))

        x = self.lin1(x)
        x_res1 = x
        x = self.dropout1(x)
        x3 = self.mamba3(x)
        if self.configs.ch_ind == 1:
            x4 = torch.permute(x, (0, 2, 1))
        else:
            x4 = x
        x4 = self.mamba4(x4)
        if self.configs.ch_ind == 1:
            x4 = torch.permute(x4, (0, 2, 1))

        x4 = x4 + x3

        x = self.lin2(x)
        x_res2 = x
        x = self.dropout2(x)

        if self.configs.ch_ind == 1:
            x1 = torch.permute(x, (0, 2, 1))
        else:
            x1 = x
        x1 = self.mamba1(x1)
        if self.configs.ch_ind == 1:
            x1 = torch.permute(x1, (0, 2, 1))

        x2 = self.mamba2(x)

        if self.configs.residual == 1:
            x = x1 + x_res2 + x2
        else:
            x = x1 + x2

        x = self.lin3(x)
        if self.configs.residual == 1:
            x = x + x_res1

        x = torch.cat([x, x4], dim=2)
        x = self.lin4(x)
        if self.configs.ch_ind == 1:
            x = torch.reshape(x, (-1, self.configs.enc_in, self.configs.pred_len))

        x = torch.permute(x, (0, 2, 1))
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'denorm')
        else:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return x

if __name__ == '__main__':
    # 假设的配置类
    class MockConfigs:
        def __init__(self):
            self.revin = 1
            self.enc_in = 10  # 假设输入特征的数量为 10
            self.seq_len = 30  # 假设序列长度为 30
            self.n1 = 64  # 第一个线性层的输出特征数
            self.n2 = 128  # 第二个线性层的输出特征数
            self.dropout = 0.1  # dropout 概率
            self.ch_ind = 1  # 假设的通道指数参数
            self.d_state = 32  # Mamba 参数
            self.dconv = 3  # 确保是在2到4之间
            self.e_fact = 2  # Mamba 参数
            self.pred_len = 30  # 假设预测长度为 5
            self.residual = 1  # 添加这行，启用残差连接


    # 实例化配置
    configs = MockConfigs()

    model = Model(configs)

    # 创建模拟输入张量
    # 假设批大小为 1，序列长度为 30，输入特征为 10
    input_tensor = torch.rand(1, configs.seq_len, configs.enc_in)

    # 将模型和张量移到 CUDA 上（如果可用）
    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()

    # 通过模型前向传播
    output = model(input_tensor)

    print("输入大小:", input_tensor.size())
    print("输出大小:", output.size())
