import torch
import torch.nn as nn

"""
FITS模块用于时间序列预测，它通过频率插值的方法来进行预测。模型的设计思路不仅仅局限于时域分析，还涉及到频域的处理，这是一个较为先进和不常见的处理方式。

模块功能
频率插值：在频域对时间序列进行插值，目的是根据已有的序列数据预测未来的数据点。这种方法的一个优点是能够在保留时间序列主要频率成分的同时，有效地预测序列的未来走势。

低通滤波（LPF）：通过对时间序列进行快速傅里叶变换（FFT），然后对变换结果进行截断，实现低通滤波。这一步骤有助于去除高频噪声，保留对预测最重要的低频成分。

频率上采样：对低频成分进行上采样，扩展频率域中的数据点数量，以适应预测长度的增加。这是实现时间序列预测扩展的关键步骤。

逆傅里叶变换：最后，通过逆傅里叶变换将处理后的频率数据转换回时域，得到扩展后的时间序列数据。

FITS模块通过在频域进行时间序列的预测，提供了一种新颖的时间序列分析方法。与传统的时域分析方法相比，频域方法在处理周期性强、含有明显频率成分的时间序列数据时，可能会显示出更好的性能。
此外，通过频率插值和低通滤波，FITS能够在保留时间序列主要特征的同时，有效地预测未来的数据点，对于时间序列预测任务具有重要的应用价值。
"""

class Configs:
    def __init__(self):
        self.seq_len = 100  # 输入序列长度
        self.individual = False  # 是否独立处理每个频道
        self.enc_in = 2  # 输入通道数
        self.cut_freq = 50  # 截断频率，即考虑的最大频率

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.dominance_freq = configs.cut_freq
        self.n_fft = self.seq_len // 2 + 1  # FFT输出的大小

        # 注意：为了处理复数数据，我们的频率上采样层的输入和输出尺寸都翻倍
        if configs.individual:
            self.freq_upsampler = nn.ModuleList([nn.Linear(self.n_fft * 2, self.n_fft * 2, bias=False) for _ in range(self.channels)])
        else:
            self.freq_upsampler = nn.Linear(self.n_fft * 2 * self.channels, self.n_fft * 2 * self.channels, bias=False)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_normalized = (x - x_mean) / torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)

        # 执行FFT变换
        low_specx = torch.fft.rfft(x_normalized, dim=1)
        low_specx[:, self.dominance_freq:, :] = 0  # 应用LPF

        # 拆分实部和虚部
        real_part = low_specx.real
        imag_part = low_specx.imag

        # 将实部和虚部拼接在一起形成实数张量
        low_specx_combined = torch.cat([real_part, imag_part], dim=-1)

        # 应用全连接层之后，假设low_specxy_combined已经是正确的形状，其中包含了合并的实部和虚部
        # low_specxy_combined的形状应该是 (batch_size, self.seq_len // 2 + 1, 2 * self.channels)

        if isinstance(self.freq_upsampler, nn.ModuleList):
            low_specxy_combined = torch.stack([
                self.freq_upsampler[i](low_specx_combined[:, :, i].view(-1, 2 * self.n_fft))
                for i in range(self.channels)
            ], dim=-1).view(-1, self.n_fft, 2)
        else:
            low_specxy_combined = self.freq_upsampler(low_specx_combined.view(-1, self.n_fft * 2 * self.channels))
            # 确保low_specxy_combined回到期望的形状
            low_specxy_combined = low_specxy_combined.view(-1, self.n_fft, 2 * self.channels)

        # 分割实部和虚部，需要考虑channels维度
        real_part, imag_part = torch.split(low_specxy_combined, self.channels, dim=-1)

        # 将real_part和imag_part的形状调整为复数操作所需的形状
        real_part = real_part.view(-1, self.seq_len // 2 + 1, self.channels)
        imag_part = imag_part.view(-1, self.seq_len // 2 + 1, self.channels)

        # 重新组合为复数张量
        low_specxy_ = torch.complex(real_part, imag_part)

        # 接下来的代码保持不变
        low_xy = torch.fft.irfft(low_specxy_, n=self.seq_len, dim=1)
        xy = (low_xy * torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)) + x_mean

        return xy


if __name__ == '__main__':
    configs = Configs()
    model = Model(configs)
    input_tensor = torch.rand(32, configs.seq_len, configs.enc_in)
    output = model(input_tensor)
    print('Input size:', input_tensor.size())
    print('Output size:', output.size())
