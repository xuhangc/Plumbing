import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
#TSLANet: Rethinking Transformers for Time Series Representation Learning(ICML 2024)
#https://arxiv.org/pdf/2404.08472
"""
ICB（倒置卷积块）
目的：通过使用一维卷积来处理数据，它包含三个主要步骤。起始和结束使用点集卷积扩展和收缩通道维数，在中间使用核大小为3的卷积捕捉局部模式。
操作：
Conv1：将输入特征从 in_features 扩展到 hidden_features。
Act：应用 GELU 非线性激活函数。
Drop：可选择应用dropout进行正则化。
Conv2：使用三点卷积核处理扩展的特征，捕捉局部模式。
Conv3：将特征从 hidden_features 缩减回 in_features。
元素级操作：在最终缩减操作前，结合conv1和conv2的输出进行乘法和加法运算。

自适应频谱块（Adaptive_Spectral_Block）
目的：该模块旨在处理频域中的数据，根据频率分量的能量来自适应地调制或强化特定的频率分量。
操作：
FFT：首先将输入时间序列转换到频域。
自适应掩膜：计算每个频率分量的能量，并根据百分位阈值创建一个掩膜以识别主导频率。
复数权重：在应用自适应掩膜之前和之后，变为复数值的权重被应用在频域表示上，用以调节信号的频谱属性。
IFFT：将数据转换回时域。
TSLANet层
综合：这一层结合了自适应频谱块（ASB）和倒置卷积块（ICB），通过外部布尔标志（args.ICB 和 args.ASB）决定是否对每个块应用数据。
顺序处理：
Norm1：首先规范输入数据。
ASB/ICB 应用：根据标志，它要么：
通过ASB然后ICB来传递数据。
只通过其中一个块来处理数据。
如果不应用任何块，直接通过数据而不做改变。
Norm2 和 ICB：对于ICB处理，数据在通过ICB前再次规范化。
"""
class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = adaptive_filter

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x

if __name__ == '__main__':
    # 设定输入特性
    in_features = 64  # 输入特征的维度
    hidden_features = 128  # 隐藏特征的维度
    seq_length = 50  # 序列长度 (假设为时间序列的长度)
    batch_size = 10  # 批量大小

    # 实例化 ICB 和 Adaptive_Spectral_Block
    icb = ICB(in_features=in_features, hidden_features=hidden_features)
    asb = Adaptive_Spectral_Block(dim=in_features)

    # 创建测试数据
    x = torch.randn(batch_size, seq_length, in_features)

    # 打印输入维度
    print(f'Input shape: {x.shape}')

    # ICB前向传播并打印输出维度
    x_icb = icb(x)
    print(f'Output shape after ICB: {x_icb.shape}')

    # ASB前向传播并打印输出维度
    x_asb = asb(x)
    print(f'Output shape after ASB: {x_asb.shape}')
