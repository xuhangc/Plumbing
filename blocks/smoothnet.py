import torch
from torch import Tensor, nn
#SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos(ECCV2022)

class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.
    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (*, in_channels)
        Output: (*, in_channels)
    参数：

        in_channels（整数）：表示残差块的输入通道数。
        hidden_channels（整数）：指定残差块内部的隐藏特征通道数。
        dropout（浮点数）：表示在残差块内应用的 dropout 概率。默认值为 0.5。
    形状：
        输入：输入形状表示为 (*, in_channels)，表示输入可以具有任意数量的维度，其中最后一个维度的大小为 in_channels。
        输出：输出形状也表示为 (*, in_channels)，表示输出保持与输入相同的形状，最后一个维度仍然具有大小为 in_channels。
    """

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out

"""SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.
    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .
    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)
    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (N, C, T) the original pose sequence
        Output: (N, C, T) the smoothed pose sequence
    """
"""
这段代码定义了一个名为 SmoothNet 的类，它是一个仅针对时间维度的网络，用于优化人体姿势。以下是提供的信息的详细解释：

类说明：SmoothNet 是一个即插即用的网络，用于优化视频中的人体姿势。它适用于 2D/3D/6D 姿势平滑化。

论文引用：SmoothNet 的设计和原理可以在 arXiv 上的论文 "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos" 中找到。更多细节可以在该论文的链接中找到。

注意事项：在说明中提到了三个变量：

N: 批处理大小
T: 姿势序列的时间长度
C: 总的姿势维度（例如关键点数乘以关键点维度）
参数：

window_size (int): 输入窗口的大小。
output_size (int): 输出窗口的大小。
hidden_size (int): 编码器、解码器和残差块中的隐藏特征维度。默认为 512。
res_hidden_size (int): 残差块内部的隐藏特征维度。默认为 256。
num_blocks (int): 残差块的数量。默认为 3。
dropout (float): Dropout 概率。默认为 0.5。
形状：

输入：(N, C, T)，原始姿势序列，其中 N 为批处理大小，C 为姿势维度，T 为时间长度。
输出：(N, C, T)，平滑后的姿势序列。
初始化方法 (__init__)：在初始化方法中，构建了编码器、残差块和解码器的层，并初始化了参数。

前向传播方法 (forward)：在前向传播方法中，输入的姿势序列经过编码器、残差块和解码器的层，然后返回平滑后的姿势序列。

这个类实现了一个平滑处理人体姿势的神经网络，通过编码器-残差块-解码器的结构来实现。
"""
class SmoothNet(nn.Module):
    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        x=x.to(torch.float32)

        assert T == self.window_size, (
            'Input sequence length must be equal to the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, output_size]

        return x
if __name__ == '__main__':
    # 创建 SmoothNet 模型实例
    model = SmoothNet(window_size=10, output_size=10)

    # 生成测试输入数据
    batch_size = 4
    pose_dim = 12
    temporal_length = 10
    input_data = torch.randn(batch_size, pose_dim, temporal_length)

    # 打印输入数据的形状
    print("输入数据形状:", input_data.shape)

    # 进行前向传播
    output_data = model(input_data)

    # 打印输出数据的形状
    print("输出数据形状:", output_data.shape)