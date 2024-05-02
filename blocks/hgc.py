import torch.nn as nn
import torch

"""
名为Hourglass Convolution (HgC) 的新型时间卷积操作，用以解决视频帧之间大位移带来的视觉线索错位问题。这种卷积操作的特点在于其时间接收域具有沙漏形状，能够在前后时间帧中扩大空间接收域，从而捕获大的位移变化。
此外，为了有效捕捉视频中的长短期动态变化，HgC网络被设计为分层结构，同时采取了如低分辨率处理短期模型和通道减少长期模型等策略以提高效率。
"""

class fhgc(nn.Module):
    def __init__(self, channel, reduction=16, n_segment=8, ME_init=True, chan_expend=1):
        super(fhgc, self).__init__()
        self.channel = channel
        self.n_segment = n_segment

        # self.spaAgg =  self.pool = nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=1)
        self.spaAgg = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=3,
            padding=2,
            groups=channel,
            dilation=2,
            bias=False)
        nn.init.xavier_normal_(self.spaAgg.weight)

        self.convt = nn.Conv3d(
            in_channels=self.channel,
            out_channels=self.channel * chan_expend,
            kernel_size=(3, 1, 1),
            padding=0,
            stride=(2, 1, 1),
            groups=channel,
            bias=False
        )
        if ME_init == True:
            self.convt.weight.requires_grad = True
            self.convt.weight.data.zero_()
            self.convt.weight.data[:, 0, 2, :, :] = -1  # shift left
            self.convt.weight.data[:, 0, 0, :, :] = 0  # shift right
            self.convt.weight.data[:, 0, 1, :, :] = 1  # fixed
        else:
            nn.init.kaiming_normal_(self.convt.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        nt, c, h, w = x.size()

        reshape_x = x.view((-1, self.n_segment) + x.size()[1:])  # n, t, c//r, h, w
        reshape_x = reshape_x.unsqueeze(2)
        # t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        agg_x = self.spaAgg(x)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_agg_x = agg_x.view((-1, self.n_segment) + agg_x.size()[1:])
        reshape_agg_x = reshape_agg_x.unsqueeze(2)
        zeros = torch.zeros(nt // self.n_segment, 1, 1, *agg_x.size()[1:]).cuda()
        target = torch.cat((reshape_agg_x[:, :-2, :, :, :], reshape_x[:, 1:-1, :, :, :], reshape_agg_x[:, 2:, :, :, :]),
                           dim=2)
        target = target.view((-1, 3 * (self.n_segment - 2)) + agg_x.size()[1:])

        target = self.convt(target.transpose(1, 2))

        target = target.transpose(1, 2).contiguous()
        target = target.view(-1, *target.shape[2:])
        return target


if __name__ == '__main__':

    # 创建 fhgc 实例，指定通道数为64。
    model = fhgc(channel=64)

    # 生成随机输入张量
    # 形状为 (batch_size * n_segment, channels, height, width)
    # 其中：
    # - batch_size * n_segment = 2 * 8，表示总的数据量，这里假设每个批次有2个样本，共有8个时间段，因此总共有16个独立数据单元。
    # - channels = 64，与模型创建时指定的通道数一致。
    # - height = 32，表示每个图像数据的高度。
    # - width = 32，表示每个图像数据的宽度。
    input_tensor = torch.rand(2 * 8, 64, 32, 32)

    # 对模型执行前向传播，传入之前生成的随机输入张量。
    output_tensor = model(input_tensor)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output tensor shape: {output_tensor.shape}")
