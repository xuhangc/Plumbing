import torch
from torch import nn

"""
文本视频检索是一项极具实用价值的任务，受到越来越多的关注，其中学习时空视频表示是研究热点之一。
最先进的视频检索模型中的视频编码器通常直接采用网络结构固定的预训练视觉主干，因此无法进一步改进以产生细粒度的时空视频表示。
在本文中，我们提出了令牌移位和选择网络（TS2-Net），这是一种新颖的令牌移位和选择变压器架构，它动态调整令牌序列并从输入视频样本中选择时间和空间维度上的信息令牌。
令牌移位模块会临时在相邻帧之间来回移动整个令牌特征，以保留完整的令牌表示并捕获微妙的运动。然后标记选择模块选择对局部空间语义贡献最大的标记。
"""


class VisualTokenRandomSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=7):
        super().__init__()
        self.max_frames = max_frames
        self.topk = topk

    def forward(self, x, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''

        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D)  # shape here is (bs, max_frames, n_patches, hid_dim)
        x = x.reshape(-1, N, D)  # shape here is (bs*max_frames, n_patches, hid_dim)

        # cls token as cls token
        cls_x_feature = x[:, :1, :]  # cls_token, shape here is (bs*max_frames, 1, hid_dim)
        # # avg pool of all tokens as cls token
        # cls_x_feature = torch.mean(x, dim=1, keepdim=True)

        spatial_x_feature = x[:, 1:, :]  # seperate the cls_token, shape here is (bs*max_frames, n_patches-1, hid_dim)
        patch_len = spatial_x_feature.shape[1]
        selected_indices = torch.randperm(patch_len)[:self.topk].sort()[0]
        selected_patch_feature = spatial_x_feature[:, selected_indices, :]

        output = torch.cat((cls_x_feature, selected_patch_feature),
                           dim=1)  # shape here is (bs*max_frames, topkPatches, hid_dim)
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1, D)  # shape here is (B, max_frames*topkPatches, D)

        return output


if __name__ == '__main__':
    max_frames = 10
    embed_dim = 512
    topk = 7
    sequence_length = max_frames * 8  # Assuming there are 8 tokens per frame

    block = VisualTokenRandomSelection(max_frames=max_frames, embed_dim=embed_dim, topk=topk)

    input = torch.rand(10, sequence_length, embed_dim)   # (batch_size, sequence_length, embed_dim)

    output = block(input, training=True)

    print('Input size:', input.size())
    print('Output size:', output.size())