import torch
from torch import nn
from torch.nn import init


from einops.einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        x = to_4d(x, 28, 28)
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        x = to_3d(x)
        return x




class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()


        self.pa = Partial_conv3(128, 2, 'split_cat')



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):  # torch.Size([32, 784, 128])

        queries = self.pa(queries)

        attn = self.mk(queries)  # torch.Size([32, 784, 8])

        attn = self.softmax(attn)  # torch.Size([32, 784, 8])
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # torch.Size([32, 784, 8])
        out = self.mv(attn)         # torch.Size([32, 784, 128])

        return out


# class Paex(nn.Module):  # 串联
#     def __init__(self):
#         super(Paex, self).__init__()
#         self.pa = Partial_conv3(128, 2, 'split_cat')
#         self.ex = ExternalAttention(d_model=128, S=8)
#
#     def forward(self, x):
#         x1 = self.pa(x)
#         x2 = self.ex(x1)
#         return x2
#


# class Paex(nn.Module):  # 并联
#     def __init__(self):
#         super(Paex, self).__init__()
#         self.pa = Partial_conv3(128, 2, 'split_cat')
#         self.ex = ExternalAttention(d_model=128, S=8)
#
#     def forward(self, x):
#         x1 = self.pa(x)
#         x2 = self.ex(x)
#         x3 = x1 + x2
#         return x3
#
#
# 输入 B C N,  输出 B C N
# if __name__ == '__main__':
#     block = Paex()
#     input = torch.rand(32, 784, 128)
#     output = block(input)
#     print(input.size())
#     print(output.size())

# 输入 B C N,  输出 B C N
if __name__ == '__main__':
    block = ExternalAttention(d_model=128, S=8)
    input = torch.rand(32, 784, 128)
    output = block(input)
    print(input.size())
    print(output.size())

