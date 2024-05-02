import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
"""
FeatUp(ICLR2024): 一个与任务和模型无关的框架，用于恢复深层特征中丢失的空间信息。
这段代码定义了两个下采样模块：`SimpleDownsampler` 和 `AttentionDownsampler`。

1. `SimpleDownsampler`：该模块使用固定的核进行下采样。它将输入图像与固定核进行卷积，将其大小缩小到所需的最终大小。

2. `AttentionDownsampler`：该模块通过关注不同的区域来对输入特征图进行下采样。它首先对输入特征图应用线性层，以计算每个补丁的注意力分数。然后，它对注意力分数应用 dropout，并使用可学习参数 `w` 和 `b` 进行调整。最后，它根据注意力分数计算输入补丁的加权和。

以下是每个模块中的主要组件和操作摘要：

### SimpleDownsampler：
- **初始化**：
  - `kernel_size`：卷积核的大小。
  - `final_size`：下采样后的期望最终大小。
  - `kernel_params`：卷积核的可学习参数。

- **前向操作**：
  - 重新塑造输入图像。
  - 根据输入大小和最终大小计算下采样的步幅。
  - 使用固定核进行二维卷积。
  - 将输出重新塑造为期望的最终大小。

### AttentionDownsampler：
- **初始化**：
  - `dim`：输入特征的维度。
  - `kernel_size`：卷积核的大小。
  - `final_size`：下采样后的期望最终大小。
  - `blur_attn`：布尔值，指示在注意力计算之前是否应用高斯模糊。
  - `attention_net`：用于计算注意力分数的线性层。
  - `w`、`b`：用于调整注意力分数的可学习参数。

- **前向注意力**：
  - 计算输入特征的注意力分数。
  
- **前向操作**：
  - 可选地对输入特征应用高斯模糊。
  - 根据输入大小和最终大小计算下采样的步幅。
  - 将输入特征展开成补丁。
  - 计算补丁的注意力分数并应用 dropout。
  - 使用可学习参数 `w` 和 `b` 调整注意力分数。
  - 根据注意力分数计算输入补丁的加权和。

这些模块可用于神经网络架构中对输入特征图进行下采样。
"""

class SimpleDownsampler(torch.nn.Module):

    def get_kernel(self):
        k = self.kernel_params.unsqueeze(0).unsqueeze(0).abs()
        k /= k.sum()
        return k

    def __init__(self, kernel_size, final_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.kernel_params = torch.nn.Parameter(torch.ones(kernel_size, kernel_size).cuda())

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        input_imgs = imgs.reshape(b * c, 1, h, w)
        stride = (h - self.kernel_size) // (self.final_size - 1)

        return F.conv2d(
            input_imgs,
            self.get_kernel(),
            stride=stride
        ).reshape(b, c, self.final_size, self.final_size)


class AttentionDownsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, final_size, blur_attn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.in_dim = dim
        self.attention_net = torch.nn.Sequential(
            torch.nn.Dropout(p=.2),
            torch.nn.Linear(self.in_dim, 1)
        )
        self.w = torch.nn.Parameter(torch.ones(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.b = torch.nn.Parameter(torch.zeros(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.blur_attn = blur_attn

    def forward_attention(self, feats):
        return self.attention_net(feats.permute(0, 2, 3, 1)).squeeze(-1).unsqueeze(1)

    def forward(self, hr_feats):
        b, c, h, w = hr_feats.shape

        if self.blur_attn:
            inputs = gaussian_blur2d(hr_feats, 5, (1.0, 1.0))
        else:
            inputs = hr_feats

        stride = (h - self.kernel_size) // (self.final_size - 1)

        patches = torch.nn.Unfold(self.kernel_size, stride=stride)(inputs) \
            .reshape(
            (b, self.in_dim, self.kernel_size * self.kernel_size, self.final_size, self.final_size * int(w / h))) \
            .permute(0, 3, 4, 2, 1)

        patch_logits = self.attention_net(patches).squeeze(-1)

        b, h, w, p = patch_logits.shape
        dropout = torch.rand(b, h, w, 1, device=patch_logits.device) > 0.2

        w = self.w.flatten().reshape(1, 1, 1, -1)
        b = self.b.flatten().reshape(1, 1, 1, -1)

        patch_attn_logits = (patch_logits * dropout) * w + b
        patch_attention = F.softmax(patch_attn_logits, dim=-1)

        downsampled = torch.einsum("bhwpc,bhwp->bchw", patches, patch_attention)

        return downsampled[:, :c, :, :]
