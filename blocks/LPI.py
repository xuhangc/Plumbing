from torch import nn
from torch.nn import Module
from einops.layers.torch import Rearrange
#XCiT: Cross-Covariance Image Transformers
#LayerScale(dim, LocalPatchInteraction(dim, local_patch_kernel_size), depth = layer)，这就是那个LPI模块，照着这个图来改代码
class LocalPatchInteraction_nchw(Module):
    def __init__(self, dim, kernel_size = 3):#这里的dim就是embeddim
        super().__init__()
        assert (kernel_size % 2) == 1
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            Rearrange('b c h w -> b h w c'),
        )

    def forward(self, x):
        return self.net(x)


class LocalPatchInteraction_bnc(nn.Module):#这里我直接提供一个bnc版本，之前的叫nchw版本,用这个版本就成功了，
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        assert (kernel_size % 2) == 1
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n 1'),
            nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=dim),
            Rearrange('b c n 1 -> b n c'),
        )

    def forward(self, x):
        return self.net(x)