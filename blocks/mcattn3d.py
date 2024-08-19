import numpy as np
from typing import Any, Callable
import torch
from torch import nn, Tensor
from typing import Optional
#https://arxiv.org/pdf/2407.07720v1
#SvANet: A Scale-variant Attention-based Network for Small Medical Object Segmentation
def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)

def callMethod(self, ElementName):
    return getattr(self, ElementName)

def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    if isinstance(Feature, Tensor):
        Feature = [Feature]

    Indexs = None
    Output = []
    for f in Feature:
        B, C, D, H, W = f.shape
        if Mode == 1:
            f = f.flatten(4)
            if Indexs is None:
                Indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, :, Indexs.to(f.device)]
            f = f.reshape(B, C, D, H, W)
        else:
            if Indexs is None:
                Indexs = [torch.randperm(D, device=f.device),
                          torch.randperm(H, device=f.device),
                          torch.randperm(W, device=f.device)]
            f = f[:, :, Indexs[0].to(f.device)]
            f = f[:, :, :, Indexs[1].to(f.device)]
            f = f[:, :, :, :, Indexs[2].to(f.device)]
        Output.append(f)
    return Output

class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool3d, self).__init__(output_size=output_size)

class BaseConv3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            groups: Optional[int] = 1,
            bias: Optional[bool] = None,
            BNorm: bool = False,
            ActLayer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            Momentum: Optional[float] = 0.1,
            **kwargs: Any
    ) -> None:
        super(BaseConv3d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)

        if bias is None:
            bias = not BNorm

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.Conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)

        self.Bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()

        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

class MCAttn(nn.Module):
    def __init__(
            self,
            InChannels: int,
            HidChannels: int = None,
            SqueezeFactor: int = 4,
            PoolRes: list = [1, 2, 3],
            Act: Callable[..., nn.Module] = nn.ReLU,
            ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
            MoCOrder: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)

        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool3d(k)
            setMethod(self, 'Pool%d' % k, Pooling)

        self.SELayer = nn.Sequential(
            BaseConv3d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv3d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )

        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder

    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(4)
                AttnMap = AttnMap[:, :, :, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, :, :, None]  # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)

        # Adjust shape to match x
        AttnMap = nn.functional.interpolate(
            AttnMap, size=x.shape[2:], mode='trilinear', align_corners=False
        )
        return AttnMap

    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        return x * self.SELayer(AttnMap)

if __name__ == '__main__':
    input = torch.randn(1, 64, 16, 128, 128)  # Adjusted for 3D tensor shape
    model = MCAttn(InChannels=64)
    output = model(input)
    print(output.shape)
