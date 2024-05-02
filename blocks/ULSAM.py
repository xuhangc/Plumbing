import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.cuda.FloatTensor)
#ULSAM: Ultra-Lightweight Subspace Attention Module for Compact Convolutional Neural Networks(WACV20)
"""
SubSpace 是一个表示 ULSAM（具有多个子空间的分组注意力块）中的子空间的类。
这个类应用深度可分离卷积，然后是最大池化，接着是逐点卷积，最后是 softmax 操作。
"""
class SubSpace(nn.Module):
    """
    Subspace class.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int) -> None:
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out

"""
ULSAM 代表"具有多个子空间的分组注意力块"。
该类根据指定的分割数量（num_splits）初始化一组 SubSpace 实例。
它根据分割数量分割输入张量，并对每个分割应用子空间操作。
然后，它将所有子空间的输出连接起来。
"""
class ULSAM(nn.Module):
    """
    Grouped Attention Block having multiple (num_splits) Subspaces.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    nout : int
        number of output feature maps

    h : int
        height of a input feature map

    w : int
        width of a input feature map

    num_splits : int
        number of subspaces

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int, nout: int, h: int, w: int, num_splits: int) -> None:
        super(ULSAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out


if __name__ == '__main__':
    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 64, 112, 112)  # 输入张量的形状为 (batch_size, channels, height, width)

    # 实例化 ULSAM 类
    ulsam = ULSAM(32, 64, 112, 112, 4)

    # 打印输入张量的形状
    print("输入张量形状：", input_tensor.shape)

    # 将输入张量传递给 ULSAM 类的 forward 方法
    output_tensor = ulsam(input_tensor)

    # 打印输出张量的形状
    print("输出张量形状：", output_tensor.shape)