from torch import nn

from MSCAA import MSCAAttention
from GHPA import Grouped_multi_axis_Hadamard_Product_Attention

#串联
class chuanlian(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model1 = MSCAAttention(in_channels=in_channels)
        self.model2 = Grouped_multi_axis_Hadamard_Product_Attention(dim_in=in_channels, dim_out=out_channels)

    def forward(self, x):
        out1 = x

        out = self.model1(x)
        out = self.model2(out)

        outfinal = out + out1

        return outfinal

#并联
class binglian(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model1 = MSCAAttention(in_channels=in_channels)
        self.model2 = Grouped_multi_axis_Hadamard_Product_Attention(dim_in=in_channels, dim_out=out_channels)

    def forward(self, x):
        out1 = x
        out2 = x
        out1 = self.model1(out1)
        out2 = self.model2(out2)
        final = out1 + out2
        return final