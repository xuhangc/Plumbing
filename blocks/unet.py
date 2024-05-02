import torch
import torch.nn as nn
import lightning as L
import torch.nn.init
from model.component import ModelSummaryMixin
from model.expo_transform import ResidualLayer


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, skip_input):
        # Input is CHW
        if skip_input is not None:
            x1 = torch.cat((skip_input, x1), dim=1)
        x1 = self.up(x1)
        return self.conv(x1)


class UNet(L.LightningModule, ModelSummaryMixin):
    """
    UNet
    in = 64
    out = 64
    """

    def __init__(self, bilinear=True, in_external=True, output_external=True, in_chans=32, out_chans=32):
        super(UNet, self).__init__()
        self.in_external = in_external
        self.output_external = output_external
        self.down1 = Down(in_chans, 64)  # in-[b,64,512,512] out-[b,128,256,256]
        self.down2 = Down(64, 128)  # in-[b,128,256,256] out-[b,256,128,128]
        self.down3 = Down(128, 256)  # in-[b,256,128,128] out-[b,512,64,64]
        self.down4 = Down(256, 512)

        # Decoder
        self.up1 = Up(
            512, 256, skip_channels=512, bilinear=bilinear
        )  # in-[b,2048x2,8,8] out-[b,1024,16,16]
        self.up2 = Up(
            256, 128, skip_channels=256, bilinear=bilinear
        )  # [16,16]->[32,32]
        self.up3 = Up(
            128, 64, skip_channels=128, bilinear=bilinear
        )  # [32,32]->[64,64]
        # self.upsample =
        self.up4 = Up(64, out_chans, skip_channels=64, bilinear=bilinear)  # [64,64]->[128,128]
        # self.up5 = Up(64, 64, skip_channels=64, bilinear=bilinear)  # [128,128]->[256,256]

    def forward(self, x, ex_feat=None):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.merger(x4)
        if self.in_external:
            _x1 = self.up1(x, x4)
            _x1 = _x1 + ex_feat[0]
            _x2 = self.up2(_x1, x3)
            _x2 = _x2 + ex_feat[1]
            _x3 = self.up3(_x2, x2)
            _x3 = _x3 + ex_feat[2]
            _x4 = self.up4(_x3, x1)
            _x4 = _x4 + ex_feat[3]
        else:
            _x1 = self.up1(x, x4)
            _x2 = self.up2(_x1, x3)
            _x3 = self.up3(_x2, x2)
            _x4 = self.up4(_x3, x1)

        if self.output_external:
            return [_x1, _x2, _x3, _x4]
        else:
            return _x4

class UNetEncoder(L.LightningModule, ModelSummaryMixin):
    """
    UNet
    in = 64
    out = 64
    """

    def __init__(self, bilinear=True, in_chans=32, out_chans=32):
        super(UNetEncoder, self).__init__()
        self.down1 = Down(in_chans, 64)  # in-[b,64,512,512] out-[b,128,256,256]
        self.down2 = Down(64, 128)  # in-[b,128,256,256] out-[b,256,128,128]
        self.down3 = Down(128, 256)  # in-[b,256,128,128] out-[b,512,64,64]
        self.down4 = Down(256, 512)

    def forward(self, x, ex_feat=None):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x1, x2, x3, x4

class UNetDecoder(L.LightningModule, ModelSummaryMixin):
    """
    UNet
    in = 64
    out = 64
    """
    def __init__(self,
                 bilinear=True,
                 in_external=True,
                 output_external=True,
                 in_chans=32,
                 out_chans=32,
                 external_gate=False,
                 merger='res'):
        super(UNetDecoder, self).__init__()
        self.in_external = in_external
        self.output_external = output_external
        self.external_gate = external_gate
        if merger == 'res':
            self.merger = nn.Sequential(
                ResidualLayer(512, 512, 3, 1),
                ResidualLayer(512, 512, 3, 1),
                ResidualLayer(512, 512, 3, 1),
                # ResidualLayer(512, 512, 3, 1),
            )
        elif merger == '1x1':
            self.merger = nn.Conv2d(512, 512, 1)
        # Decoder
        self.up1 = Up(
            512, 256, skip_channels=512, bilinear=bilinear
        )  # in-[b,2048x2,8,8] out-[b,1024,16,16]
        self.up2 = Up(
            256, 128, skip_channels=256, bilinear=bilinear
        )  # [16,16]->[32,32]
        self.up3 = Up(
            128, 64, skip_channels=128, bilinear=bilinear
        )  # [32,32]->[64,64]
        # self.upsample =
        self.up4 = Up(64, out_chans, skip_channels=64, bilinear=bilinear)  # [64,64]->[128,128]
        # self.up5 = Up(64, 64, skip_channels=64, bilinear=bilinear)  # [128,128]->[256,256]

        if external_gate:
            self.gate_conv1 = nn.Conv2d(512, 256, 3, 1, 1)
            self.gate_conv2 = nn.Conv2d(256, 128, 3, 1, 1)
            self.gate_conv3 = nn.Conv2d(128, 64, 3, 1, 1)
            self.gate_conv4 = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, x1, x2, x3, x4, ex_feat=None):
        x = self.merger(x4)
        # ----------External Input(feature inject)------------
        if self.in_external:
            if self.external_gate:
                _x1 = self.up1(x, x4)
                # print(f'{_x1.shape}, {ex_feat[0].shape}')
                gate1 = self.gate_conv1(torch.cat((_x1, ex_feat[0]), 1))
                _x1 = _x1 + ex_feat[0] * gate1
                _x2 = self.up2(_x1, x3)
                gate2 = self.gate_conv2(torch.cat((_x2, ex_feat[1]), 1))
                _x2 = _x2 + ex_feat[1] * gate2
                _x3 = self.up3(_x2, x2)
                gate3 = self.gate_conv3(torch.cat((_x3, ex_feat[2]), 1))
                _x3 = _x3 + ex_feat[2] * gate3
                _x4 = self.up4(_x3, x1)
                gate4 = self.gate_conv4(torch.cat((_x4, ex_feat[3]), 1))
                _x4 = _x4 + ex_feat[3] * gate4
            else:
                _x1 = self.up1(x, x4)
                _x1 = _x1 + ex_feat[0]
                _x2 = self.up2(_x1, x3)
                _x2 = _x2 + ex_feat[1]
                _x3 = self.up3(_x2, x2)
                _x3 = _x3 + ex_feat[2]
                _x4 = self.up4(_x3, x1)
                _x4 = _x4 + ex_feat[3]
        # ----------External Input(feature inject)------------
        else:
            _x1 = self.up1(x, x4)
            _x2 = self.up2(_x1, x3)
            _x3 = self.up3(_x2, x2)
            _x4 = self.up4(_x3, x1)

        # ------------ External Output(layer features)---------
        if self.output_external:
            # ---------Tanh activation-------
            # _x1 = torch.sigmoid(_x1)
            # _x2 = torch.tanh(_x2)
            # _x3 = torch.tanh(_x3)
            # _x4 = torch.tanh(_x4)
            # ---------Tanh activation-------
            return [_x1, _x2, _x3, _x4]
        # ------------ External Output(layer features)---------
        else:
            return _x4

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module, ModelSummaryMixin):
    def __init__(self, num_classes=128, input_channels=64, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], 512, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 256, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 128, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 64, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

if __name__ == '__main__':
    unetpp = NestedUNet(input_channels=128, num_classes=64)
    unetpp.print_summary(input_size=(1,128,128,128))