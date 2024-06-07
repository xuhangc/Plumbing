import torch
import torch.nn as nn
from torch.nn import functional as F

class SpatialGCN(nn.Module):
    def __init__(self, in_channels):
        super(SpatialGCN, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 2
        self.theta = nn.Conv2d(in_channels, self.channels, kernel_size=1)
        self.nu = nn.Conv2d(in_channels, self.channels, kernel_size=1)
        self.xi = nn.Conv2d(in_channels, self.channels, kernel_size=1)
        self.final_conv = nn.Conv2d(self.channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        theta = self.theta(x).view(b, -1, self.channels)
        nu = F.softmax(self.nu(x).view(b, -1, self.channels), dim=0)
        xi = F.softmax(self.xi(x).view(b, -1, self.channels), dim=0)
        F_s = torch.matmul(nu.transpose(1, 2), xi)
        AF_s = torch.matmul(theta, F_s)
        AF_s = AF_s.reshape(b, self.channels, h, w)
        F_sGCN = self.final_conv(AF_s)
        return F_sGCN + x
    
class ChannelGCN(nn.Module):
    def __init__(self, in_channels):
        super(ChannelGCN, self).__init__()
        self.in_channels = in_channels
        self.C = in_channels // 2
        self.N = in_channels // 4
        self.zeta = nn.Conv2d(in_channels, self.C, kernel_size=1)
        self.kappa = nn.Conv2d(in_channels, self.N, kernel_size=1)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.C, self.C, kernel_size=1)
        )
        self.final_conv = nn.Conv2d(self.N, in_channels, kernel_size=1)

    def forward(self, x):
        # b, c, h, w = x.size()
        # zeta = self.zeta(x).view(b, self.C, h * w)
        # kappa = self.kappa(x).view(b, self.N, h * w).transpose(1, 2)
        # print(kappa.shape, zeta.shape)
        # F_c = torch.matmul(kappa, zeta)
        # F_c = F.softmax(F_c.view(b, self.C, self.N), dim=2)
        # F_c = self.middle_conv(F_c.unsqueeze(2)).squeeze(2)
        # F_c = torch.matmul(zeta.transpose(1, 2), F_c)
        # F_cGCN = self.final_conv(F_c.view(b, h, w, self.N))
        # return F_cGCN + x
        b, c, h, w = x.size()
        # Adjusting reshapes and operations to match b, c, h, w format
        zeta = self.zeta(x).view(b, h * w, self.C).permute(0, 2, 1)
        kappa = self.kappa(x).view(b, h * w, self.N)
        F_c = torch.matmul(zeta, kappa)
        F_c = F.softmax(F_c, dim=0)
        F_c = self.middle_conv(F_c.unsqueeze(2)).squeeze(2)

        # Reshape and permute zeta for compatibility
        F_c = torch.matmul(F_c.permute(0, 2, 1), zeta)

        # Reshape back to the original spatial dimensions
        F_c = F_c.permute(0, 2, 1).contiguous().view(b, self.N, h, w)
        F_cGCN = self.final_conv(F_c)

        # Ensure the output matches the original tensor's spatial dimensions
        return F_cGCN + x
    
class BasicUnit(nn.Module):
    def __init__(self, channels):
        super(BasicUnit, self).__init__()
        self.spatial_gcn = SpatialGCN(channels)
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3)
        ])
        self.concat_conv = nn.Conv2d(channels * 5, channels, kernel_size=1)
        self.channel_gcn = ChannelGCN(channels)

    def forward(self, x):
        F_sGCN = self.spatial_gcn(x)
        features = [F_sGCN]
        for conv in self.convs:
            features.append(F.relu(conv(F_sGCN)))
        tmp = torch.cat(features, dim=1)
        F_DCM = F.relu(self.concat_conv(tmp))
        F_cGCN = self.channel_gcn(F_DCM)
        return F_cGCN + x
    
class Inference(nn.Module):
    def __init__(self, channels=72):
        super(Inference, self).__init__()
        self.channels = channels
        self.basic_block = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.encoder_blocks = nn.ModuleList([BasicUnit(channels) for _ in range(5)])
        self.middle_block = BasicUnit(channels)
        self.decoder_blocks = nn.ModuleList([BasicUnit(channels * 2 * (i + 1)) for i in range(5)])
        self.reconstruct = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        basic_fea1 = self.basic_block(x)
        encode_outputs = [self.encoder_blocks[i](basic_fea1) for i in range(5)]
        middle_layer = self.middle_block(encode_outputs[-1])
        decoder_inputs = [(middle_layer, encode_outputs[-1])]
        decoder_inputs.extend(zip(encode_outputs[:-1][::-1], decoder_inputs[:-1]))
        decoder_outputs = [self.decoder_blocks[i](torch.cat(inputs, dim=1)) for i, inputs in enumerate(decoder_inputs)]
        decoding_end = torch.cat([decoder_outputs[0], basic_fea1], dim=1)
        decoding_end = self.reconstruct(decoding_end)
        output = x + decoding_end
        return output
    

if __name__ == "__main__":
    model = Inference(channels=72)
    # model = BasicUnit(16)
    input_x = torch.randn(1, 3, 256, 256)
    output = model(input_x)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters' number: %d" % total_params)