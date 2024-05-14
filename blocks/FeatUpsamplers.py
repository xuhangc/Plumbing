import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.save_for_backward(input, filters)
        output = F.conv2d(input, filters)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, filters = ctx.saved_tensors
        grad_input = grad_filters = None

        if ctx.needs_input_grad[0]:
            # 使用torch.nn.grad.conv2d_input获取输入的梯度
            grad_input = torch.nn.grad.conv2d_input(input.shape, filters, grad_output)

        if ctx.needs_input_grad[1]:
            # 使用torch.nn.grad.conv2d_weight获取权重的梯度
            grad_filters = torch.nn.grad.conv2d_weight(input, filters.shape, grad_output)

        return grad_input, grad_filters

class SimpleImplicitFeaturizer(torch.nn.Module):

    def __init__(self, n_freqs=20):
        super().__init__()
        self.n_freqs = n_freqs
        self.dim_multiplier = 2

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        feat_list = [feats]
        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1)
        feats = (feats * freqs)

        feats = feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

        all_feats = [torch.sin(feats), torch.cos(feats), original_image]

        return torch.cat(all_feats, dim=1)


class IFA(torch.nn.Module):

    def __init__(self, feat_dim, num_scales=20):
        super().__init__()
        self.scales = 2 * torch.exp(torch.tensor(torch.arange(1, num_scales + 1)))
        self.feat_dim = feat_dim
        self.sin_feats = SimpleImplicitFeaturizer()
        self.mlp = nn.Sequential(
            nn.Conv2d(feat_dim + (num_scales * 4) + 2, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(),
            nn.Conv2d(feat_dim, feat_dim, 1),
        )

    def forward(self, source, guidance):
        b, c, h, w = source.shape
        up_source = F.interpolate(source, (h * 2, w * 2), mode="nearest")
        assert h == w
        lr_cord = torch.linspace(0, h, steps=h, device=source.device)
        hr_cord = torch.linspace(0, h, steps=2 * h, device=source.device)
        lr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(lr_cord, lr_cord)], dim=0).unsqueeze(0)
        hr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(hr_cord, hr_cord)], dim=0).unsqueeze(0)
        up_lr_coords = F.interpolate(lr_coords, (h * 2, w * 2), mode="nearest")
        coord_diff = up_lr_coords - hr_coords
        coord_diff_feats = self.sin_feats(coord_diff)
        c2 = coord_diff_feats.shape[1]
        bcast_coord_feats = torch.broadcast_to(coord_diff_feats, (b, c2, h * 2, w * 2))
        return self.mlp(torch.cat([up_source, bcast_coord_feats], dim=1))  # + up_source




class CarafeUpsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from mmcv.ops import CARAFEPack
        self.up1 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)
        self.up2 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)
        self.up3 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)
        self.up4 = CARAFEPack(dim, up_kernel=3, up_group=1, scale_factor=2)

    def forward(self, source, guidance):
        source_2 = self.up1(source)
        source_4 = self.up2(source_2)
        source_8 = self.up3(source_4)
        source_16 = self.up4(source_8)
        return source_16


class LayeredResizeConv(torch.nn.Module):

    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")

    def apply_conv(self, source, guidance, conv, activation):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = activation(conv(torch.cat([big_source, small_guidance], dim=1)))
        return big_source + output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, F.relu)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, lambda x: x)
        return source_16


class JBULearnedRange(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self, device):
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel(source.device)
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        result =  AdaptiveConv.apply(hr_source_padded, combined_kernel)
        return result


class JBUStack(torch.nn.Module):

    def __init__(self, feat_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up1 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up2 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up3 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up4 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1))

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled

    def forward(self, source, guidance):
        source_2 = self.upsample(source, guidance, self.up1)
        source_4 = self.upsample(source_2, guidance, self.up2)
        source_8 = self.upsample(source_4, guidance, self.up3)
        source_16 = self.upsample(source_8, guidance, self.up4)
        return self.fixup_proj(source_16) * 0.1 + source_16


class Bilinear(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feats, img):
        _, _, h, w = img.shape
        return F.interpolate(feats, (h, w), mode="bilinear")


def get_upsampler(upsampler, dim):
    if upsampler == 'bilinear':
        return Bilinear()
    elif upsampler == 'jbu_stack':
        return JBUStack(dim)
    elif upsampler == 'resize_conv':
        return LayeredResizeConv(dim, 1)
    elif upsampler == 'carafe':
        return CarafeUpsampler(dim, 1)
    elif upsampler == 'ifa':
        return IFA(dim)
    else:
        raise ValueError(f"Unknown upsampler {upsampler}")