import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, requested=8):
    groups = min(int(channels), int(requested))
    while channels % groups:
        groups -= 1
    return groups


class DWResidualBlock(nn.Module):
    def __init__(self, channels, expansion=2, groups=8):
        super().__init__()
        hidden = int(channels * expansion)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels)
        self.norm = nn.GroupNorm(_group_count(channels, groups), channels)
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.project = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = F.gelu(x)
        x = self.project(x)
        return residual + x


class SharedFrameEncoder(nn.Module):
    def __init__(self, in_channels, channels, blocks):
        super().__init__()
        if len(channels) != 4 or len(blocks) != 4:
            raise ValueError('Temporal U-Net requires four channel and block stages')
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (width, depth) in enumerate(zip(channels, blocks)):
            self.stages.append(nn.Sequential(*[
                DWResidualBlock(width) for _ in range(int(depth))]))
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(
                    width, channels[index + 1], 3, stride=2, padding=1))

    def forward(self, history):
        if history.ndim != 5:
            raise ValueError('history must be [B,T,C,H,W]')
        batch, time, channels, height, width = history.shape
        x = history.reshape(batch * time, channels, height, width)
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x.reshape(batch, time, *x.shape[1:]))
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return features


class TemporalWeightedFusion(nn.Module):
    """Depthwise temporal filtering followed by per-pixel time weighting."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('temporal kernel size must be odd')
        self.temporal_conv = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0), groups=channels)
        self.attention = nn.Conv3d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.temporal_conv.weight)
        nn.init.zeros_(self.temporal_conv.bias)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError('temporal feature must be [B,T,C,H,W]')
        latest = x[:, -1]
        filtered = self.temporal_conv(x.permute(0, 2, 1, 3, 4))
        weights = torch.softmax(self.attention(filtered), dim=2)
        correction = (filtered * weights).sum(dim=2)
        return latest + correction


class TemporalFeaturePyramid(nn.Module):
    def __init__(self, in_channels, channels, blocks, mix_scales,
                 temporal_kernel=3):
        super().__init__()
        self.encoder = SharedFrameEncoder(in_channels, channels, blocks)
        self.mix_scales = tuple(int(index) for index in mix_scales)
        if any(index < 0 or index >= len(channels) for index in self.mix_scales):
            raise ValueError('temporal mix scale is outside the encoder pyramid')
        self.mixers = nn.ModuleDict({
            str(index): TemporalWeightedFusion(
                channels[index], kernel_size=temporal_kernel)
            for index in self.mix_scales
        })

    def forward(self, history):
        temporal_features = self.encoder(history)
        return [
            self.mixers[str(index)](feature)
            if index in self.mix_scales else feature[:, -1]
            for index, feature in enumerate(temporal_features)
        ]


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, fpn_channels):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(width, fpn_channels, kernel_size=1)
            for width in encoder_channels
        ])
        self.refine = nn.ModuleList([
            DWResidualBlock(fpn_channels) for _ in encoder_channels
        ])

    def forward(self, features):
        if len(features) != len(self.laterals):
            raise ValueError('feature pyramid depth does not match decoder')
        decoded = [None] * len(features)
        decoded[-1] = self.refine[-1](self.laterals[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            lateral = self.laterals[index](features[index])
            top_down = F.interpolate(
                decoded[index + 1], size=lateral.shape[-2:],
                mode='bilinear', align_corners=False)
            decoded[index] = self.refine[index](lateral + top_down)
        return {
            'fine': decoded[0],
            'middle': decoded[1],
            'coarse': decoded[2],
            'bottleneck': decoded[3],
        }


class UNetMotionHead(nn.Module):
    def __init__(self, fpn_channels, hidden_channels, forecast_steps):
        super().__init__()
        self.project = nn.Conv2d(2 * fpn_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            DWResidualBlock(hidden_channels),
            DWResidualBlock(hidden_channels),
        )
        self.output = nn.Conv2d(hidden_channels, forecast_steps * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, coarse, bottleneck):
        bottleneck = F.interpolate(
            bottleneck, size=coarse.shape[-2:], mode='bilinear',
            align_corners=False)
        x = self.project(torch.cat([coarse, bottleneck], dim=1))
        return self.output(self.blocks(x))


class UNetFactorizedSourceHead(nn.Module):
    """Predict growth/steady/decay and bounded source magnitudes."""

    def __init__(self, fpn_channels, source_channels=32, hidden_channels=64,
                 field_channels=1):
        super().__init__()
        self.source_projection = nn.Conv2d(
            fpn_channels, source_channels, kernel_size=1)
        input_channels = source_channels + field_channels + 2 + field_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(
                _group_count(hidden_channels, 8), hidden_channels),
            nn.SiLU(),
            DWResidualBlock(hidden_channels),
        )
        self.regime_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.growth_head = nn.Conv2d(
            hidden_channels, field_channels, kernel_size=1)
        self.decay_head = nn.Conv2d(
            hidden_channels, field_channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.regime_head.weight)
        nn.init.zeros_(self.growth_head.weight)
        nn.init.zeros_(self.decay_head.weight)
        with torch.no_grad():
            self.regime_head.bias.copy_(torch.tensor([0.0, 20.0, 0.0]))
        nn.init.constant_(self.growth_head.bias, -20.0)
        nn.init.constant_(self.decay_head.bias, -20.0)

    def encode(self, fine_feature):
        return self.source_projection(fine_feature)

    def forward(self, source_feature, advected_rain, flow,
                gradient_magnitude, max_rain, max_displacement):
        source_input = torch.cat([
            source_feature,
            advected_rain / max(float(max_rain), 1e-6),
            flow / max(float(max_displacement), 1e-6),
            gradient_magnitude / max(float(max_rain), 1e-6),
        ], dim=1)
        hidden = self.trunk(source_input)
        return {
            'regime_logits': self.regime_head(hidden),
            'growth_logit': self.growth_head(hidden),
            'decay_logit': self.decay_head(hidden),
        }
