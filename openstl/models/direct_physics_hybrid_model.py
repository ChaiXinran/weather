import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.models.convlstm_model import ConvLSTM_Model
from openstl.modules import EvolutionOperator
from openstl.modules.evolution_operator import (
    normalized_dbz_to_rain, rain_to_normalized_dbz)
from openstl.modules.temporal_unet_modules import (
    TemporalFeaturePyramid, FPNDecoder, DWResidualBlock)
from openstl.utils import reshape_patch, reshape_patch_back


class DirectPhysicsHybrid_Model(nn.Module):
    """Frozen direct ConvLSTM plus a zero-start motion/source correction."""

    def __init__(self, configs):
        super().__init__()
        hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.configs = configs
        self.direct_model = ConvLSTM_Model(len(hidden), hidden, configs)

        channels = list(configs.hybrid_unet_channels)
        self.features = TemporalFeaturePyramid(
            configs.in_shape[1], channels, configs.hybrid_unet_blocks,
            configs.hybrid_temporal_mix_scales,
            temporal_kernel=configs.hybrid_temporal_kernel,
            convlstm_scales=configs.hybrid_convlstm_scales,
            convlstm_kernel=configs.hybrid_convlstm_kernel)
        fpn_channels = int(configs.hybrid_fpn_channels)
        self.decoder = FPNDecoder(channels, fpn_channels)
        lead_channels = int(configs.hybrid_lead_channels)
        self.lead_embedding = nn.Embedding(configs.aft_seq_length, lead_channels)
        head_channels = int(configs.hybrid_head_channels)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels + configs.in_shape[1] + lead_channels,
                      head_channels, 3, padding=1),
            nn.GroupNorm(8, head_channels), nn.SiLU(),
            DWResidualBlock(head_channels))
        self.flow_head = nn.Conv2d(head_channels, 2, 1)
        self.source_head = nn.Conv2d(head_channels, configs.in_shape[1], 1)
        self.motion_gate_head = nn.Conv2d(head_channels, 1, 1)
        self.source_gate_head = nn.Conv2d(head_channels, 1, 1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.source_head.weight)
        nn.init.zeros_(self.source_head.bias)
        nn.init.zeros_(self.motion_gate_head.weight)
        nn.init.constant_(self.motion_gate_head.bias, -8.0)
        nn.init.zeros_(self.source_gate_head.weight)
        nn.init.constant_(self.source_gate_head.bias, -8.0)
        self.operator = EvolutionOperator(
            field_space='rain_rate', value_scale=configs.radar_value_scale,
            zr_a=configs.zr_a, zr_b=configs.zr_b)

    def load_direct_checkpoint(self, path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f'direct ConvLSTM checkpoint not found: {path}')
        checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        mapped = {}
        for key, value in state.items():
            key = key[6:] if key.startswith('model.') else key
            if key.startswith('cell_list.') or key.startswith('conv_last.'):
                mapped[key] = value
        missing, unexpected = self.direct_model.load_state_dict(mapped, strict=False)
        if missing or unexpected or not mapped:
            raise RuntimeError(
                f'incomplete direct checkpoint: missing={missing}, unexpected={unexpected}')
        return len(mapped)

    def freeze_direct(self):
        self.direct_model.requires_grad_(False)
        self.direct_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.direct_model.parameters()):
            self.direct_model.eval()
        return self

    def direct_forecast(self, history):
        b, _, c, h, w = history.shape
        future = history.new_zeros(b, self.configs.aft_seq_length, c, h, w)
        sequence = torch.cat([history, future], 1).permute(0, 1, 3, 4, 2)
        patches = reshape_patch(sequence.contiguous(), self.configs.patch_size)
        hp, wp, cp = patches.shape[2:]
        mask = history.new_zeros(
            b, self.configs.aft_seq_length - 1, hp, wp, cp)
        generated, _ = self.direct_model(patches, mask, return_loss=False)
        generated = reshape_patch_back(generated, self.configs.patch_size)
        return generated[:, -self.configs.aft_seq_length:].permute(
            0, 1, 4, 2, 3).contiguous()

    def forward(self, history, return_aux=False, blend_enabled=True):
        direct = self.direct_forecast(history)
        pyramid = self.decoder(self.features(history))
        b, t, c, h, w = direct.shape
        fine = pyramid['fine'][:, None].expand(-1, t, -1, -1, -1)
        lead = self.lead_embedding.weight[None, :, :, None, None].expand(
            b, -1, -1, h, w)
        head_input = torch.cat([fine, direct, lead], dim=2).reshape(b*t, -1, h, w)
        hidden = self.head(head_input)
        flow = float(self.configs.hybrid_max_residual_displacement) * torch.tanh(
            self.flow_head(hidden)).reshape(b, t, 2, h, w)
        source_logit = self.source_head(hidden).reshape(b, t, c, h, w)
        motion_gate_logit = self.motion_gate_head(hidden).reshape(b, t, 1, h, w)
        source_gate_logit = self.source_gate_head(hidden).reshape(b, t, 1, h, w)

        flat_direct = direct.reshape(b*t, c, h, w)
        warped = self.operator.warp(
            flat_direct, flow.reshape(b*t, 2, h, w)).reshape(b, t, c, h, w)
        warped_rain = normalized_dbz_to_rain(
            warped, self.configs.radar_value_scale,
            self.configs.zr_a, self.configs.zr_b)
        source_rain = float(self.configs.hybrid_max_source_rain) * torch.tanh(source_logit)
        physics = rain_to_normalized_dbz(
            (warped_rain + source_rain).clamp_min(0),
            self.configs.radar_value_scale, self.configs.zr_a, self.configs.zr_b)
        direct_rain = normalized_dbz_to_rain(
            direct, self.configs.radar_value_scale,
            self.configs.zr_a, self.configs.zr_b)
        motion_rain = warped_rain
        source_candidate_rain = (warped_rain + source_rain).clamp_min(0)
        motion_gate = torch.sigmoid(motion_gate_logit)
        source_gate = torch.sigmoid(source_gate_logit)
        motion_alpha_max = float(getattr(
            self.configs, 'hybrid_motion_alpha_max', 0.5))
        source_alpha_max = float(getattr(
            self.configs, 'hybrid_source_alpha_max', 0.25))
        if blend_enabled:
            motion_weight = motion_alpha_max * motion_gate
            source_weight = source_alpha_max * source_gate
        else:
            motion_weight = torch.zeros_like(motion_gate)
            source_weight = torch.zeros_like(source_gate)
        fused_rain = (direct_rain
                      + motion_weight * (motion_rain - direct_rain)
                      + source_weight * (source_candidate_rain - motion_rain))
        prediction = rain_to_normalized_dbz(
            fused_rain.clamp_min(0), self.configs.radar_value_scale,
            self.configs.zr_a, self.configs.zr_b)
        if not return_aux:
            return prediction
        return dict(prediction=prediction, direct_prediction=direct,
                    physics_prediction=physics, residual_flow=flow,
                    source_rain=source_rain, growth=source_rain.clamp_min(0),
                    decay=(-source_rain).clamp_min(0),
                    blend_alpha=motion_weight + source_weight,
                    learned_blend_alpha=motion_weight + source_weight,
                    motion_prediction=rain_to_normalized_dbz(
                        motion_rain.clamp_min(0), self.configs.radar_value_scale,
                        self.configs.zr_a, self.configs.zr_b),
                    source_prediction=rain_to_normalized_dbz(
                        source_candidate_rain, self.configs.radar_value_scale,
                        self.configs.zr_a, self.configs.zr_b),
                    direct_rain=direct_rain, motion_rain=motion_rain,
                    source_candidate_rain=source_candidate_rain,
                    motion_gate=motion_weight, source_gate=source_weight,
                    raw_motion_gate=motion_gate,
                    raw_source_gate=source_gate)
