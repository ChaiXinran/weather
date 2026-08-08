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


class DirectPhysicsRouted_Model(nn.Module):
    """ConvLSTM prior with strongly routed preserve/motion/decay experts."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        hidden = [int(value) for value in configs.num_hidden.split(',')]
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
        field_channels = int(configs.in_shape[1])
        # Previous/current/next prior, temporal difference and spatial gradient.
        context_channels = 5 * field_channels
        head_channels = int(configs.hybrid_head_channels)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels + context_channels + lead_channels,
                      head_channels, 3, padding=1),
            nn.GroupNorm(8, head_channels), nn.SiLU(),
            DWResidualBlock(head_channels))
        self.flow_head = nn.Conv2d(head_channels, 2, 1)
        self.decay_head = nn.Conv2d(head_channels, field_channels, 1)
        self.router_trunk = nn.Sequential(
            DWResidualBlock(head_channels),
            DWResidualBlock(head_channels))
        self.router_head = nn.Conv2d(head_channels, 3, 1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.decay_head.weight)
        nn.init.constant_(self.decay_head.bias, -3.0)
        nn.init.zeros_(self.router_head.weight)
        initial = torch.tensor(configs.v3a_initial_route_probability).float()
        if initial.numel() != 3 or (initial <= 0).any():
            raise ValueError('v3a_initial_route_probability must be 3 positive values')
        initial = initial / initial.sum()
        with torch.no_grad():
            self.router_head.bias.copy_(initial.log())
        self.operator = EvolutionOperator(
            field_space='rain_rate', value_scale=configs.radar_value_scale,
            zr_a=configs.zr_a, zr_b=configs.zr_b)

    def load_direct_checkpoint(self, path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f'direct ConvLSTM checkpoint not found: {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        mapped = {}
        for key, value in state.items():
            key = key[6:] if key.startswith('model.') else key
            if key.startswith('cell_list.') or key.startswith('conv_last.'):
                mapped[key] = value
        missing, unexpected = self.direct_model.load_state_dict(mapped, strict=False)
        if missing or unexpected or not mapped:
            raise RuntimeError(
                f'incomplete direct checkpoint: missing={missing}, '
                f'unexpected={unexpected}')
        return len(mapped)

    def load_v2_correction_checkpoint(self, path):
        """Load all shape-compatible V2 U-Net/motion weights."""
        if not path:
            return 0, []
        if not os.path.isfile(path):
            raise FileNotFoundError(f'V2 correction checkpoint not found: {path}')
        state = torch.load(path, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        target = self.state_dict()
        loaded, skipped = {}, []
        prefixes = ('features.', 'decoder.', 'head.', 'flow_head.')
        for key, value in state.items():
            key = key[6:] if key.startswith('model.') else key
            if not key.startswith(prefixes):
                continue
            if key in target and target[key].shape == value.shape:
                loaded[key] = value
            elif key == 'head.0.weight' and key in target:
                # V2 input order was [fine, direct_t, lead]. V3a expands the
                # direct context to [prev,current,next,delta,gradient] while
                # preserving fine and lead channels exactly.
                expanded = torch.zeros_like(target[key])
                fine_channels = int(self.configs.hybrid_fpn_channels)
                lead_channels = int(self.configs.hybrid_lead_channels)
                if value.shape[1] == fine_channels + 1 + lead_channels:
                    expanded[:, :fine_channels] = value[:, :fine_channels]
                    expanded[:, fine_channels + 1:fine_channels + 2] = \
                        value[:, fine_channels:fine_channels + 1]
                    expanded[:, -lead_channels:] = value[:, -lead_channels:]
                    loaded[key] = expanded
                else:
                    skipped.append(key)
            else:
                skipped.append(key)
        self.load_state_dict(loaded, strict=False)
        return len(loaded), skipped

    def freeze_direct(self):
        self.direct_model.requires_grad_(False)
        self.direct_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad
                   for parameter in self.direct_model.parameters()):
            self.direct_model.eval()
        return self

    def direct_forecast(self, history):
        batch, _, channels, height, width = history.shape
        future = history.new_zeros(
            batch, self.configs.aft_seq_length, channels, height, width)
        sequence = torch.cat([history, future], 1).permute(0, 1, 3, 4, 2)
        patches = reshape_patch(sequence.contiguous(), self.configs.patch_size)
        patch_height, patch_width, patch_channels = patches.shape[2:]
        mask = history.new_zeros(
            batch, self.configs.aft_seq_length - 1,
            patch_height, patch_width, patch_channels)
        generated, _ = self.direct_model(patches, mask, return_loss=False)
        generated = reshape_patch_back(generated, self.configs.patch_size)
        return generated[:, -self.configs.aft_seq_length:].permute(
            0, 1, 4, 2, 3).contiguous()

    @staticmethod
    def _spatial_gradient(field):
        dx = F.pad(field[..., :, 1:] - field[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(field[..., 1:, :] - field[..., :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-8)

    def forward(self, history, return_aux=False):
        direct = self.direct_forecast(history)
        decoded = self.decoder(self.features(history))
        batch, time, channels, height, width = direct.shape
        previous = torch.cat([direct[:, :1], direct[:, :-1]], dim=1)
        following = torch.cat([direct[:, 1:], direct[:, -1:]], dim=1)
        delta = direct - previous
        gradient = self._spatial_gradient(direct)
        prior_context = torch.cat(
            [previous, direct, following, delta, gradient], dim=2)
        fine = decoded['fine'][:, None].expand(-1, time, -1, -1, -1)
        lead = self.lead_embedding.weight[None, :, :, None, None].expand(
            batch, -1, -1, height, width)
        head_input = torch.cat([fine, prior_context, lead], dim=2).reshape(
            batch * time, -1, height, width)
        hidden = self.head(head_input)
        flow = float(self.configs.hybrid_max_residual_displacement) * torch.tanh(
            self.flow_head(hidden)).reshape(batch, time, 2, height, width)
        decay_fraction = torch.sigmoid(self.decay_head(hidden)).reshape(
            batch, time, channels, height, width)
        route_logits = self.router_head(self.router_trunk(hidden)).reshape(
            batch, time, 3, height, width)
        temperature = max(float(self.configs.v3a_router_temperature), 1e-4)
        route_probability = torch.softmax(route_logits / temperature, dim=2)

        flat_direct = direct.reshape(batch * time, channels, height, width)
        warped = self.operator.warp(
            flat_direct, flow.reshape(batch * time, 2, height, width)).reshape(
                batch, time, channels, height, width)
        direct_rain = normalized_dbz_to_rain(
            direct, self.configs.radar_value_scale,
            self.configs.zr_a, self.configs.zr_b)
        motion_rain = normalized_dbz_to_rain(
            warped, self.configs.radar_value_scale,
            self.configs.zr_a, self.configs.zr_b)
        decay_rain = direct_rain * (1.0 - decay_fraction)
        candidates = torch.stack(
            [direct_rain, motion_rain, decay_rain], dim=2)
        fused_rain = (route_probability.unsqueeze(3) * candidates).sum(dim=2)
        prediction = rain_to_normalized_dbz(
            fused_rain.clamp_min(0), self.configs.radar_value_scale,
            self.configs.zr_a, self.configs.zr_b)
        if not return_aux:
            return prediction
        candidate_prediction = [
            rain_to_normalized_dbz(
                candidate.clamp_min(0), self.configs.radar_value_scale,
                self.configs.zr_a, self.configs.zr_b)
            for candidate in (direct_rain, motion_rain, decay_rain)]
        return {
            'prediction': prediction,
            'direct_prediction': direct,
            'preserve_prediction': candidate_prediction[0],
            'motion_prediction': candidate_prediction[1],
            'decay_prediction': candidate_prediction[2],
            'direct_rain': direct_rain,
            'motion_rain': motion_rain,
            'decay_rain': decay_rain,
            'fused_rain': fused_rain,
            'residual_flow': flow,
            'decay_fraction': decay_fraction,
            'route_probability': route_probability,
            'route_logits': route_logits,
        }
