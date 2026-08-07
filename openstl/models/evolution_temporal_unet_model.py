import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import (EvolutionOperator, FPNDecoder,
                             normalized_dbz_to_rain,
                             TemporalFeaturePyramid,
                             UNetFactorizedSourceHead, UNetMotionHead)


class EvolutionTemporalUNet_Model(nn.Module):
    """Temporal U-Net conditioned motion and factorized physical evolution."""

    def __init__(self, configs, **kwargs):
        super().__init__()
        _, in_channels, _, _ = configs.in_shape
        self.configs = configs
        channels = list(getattr(
            configs, 'temporal_unet_channels', [32, 64, 128, 192]))
        blocks = list(getattr(configs, 'temporal_unet_blocks', [2, 2, 2, 2]))
        mix_scales = list(getattr(
            configs, 'temporal_unet_mix_scales', [1, 2, 3]))
        fpn_channels = int(getattr(configs, 'temporal_unet_fpn_channels', 96))
        head_channels = int(getattr(
            configs, 'evolution_head_channels', 128))
        temporal_kernel = int(getattr(
            configs, 'temporal_unet_temporal_kernel', 3))
        convlstm_scales = list(getattr(
            configs, 'temporal_unet_convlstm_scales', []))
        convlstm_kernel = int(getattr(
            configs, 'temporal_unet_convlstm_kernel', 3))

        self.backbone = TemporalFeaturePyramid(
            in_channels, channels, blocks, mix_scales, temporal_kernel,
            convlstm_scales=convlstm_scales,
            convlstm_kernel=convlstm_kernel)
        self.decoder = FPNDecoder(channels, fpn_channels)
        self.motion_head = UNetMotionHead(
            fpn_channels, head_channels, configs.aft_seq_length)
        self.use_source = bool(getattr(configs, 'evolution_use_source', False))
        self.source_parameterization = str(getattr(
            configs, 'evolution_source_parameterization',
            'factorized_regime'))
        if self.use_source:
            if self.source_parameterization != 'factorized_regime':
                raise ValueError(
                    'EvolutionTemporalUNet supports only factorized_regime '
                    'source parameterization')
            self.source_max_rain = float(getattr(
                configs, 'evolution_source_max_rain', 35.0))
            if self.source_max_rain <= 0:
                raise ValueError('evolution_source_max_rain must be positive')
            self.source_head = UNetFactorizedSourceHead(
                fpn_channels=fpn_channels,
                source_channels=int(getattr(
                    configs, 'temporal_unet_source_channels', 32)),
                hidden_channels=int(getattr(
                    configs, 'temporal_unet_source_hidden_channels', 64)),
                field_channels=in_channels)
        self.max_displacement = float(getattr(
            configs, 'evolution_max_displacement', 2.0))
        self.forecast_steps = int(getattr(
            configs, 'evolution_forecast_steps', configs.aft_seq_length))
        if not 1 <= self.forecast_steps <= configs.aft_seq_length:
            raise ValueError(
                'evolution_forecast_steps must be in [1, aft_seq_length]')
        self.use_flow_gate = False
        self.operator = EvolutionOperator(
            align_corners=bool(getattr(
                configs, 'evolution_align_corners', True)),
            padding_mode=getattr(configs, 'evolution_padding_mode', 'zeros'),
            field_space=getattr(
                configs, 'evolution_field_space', 'normalized_dbz'),
            value_scale=float(getattr(configs, 'radar_value_scale', 50.0)),
            zr_a=float(getattr(configs, 'zr_a', 200.0)),
            zr_b=float(getattr(configs, 'zr_b', 1.6)),
            stop_gradient=bool(getattr(
                configs, 'evolution_stop_gradient', False)))

    def encode_history(self, history):
        pyramid = self.backbone(history)
        decoded = self.decoder(pyramid)
        return {'pyramid': pyramid, **decoded}

    def backbone_parameters(self):
        return list(self.backbone.parameters()) + list(
            self.decoder.parameters())

    def motion_parameters(self):
        return list(self.motion_head.parameters())

    def source_parameters(self):
        if not self.use_source:
            return []
        return list(self.source_head.parameters())

    @staticmethod
    def _gradient_magnitude(field):
        dx = torch.zeros_like(field)
        dy = torch.zeros_like(field)
        dx[..., :, 1:] = field[..., :, 1:] - field[..., :, :-1]
        dy[..., 1:, :] = field[..., 1:, :] - field[..., :-1, :]
        return torch.sqrt(dx.square() + dy.square() + 1e-12)

    @staticmethod
    def _erode_mask(mask, radius=1):
        flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
        eroded = -F.max_pool2d(
            -flat, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return (eroded > 0.5).reshape(mask.shape)

    def _factorized_capacity(self, advected_rain):
        values = getattr(
            self.configs, 'evolution_source_capacity_values', None)
        edges = getattr(
            self.configs, 'evolution_source_capacity_edges', None)
        if values is None:
            return torch.full_like(advected_rain, self.source_max_rain)
        values = [float(value) for value in values]
        edges = ([8.0, 16.0, 32.0] if edges is None
                 else [float(value) for value in edges])
        if len(values) != len(edges) + 1:
            raise ValueError(
                'evolution_source_capacity_values must have one more value '
                'than evolution_source_capacity_edges')
        capacity = torch.full_like(advected_rain, values[-1])
        previous = 0.1
        for edge, value in zip(edges, values):
            mask = (advected_rain >= previous) & (advected_rain < edge)
            capacity = torch.where(
                mask, torch.full_like(capacity, value), capacity)
            previous = edge
        return capacity

    def _factorized_source_forward(self, history, source_feature, flow,
                                   teacher_forcing=None,
                                   teacher_forcing_ratio=1.0):
        current = history[:, -1]
        collected = {
            key: [] for key in (
                'prediction', 'advected', 'advected_rain',
                'regime_probability', 'growth_fraction', 'decay_fraction',
                'growth_state', 'steady_state', 'decay_state',
                'growth_source', 'sink', 'net_source', 'source_rain',
                'positive_capacity', 'evolved_rain', 'regime_logits',
                'growth_logit', 'decay_logit')}
        active_threshold = float(getattr(
            self.configs, 'evolution_source_active_threshold', 0.1))
        for step in range(flow.shape[1]):
            if teacher_forcing is not None and step > 0:
                current = teacher_forcing[:, step - 1]
            transport_input = (
                current.detach() if self.operator.stop_gradient else current)
            advected = self.operator.warp(transport_input, flow[:, step])
            advected_rain = normalized_dbz_to_rain(
                advected, value_scale=self.operator.value_scale,
                zr_a=self.operator.zr_a, zr_b=self.operator.zr_b)
            source_prediction = self.source_head(
                source_feature, advected_rain, flow[:, step],
                self._gradient_magnitude(advected_rain),
                self.operator.max_rain, self.max_displacement)
            capacity = self._factorized_capacity(advected_rain)
            source_mask = None
            if teacher_forcing is not None:
                source_mask = self._erode_mask(
                    advected_rain >= active_threshold, radius=1)
            elif bool(getattr(
                    self.configs,
                    'evolution_factorized_mask_advected_inference', True)):
                source_mask = self._erode_mask(
                    advected_rain >= active_threshold, radius=1)
            step_result = self.operator.evolve_factorized_step(
                current, flow[:, step],
                source_prediction['regime_logits'],
                source_prediction['growth_logit'],
                source_prediction['decay_logit'], capacity,
                source_mask=source_mask)
            current = step_result['prediction']
            step_result.update(source_prediction)
            for key in collected:
                collected[key].append(step_result[key])
        result = {
            key: torch.stack(value, dim=1)
            for key, value in collected.items()}
        result['flow'] = flow
        result['source'] = result['source_rain']
        return result

    def forward(self, history, return_aux=False, teacher_forcing=None,
                teacher_forcing_ratio=1.0):
        if history.ndim != 5:
            raise ValueError('history must be [B,T,C,H,W]')
        if teacher_forcing is not None:
            expected = (history.shape[0], self.configs.aft_seq_length,
                        history.shape[2], history.shape[3], history.shape[4])
            if tuple(teacher_forcing.shape) != expected:
                raise ValueError(f'teacher_forcing must have shape {expected}')

        features = self.encode_history(history)
        raw_flow = self.motion_head(
            features['coarse'], features['bottleneck'])
        height, width = history.shape[-2:]
        raw_flow = F.interpolate(
            raw_flow, size=(height, width), mode='bilinear',
            align_corners=False)
        raw_flow = raw_flow.reshape(
            history.shape[0], self.configs.aft_seq_length,
            2, height, width)
        raw_flow = self.max_displacement * torch.tanh(raw_flow)
        flow = raw_flow[:, :self.forecast_steps]
        if self.use_source:
            source_feature = self.source_head.encode(features['fine'])
            result = self._factorized_source_forward(
                history, source_feature, flow,
                teacher_forcing=teacher_forcing,
                teacher_forcing_ratio=teacher_forcing_ratio)
        else:
            result = self.operator(history[:, -1], flow, source=None)
        result['raw_flow'] = raw_flow[:, :self.forecast_steps]
        result['flow_gate'] = raw_flow.new_ones(
            history.shape[0], self.forecast_steps, 1, height, width)
        if not self.use_source:
            result['raw_source'] = None
        return result if return_aux else result['prediction']

    def load_pretrained_motion(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        prefixes = ('backbone.', 'decoder.', 'motion_head.')
        extracted = {}
        for key, value in state.items():
            local_key = key[len('model.'):] if key.startswith('model.') else key
            if local_key.startswith(prefixes):
                extracted[local_key] = value
        target = {key for key in self.state_dict() if key.startswith(prefixes)}
        if set(extracted) != target:
            raise ValueError(
                f'Incompatible Temporal U-Net motion checkpoint: '
                f'missing={sorted(target-set(extracted))}, '
                f'unexpected={sorted(set(extracted)-target)}')
        self.load_state_dict(extracted, strict=False)
        return len(extracted)

    def load_pretrained_source(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        own_state = self.state_dict()
        extracted = {}
        for key, value in state.items():
            local_key = key[len('model.'):] if key.startswith('model.') else key
            if (local_key.startswith('source_head.')
                    and local_key in own_state
                    and own_state[local_key].shape == value.shape):
                extracted[local_key] = value
        self.load_state_dict(extracted, strict=False)
        return len(extracted)
