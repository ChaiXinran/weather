import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import (ConvLSTMCell, EvolutionOperator,
                             normalized_dbz_to_rain)


def _group_count(channels, requested):
    groups = min(channels, requested)
    while channels % groups:
        groups -= 1
    return groups


class EvolutionConvLSTM_Model(nn.Module):
    """History-only ConvLSTM encoder followed by explicit physical evolution."""

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super().__init__()
        _, channels, height, width = configs.in_shape
        self.configs = configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        self.frame_channel = channels * self.patch_size ** 2
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        cells = []
        for index in range(num_layers):
            in_channels = self.frame_channel if index == 0 else num_hidden[index - 1]
            cells.append(ConvLSTMCell(
                in_channels, num_hidden[index], patch_height, patch_width,
                configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cells)

        head_channels = int(getattr(configs, 'evolution_head_channels', num_hidden[-1]))
        groups = _group_count(head_channels, int(getattr(configs, 'evolution_head_groups', 8)))
        self.motion_head = nn.Sequential(
            nn.Conv2d(num_hidden[-1], head_channels, 3, padding=1),
            nn.GroupNorm(groups, head_channels), nn.SiLU(),
            nn.Conv2d(head_channels, head_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(head_channels, configs.aft_seq_length * 2, 1))
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        self.use_flow_gate = bool(getattr(configs, 'evolution_use_flow_gate', False))
        if self.use_flow_gate:
            self.flow_gate_head = nn.Sequential(
                nn.Conv2d(num_hidden[-1], head_channels, 3, padding=1),
                nn.GroupNorm(groups, head_channels), nn.SiLU(),
                nn.Conv2d(head_channels, configs.aft_seq_length, 1))
            nn.init.zeros_(self.flow_gate_head[-1].weight)
            initial_gate = float(
                getattr(configs, 'evolution_gate_initial', None) or 0.5)
            if not 0.0 < initial_gate < 1.0:
                raise ValueError('evolution_gate_initial must be between 0 and 1')
            nn.init.constant_(
                self.flow_gate_head[-1].bias,
                torch.logit(torch.tensor(initial_gate)).item())
        self.use_source = bool(getattr(configs, 'evolution_use_source', False))
        self.source_parameterization = str(getattr(
            configs, 'evolution_source_parameterization', 'legacy_signed'))
        if self.use_source:
            self.source_max_rain = float(
                getattr(configs, 'evolution_source_max_rain', 35.0))
            if self.source_max_rain <= 0:
                raise ValueError('evolution_source_max_rain must be positive')
            if self.source_parameterization == 'factorized_regime':
                input_channels = num_hidden[-1] + channels + 2 + channels
                self.factorized_trunk = nn.Sequential(
                    nn.Conv2d(input_channels, head_channels, 3, padding=1),
                    nn.GroupNorm(groups, head_channels), nn.SiLU(),
                    nn.Conv2d(head_channels, head_channels, 3, padding=1),
                    nn.SiLU())
                self.regime_head = nn.Conv2d(head_channels, 3, 1)
                self.growth_head = nn.Conv2d(head_channels, channels, 1)
                self.decay_head = nn.Conv2d(head_channels, channels, 1)
                self.use_edge_residual_flow = bool(getattr(
                    configs, 'evolution_use_edge_residual_flow', False))
                if self.use_edge_residual_flow:
                    self.edge_flow_head = nn.Conv2d(head_channels, 2, 1)
                    nn.init.zeros_(self.edge_flow_head.weight)
                    nn.init.zeros_(self.edge_flow_head.bias)
                    self.edge_flow_max_displacement = float(getattr(
                        configs, 'evolution_edge_flow_max_displacement', 0.25))
                nn.init.zeros_(self.regime_head.weight)
                nn.init.zeros_(self.growth_head.weight)
                nn.init.zeros_(self.decay_head.weight)
                with torch.no_grad():
                    self.regime_head.bias.copy_(
                        torch.tensor([0.0, 20.0, 0.0]))
                nn.init.constant_(self.growth_head.bias, -20.0)
                nn.init.constant_(self.decay_head.bias, -20.0)
            elif self.source_parameterization == 'bounded_state':
                self.source_lead_dim = int(getattr(
                    configs, 'evolution_source_lead_dim', 8))
                if self.source_lead_dim <= 0:
                    raise ValueError('source lead dimension must be positive')
                input_channels = (num_hidden[-1] + channels + 2
                                  + self.source_lead_dim)
                self.source_lead_embedding = nn.Embedding(
                    configs.aft_seq_length, self.source_lead_dim)
                self.source_decoder = nn.Sequential(
                    nn.Conv2d(input_channels, head_channels, 3, padding=1),
                    nn.GroupNorm(groups, head_channels), nn.SiLU(),
                    nn.Conv2d(head_channels, head_channels, 3, padding=1),
                    nn.SiLU(), nn.Conv2d(head_channels, channels, 1))
                nn.init.zeros_(self.source_decoder[-1].weight)
                nn.init.zeros_(self.source_decoder[-1].bias)
            elif self.source_parameterization == 'legacy_signed':
                self.source_head = nn.Sequential(
                    nn.Conv2d(num_hidden[-1], head_channels, 3, padding=1),
                    nn.GroupNorm(groups, head_channels), nn.SiLU(),
                    nn.Conv2d(head_channels, head_channels, 3, padding=1),
                    nn.SiLU(), nn.Conv2d(
                        head_channels, configs.aft_seq_length * channels, 1))
                nn.init.zeros_(self.source_head[-1].weight)
                nn.init.zeros_(self.source_head[-1].bias)
            else:
                raise ValueError(
                    'evolution_source_parameterization must be '
                    'legacy_signed, bounded_state, or factorized_regime')
        self.max_displacement = float(getattr(configs, 'evolution_max_displacement', 2.0))
        self.forecast_steps = int(getattr(
            configs, 'evolution_forecast_steps', configs.aft_seq_length))
        if not 1 <= self.forecast_steps <= configs.aft_seq_length:
            raise ValueError(
                'evolution_forecast_steps must be in [1, aft_seq_length]')
        self.operator = EvolutionOperator(
            align_corners=bool(getattr(configs, 'evolution_align_corners', True)),
            padding_mode=getattr(configs, 'evolution_padding_mode', 'zeros'),
            field_space=getattr(configs, 'evolution_field_space', 'normalized_dbz'),
            value_scale=float(getattr(configs, 'radar_value_scale', 50.0)),
            zr_a=float(getattr(configs, 'zr_a', 200.0)),
            zr_b=float(getattr(configs, 'zr_b', 1.6)),
            stop_gradient=bool(getattr(
                configs, 'evolution_stop_gradient', False)))

    def _patchify(self, frames):
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        patched = F.pixel_unshuffle(flat, self.patch_size)
        return patched.reshape(batch, time, *patched.shape[1:])

    def encode_history(self, frames):
        frames = self._patchify(frames)
        batch, _, _, height, width = frames.shape
        h = [frames.new_zeros(batch, hidden, height, width)
             for hidden in self.num_hidden]
        c = [tensor.clone() for tensor in h]
        for step in range(frames.shape[1]):
            h[0], c[0] = self.cell_list[0](frames[:, step], h[0], c[0])
            for layer in range(1, self.num_layers):
                h[layer], c[layer] = self.cell_list[layer](
                    h[layer - 1], h[layer], c[layer])
        return h[-1]

    def source_parameters(self):
        if not self.use_source:
            return []
        if self.source_parameterization == 'factorized_regime':
            parameters = (list(self.factorized_trunk.parameters())
                    + list(self.regime_head.parameters())
                    + list(self.growth_head.parameters())
                    + list(self.decay_head.parameters()))
            if getattr(self, 'use_edge_residual_flow', False):
                parameters += list(self.edge_flow_head.parameters())
            return parameters
        if self.source_parameterization == 'bounded_state':
            return list(self.source_decoder.parameters()) + list(
                self.source_lead_embedding.parameters())
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

    @staticmethod
    def _edge_mask(mask, radius=1):
        flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
        dilated = F.max_pool2d(
            flat, kernel_size=2 * radius + 1, stride=1, padding=radius)
        eroded = -F.max_pool2d(
            -flat, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return ((dilated > 0.5) & (eroded <= 0.5)).reshape(mask.shape)

    def _factorized_capacity(self, advected_rain):
        values = getattr(self.configs, 'evolution_source_capacity_values', None)
        edges = getattr(self.configs, 'evolution_source_capacity_edges', None)
        if values is None:
            return torch.full_like(advected_rain, self.source_max_rain)
        values = [float(value) for value in values]
        if edges is None:
            edges = [8.0, 16.0, 32.0]
        edges = [float(value) for value in edges]
        if len(values) != len(edges) + 1:
            raise ValueError(
                'evolution_source_capacity_values must have one more value '
                'than evolution_source_capacity_edges')
        capacity = torch.full_like(advected_rain, values[-1])
        previous = 0.1
        for edge, value in zip(edges, values):
            mask = (advected_rain >= previous) & (advected_rain < edge)
            capacity = torch.where(mask, torch.full_like(capacity, value), capacity)
            previous = edge
        return capacity

    def _bounded_source_forward(self, history, feature, flow,
                                teacher_forcing=None):
        batch, _, channels, full_height, full_width = history.shape
        patch_size = feature.shape[-2:]
        current = history[:, -1]
        collected = {
            key: [] for key in (
                'prediction', 'advected', 'advected_rain', 'source_rain',
                'source_tendency', 'source_positive_capacity',
                'evolved_rain', 'raw_source')}
        flow = flow[:, :self.forecast_steps]
        for step in range(flow.shape[1]):
            if teacher_forcing is not None and step > 0:
                current = teacher_forcing[:, step - 1]
            transport_input = current.detach() if self.operator.stop_gradient else current
            advected = self.operator.warp(transport_input, flow[:, step])
            advected_rain = normalized_dbz_to_rain(
                advected, value_scale=self.operator.value_scale,
                zr_a=self.operator.zr_a, zr_b=self.operator.zr_b)
            rain_feature = F.interpolate(
                advected_rain, size=patch_size, mode='area') / self.operator.max_rain
            patch_flow = F.interpolate(
                flow[:, step], size=patch_size, mode='bilinear',
                align_corners=False) / max(self.max_displacement, 1e-6)
            lead = self.source_lead_embedding.weight[step].view(
                1, -1, 1, 1).expand(batch, -1, *patch_size)
            decoder_input = torch.cat(
                [feature, rain_feature, patch_flow, lead], dim=1)
            raw_source = self.source_decoder(decoder_input)
            raw_source = F.interpolate(
                raw_source, size=(full_height, full_width), mode='bilinear',
                align_corners=False)
            step_result = self.operator.evolve_bounded_step(
                current, flow[:, step], raw_source, self.source_max_rain)
            current = step_result['prediction']
            collected['raw_source'].append(raw_source)
            for key in collected:
                if key != 'raw_source':
                    collected[key].append(step_result[key])
        result = {key: torch.stack(value, dim=1)
                  for key, value in collected.items()}
        result['flow'] = flow
        result['source'] = result['source_rain']
        return result

    def _factorized_source_forward(self, history, feature, flow,
                                   teacher_forcing=None):
        batch, _, channels, full_height, full_width = history.shape
        patch_size = feature.shape[-2:]
        current = history[:, -1]
        collected = {
            key: [] for key in (
                'prediction', 'advected', 'advected_rain',
                'regime_probability', 'growth_fraction', 'decay_fraction',
                'growth_state', 'steady_state', 'decay_state',
                'growth_source', 'sink', 'net_source', 'source_rain',
                'positive_capacity', 'evolved_rain', 'regime_logits',
                'growth_logit', 'decay_logit')}
        if getattr(self, 'use_edge_residual_flow', False):
            for key in ('base_flow', 'edge_flow', 'edge_mask'):
                collected[key] = []
        flow = flow[:, :self.forecast_steps]
        active_threshold = float(getattr(
            self.configs, 'evolution_source_active_threshold', 0.1))
        for step in range(flow.shape[1]):
            if teacher_forcing is not None and step > 0:
                current = teacher_forcing[:, step - 1]
            transport_input = current.detach() if self.operator.stop_gradient else current
            advected = self.operator.warp(transport_input, flow[:, step])
            advected_rain = normalized_dbz_to_rain(
                advected, value_scale=self.operator.value_scale,
                zr_a=self.operator.zr_a, zr_b=self.operator.zr_b)
            rain_feature = F.interpolate(
                advected_rain, size=patch_size, mode='area') / self.operator.max_rain
            gradient_feature = F.interpolate(
                self._gradient_magnitude(advected_rain), size=patch_size,
                mode='area') / self.operator.max_rain
            patch_flow = F.interpolate(
                flow[:, step], size=patch_size, mode='bilinear',
                align_corners=False) / max(self.max_displacement, 1e-6)
            decoder_input = torch.cat(
                [feature, rain_feature, patch_flow, gradient_feature], dim=1)
            hidden = self.factorized_trunk(decoder_input)
            step_flow = flow[:, step]
            if getattr(self, 'use_edge_residual_flow', False):
                edge_flow = F.interpolate(
                    self.edge_flow_head(hidden),
                    size=(full_height, full_width), mode='bilinear',
                    align_corners=False)
                edge_flow = self.edge_flow_max_displacement * torch.tanh(edge_flow)
                edge_mask = self._edge_mask(
                    advected_rain >= active_threshold, radius=1)
                edge_flow = edge_flow * edge_mask.to(edge_flow.dtype)
                step_flow = step_flow + edge_flow
                advected = self.operator.warp(transport_input, step_flow)
                advected_rain = normalized_dbz_to_rain(
                    advected, value_scale=self.operator.value_scale,
                    zr_a=self.operator.zr_a, zr_b=self.operator.zr_b)
            regime_logits = F.interpolate(
                self.regime_head(hidden), size=(full_height, full_width),
                mode='bilinear', align_corners=False)
            growth_logit = F.interpolate(
                self.growth_head(hidden), size=(full_height, full_width),
                mode='bilinear', align_corners=False)
            decay_logit = F.interpolate(
                self.decay_head(hidden), size=(full_height, full_width),
                mode='bilinear', align_corners=False)
            capacity = self._factorized_capacity(advected_rain)
            source_mask = None
            if teacher_forcing is not None:
                target_rain = normalized_dbz_to_rain(
                    teacher_forcing[:, step], value_scale=self.operator.value_scale,
                    zr_a=self.operator.zr_a, zr_b=self.operator.zr_b)
                source_mask = self._erode_mask(
                    (advected_rain >= active_threshold)
                    & (target_rain >= active_threshold), radius=1)
            elif bool(getattr(
                    self.configs,
                    'evolution_factorized_mask_advected_inference', True)):
                source_mask = self._erode_mask(
                    advected_rain >= active_threshold, radius=1)
            step_result = self.operator.evolve_factorized_step(
                current, step_flow, regime_logits, growth_logit,
                decay_logit, capacity, source_mask=source_mask)
            current = step_result['prediction']
            step_result['regime_logits'] = regime_logits
            step_result['growth_logit'] = growth_logit
            step_result['decay_logit'] = decay_logit
            step_result['positive_capacity'] = step_result['positive_capacity']
            if getattr(self, 'use_edge_residual_flow', False):
                step_result['base_flow'] = flow[:, step]
                step_result['edge_flow'] = edge_flow
                step_result['edge_mask'] = edge_mask
            for key in collected:
                collected[key].append(step_result[key])
        result = {key: torch.stack(value, dim=1)
                  for key, value in collected.items()}
        result['flow'] = (result['base_flow'] + result['edge_flow']
                          if getattr(self, 'use_edge_residual_flow', False)
                          else flow)
        result['source'] = result['source_rain']
        return result

    def forward(self, history, return_aux=False, teacher_forcing=None):
        if history.ndim != 5:
            raise ValueError('history must be [B,T,C,H,W]')
        if teacher_forcing is not None:
            expected = (history.shape[0], self.configs.aft_seq_length,
                        history.shape[2], history.shape[3], history.shape[4])
            if tuple(teacher_forcing.shape) != expected:
                raise ValueError(f'teacher_forcing must have shape {expected}')
        feature = self.encode_history(history)
        batch, _, height, width = history.shape[0], history.shape[2], history.shape[3], history.shape[4]
        raw_flow = self.motion_head(feature)
        raw_flow = F.interpolate(raw_flow, size=(height, width), mode='bilinear',
                                 align_corners=False)
        raw_flow = raw_flow.reshape(
            batch, self.configs.aft_seq_length, 2, height, width)
        raw_flow = self.max_displacement * torch.tanh(raw_flow)
        if self.use_flow_gate:
            gate_logits = self.flow_gate_head(feature)
            gate_logits = F.interpolate(
                gate_logits, size=(height, width), mode='bilinear',
                align_corners=False)
            flow_gate = torch.sigmoid(gate_logits).unsqueeze(2)
        else:
            flow_gate = raw_flow.new_ones(
                batch, self.configs.aft_seq_length, 1, height, width)
        flow = raw_flow * flow_gate
        raw_source = None
        source_rain = None
        if self.use_source and self.source_parameterization == 'legacy_signed':
            raw_source = self.source_head(feature)
            raw_source = F.interpolate(
                raw_source, size=(height, width), mode='bilinear',
                align_corners=False)
            raw_source = raw_source.reshape(
                batch, self.configs.aft_seq_length,
                history.shape[2], height, width)
            source_rain = self.source_max_rain * torch.tanh(raw_source)
        if self.use_source and self.source_parameterization == 'bounded_state':
            result = self._bounded_source_forward(
                history, feature, flow, teacher_forcing=teacher_forcing)
        elif self.use_source and self.source_parameterization == 'factorized_regime':
            result = self._factorized_source_forward(
                history, feature, flow, teacher_forcing=teacher_forcing)
        else:
            result = self.operator(history[:, -1], flow, source=source_rain)
        result['raw_flow'] = raw_flow
        result['flow_gate'] = flow_gate
        if not (self.use_source
                and self.source_parameterization in (
                    'bounded_state', 'factorized_regime')):
            result['raw_source'] = raw_source
        return result if return_aux else result['prediction']

    def load_pretrained_motion(self, checkpoint_path):
        """Load encoder and raw motion head while leaving a new gate untouched."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        prefixes = ('cell_list.', 'motion_head.')
        extracted = {}
        for key, value in state.items():
            local_key = key[len('model.'):] if key.startswith('model.') else key
            if local_key.startswith(prefixes):
                extracted[local_key] = value
        target = {key for key in self.state_dict() if key.startswith(prefixes)}
        if set(extracted) != target:
            raise ValueError(
                f'Incompatible motion checkpoint: missing={sorted(target-set(extracted))}, '
                f'unexpected={sorted(set(extracted)-target)}')
        self.load_state_dict(extracted, strict=False)
        return len(extracted)

    def load_pretrained_source(self, checkpoint_path):
        """Load compatible source tensors while leaving newly added heads fresh."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        prefixes = ('factorized_trunk.', 'regime_head.', 'growth_head.',
                    'decay_head.')
        own_state = self.state_dict()
        extracted = {}
        for key, value in state.items():
            local_key = key[len('model.'):] if key.startswith('model.') else key
            if (local_key.startswith(prefixes)
                    and local_key in own_state
                    and own_state[local_key].shape == value.shape):
                extracted[local_key] = value
        self.load_state_dict(extracted, strict=False)
        return len(extracted)

    def load_pretrained_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        extracted = {}
        for key, value in state.items():
            marker = 'model.cell_list.'
            if marker in key:
                extracted[key[key.index(marker) + len('model.'):]] = value
            elif key.startswith('cell_list.'):
                extracted[key] = value
        target = {key for key in self.state_dict() if key.startswith('cell_list.')}
        if not extracted:
            raise ValueError(f'No ConvLSTM encoder tensors found in {checkpoint_path}')
        unknown = set(extracted) - target
        missing = target - set(extracted)
        if unknown or missing:
            raise ValueError(
                f'Incompatible ConvLSTM encoder checkpoint: missing={sorted(missing)}, '
                f'unexpected={sorted(unknown)}')
        self.load_state_dict(extracted, strict=False)
        return len(extracted)
