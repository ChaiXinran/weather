import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import ConvLSTMCell, EvolutionOperator


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
        self.max_displacement = float(getattr(configs, 'evolution_max_displacement', 2.0))
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

    def forward(self, history, return_aux=False):
        if history.ndim != 5:
            raise ValueError('history must be [B,T,C,H,W]')
        feature = self.encode_history(history)
        batch, _, height, width = history.shape[0], history.shape[2], history.shape[3], history.shape[4]
        raw_flow = self.motion_head(feature)
        raw_flow = F.interpolate(raw_flow, size=(height, width), mode='bilinear',
                                 align_corners=False)
        flow = raw_flow.reshape(batch, self.configs.aft_seq_length, 2, height, width)
        flow = self.max_displacement * torch.tanh(flow)
        result = self.operator(history[:, -1], flow)
        return result if return_aux else result['prediction']

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
