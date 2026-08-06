import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_dbz_to_rain(field, value_scale=50.0, zr_a=200.0, zr_b=1.6):
    """Convert a normalized dBZ tensor to rain rate in mm/h."""
    dbz = field * float(value_scale)
    linear_z = torch.pow(10.0, dbz / 10.0)
    return torch.pow(linear_z / float(zr_a), 1.0 / float(zr_b))


def rain_to_normalized_dbz(rain, value_scale=50.0, zr_a=200.0,
                           zr_b=1.6, eps=1e-6):
    """Convert rain rate in mm/h to the normalized dBZ model field."""
    linear_z = float(zr_a) * torch.pow(rain.clamp_min(0.0), float(zr_b))
    dbz = 10.0 * torch.log10(linear_z.clamp_min(eps))
    return (dbz / float(value_scale)).clamp(0.0, 1.0)


def backward_warp(field, flow, align_corners=True, padding_mode='zeros'):
    """Warp ``field`` by a flow in pixels.

    ``flow[:, 0]`` is dx (positive moves content right) and ``flow[:, 1]``
    is dy (positive moves content down).
    """
    if field.ndim != 4 or flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError('field must be [B,C,H,W] and flow [B,2,H,W]')
    if field.shape[0] != flow.shape[0] or field.shape[-2:] != flow.shape[-2:]:
        raise ValueError('field and flow batch/spatial dimensions must match')

    batch, _, height, width = field.shape
    dtype = field.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=field.device, dtype=dtype),
        torch.arange(width, device=field.device, dtype=dtype), indexing='ij')
    x = x.unsqueeze(0).expand(batch, -1, -1) - flow[:, 0]
    y = y.unsqueeze(0).expand(batch, -1, -1) - flow[:, 1]
    if align_corners:
        x = 2.0 * x / max(width - 1, 1) - 1.0
        y = 2.0 * y / max(height - 1, 1) - 1.0
    else:
        x = 2.0 * (x + 0.5) / width - 1.0
        y = 2.0 * (y + 0.5) / height - 1.0
    grid = torch.stack((x, y), dim=-1)
    return F.grid_sample(field, grid, mode='bilinear',
                         padding_mode=padding_mode,
                         align_corners=align_corners)


def warp_field(field, flow, field_space='normalized_dbz', value_scale=50.0,
               zr_a=200.0, zr_b=1.6, align_corners=True,
               padding_mode='zeros', eps=1e-6):
    """Warp a normalized-dBZ field in a selected physical variable space."""
    if field_space == 'normalized_dbz':
        return backward_warp(
            field, flow, align_corners=align_corners,
            padding_mode=padding_mode)
    dbz = field * float(value_scale)
    linear_z = torch.pow(10.0, dbz / 10.0)
    if field_space == 'linear_z':
        warped = backward_warp(
            linear_z, flow, align_corners=align_corners,
            padding_mode=padding_mode)
        warped_dbz = 10.0 * torch.log10(warped.clamp_min(eps))
    elif field_space == 'rain_rate':
        rain_rate = normalized_dbz_to_rain(
            field, value_scale=value_scale, zr_a=zr_a, zr_b=zr_b)
        warped = backward_warp(
            rain_rate, flow, align_corners=align_corners,
            padding_mode=padding_mode)
        return rain_to_normalized_dbz(
            warped, value_scale=value_scale, zr_a=zr_a, zr_b=zr_b, eps=eps)
    else:
        raise ValueError(
            'field_space must be normalized_dbz, linear_z, or rain_rate')
    return (warped_dbz / float(value_scale)).clamp(0.0, 1.0)


class EvolutionOperator(nn.Module):
    """Autoregressive differentiable advection with an optional source term."""

    def __init__(self, align_corners=True, padding_mode='zeros',
                 field_space='normalized_dbz', value_scale=50.0,
                 zr_a=200.0, zr_b=1.6, stop_gradient=False):
        super().__init__()
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.field_space = field_space
        self.value_scale = float(value_scale)
        self.zr_a = float(zr_a)
        self.zr_b = float(zr_b)
        self.stop_gradient = bool(stop_gradient)
        self.max_rain = float(
            (10.0 ** (self.value_scale / 10.0) / self.zr_a)
            ** (1.0 / self.zr_b))

    def warp(self, field, flow, field_space=None, padding_mode=None):
        return warp_field(
            field, flow,
            field_space=self.field_space if field_space is None else field_space,
            value_scale=self.value_scale, zr_a=self.zr_a, zr_b=self.zr_b,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode if padding_mode is None else padding_mode)

    def bounded_source(self, advected_rain, source_logit, source_max_rain):
        """Map one signed logit to a physically representable rain tendency."""
        if self.field_space != 'rain_rate':
            raise ValueError(
                'bounded physical source requires field_space="rain_rate"')
        if advected_rain.shape != source_logit.shape:
            raise ValueError('advected_rain and source_logit must share shape')
        source_max_rain = float(source_max_rain)
        if source_max_rain <= 0:
            raise ValueError('source_max_rain must be positive')
        tendency = torch.tanh(source_logit)
        positive_capacity = torch.minimum(
            torch.full_like(advected_rain, source_max_rain),
            (self.max_rain - advected_rain).clamp_min(0.0))
        source = torch.where(
            tendency >= 0.0,
            positive_capacity * tendency,
            advected_rain * tendency)
        return source, tendency, positive_capacity

    def evolve_bounded_step(self, field, flow, source_logit,
                            source_max_rain):
        """Advect and apply one bounded source step without output clamping."""
        transport_input = field.detach() if self.stop_gradient else field
        advected = self.warp(transport_input, flow)
        advected_rain = normalized_dbz_to_rain(
            advected, value_scale=self.value_scale,
            zr_a=self.zr_a, zr_b=self.zr_b)
        source, tendency, positive_capacity = self.bounded_source(
            advected_rain, source_logit, source_max_rain)
        evolved_rain = advected_rain + source
        evolved = rain_to_normalized_dbz(
            evolved_rain, value_scale=self.value_scale,
            zr_a=self.zr_a, zr_b=self.zr_b)
        return {
            'prediction': evolved,
            'advected': advected,
            'advected_rain': advected_rain,
            'source_rain': source,
            'source_tendency': tendency,
            'source_positive_capacity': positive_capacity,
            'evolved_rain': evolved_rain,
        }

    def evolve_factorized_step(self, field, flow, regime_logits,
                               growth_fraction, decay_fraction,
                               positive_capacity, source_mask=None):
        """Advect and mix growth/steady/decay rain-rate candidate states.

        ``regime_logits`` has three channels ordered as growth, steady, decay.
        ``growth_fraction`` and ``decay_fraction`` are raw logits mapped through
        sigmoid so the source/sink magnitudes stay within physical bounds.
        """
        if self.field_space != 'rain_rate':
            raise ValueError(
                'factorized physical source requires field_space="rain_rate"')
        if field.ndim != 4 or flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError('field must be [B,C,H,W] and flow [B,2,H,W]')
        if regime_logits.ndim != 4 or regime_logits.shape[1] != 3:
            raise ValueError('regime_logits must have shape [B,3,H,W]')
        expected = field.shape
        for name, value in (('growth_fraction', growth_fraction),
                            ('decay_fraction', decay_fraction)):
            if tuple(value.shape) != tuple(expected):
                raise ValueError(f'{name} must have shape {tuple(expected)}')

        transport_input = field.detach() if self.stop_gradient else field
        advected = self.warp(transport_input, flow)
        advected_rain = normalized_dbz_to_rain(
            advected, value_scale=self.value_scale,
            zr_a=self.zr_a, zr_b=self.zr_b)
        capacity = torch.as_tensor(
            positive_capacity, device=advected_rain.device,
            dtype=advected_rain.dtype)
        if capacity.ndim == 0:
            capacity = torch.full_like(advected_rain, float(capacity))
        if tuple(capacity.shape) != tuple(expected):
            raise ValueError(
                f'positive_capacity must be scalar or shape {tuple(expected)}')
        capacity = torch.minimum(
            capacity.clamp_min(0.0),
            (self.max_rain - advected_rain).clamp_min(0.0))

        regime_probability = torch.softmax(regime_logits, dim=1)
        growth_alpha = torch.sigmoid(growth_fraction)
        decay_alpha = torch.sigmoid(decay_fraction)
        growth_source = growth_alpha * capacity
        sink = decay_alpha * advected_rain
        growth_state = advected_rain + growth_source
        steady_state = advected_rain
        decay_state = (advected_rain - sink).clamp_min(0.0)
        mixed_rain = (
            regime_probability[:, 0:1] * growth_state
            + regime_probability[:, 1:2] * steady_state
            + regime_probability[:, 2:3] * decay_state)

        net_source = mixed_rain - advected_rain
        if source_mask is not None:
            if tuple(source_mask.shape) != tuple(expected):
                raise ValueError(
                    f'source_mask must have shape {tuple(expected)}')
            net_source = net_source * source_mask.to(net_source.dtype)
            mixed_rain = advected_rain + net_source
        evolved_rain = mixed_rain.clamp_min(0.0)
        prediction = rain_to_normalized_dbz(
            evolved_rain, value_scale=self.value_scale,
            zr_a=self.zr_a, zr_b=self.zr_b)
        return {
            'prediction': prediction,
            'advected': advected,
            'advected_rain': advected_rain,
            'regime_probability': regime_probability,
            'growth_fraction': growth_alpha,
            'decay_fraction': decay_alpha,
            'growth_state': growth_state,
            'steady_state': steady_state,
            'decay_state': decay_state,
            'growth_source': growth_source,
            'sink': sink,
            'net_source': net_source,
            'source_rain': net_source,
            'positive_capacity': capacity,
            'evolved_rain': evolved_rain,
        }

    def forward(self, initial_field, incremental_flow, source=None):
        if incremental_flow.ndim != 5 or incremental_flow.shape[2] != 2:
            raise ValueError('incremental_flow must be [B,T,2,H,W]')
        if source is not None:
            expected = (incremental_flow.shape[0], incremental_flow.shape[1],
                        initial_field.shape[1], *initial_field.shape[-2:])
            if tuple(source.shape) != expected:
                raise ValueError(f'source must have shape {expected}')
            if self.field_space != 'rain_rate':
                raise ValueError(
                    'physical source is supported only when field_space="rain_rate"')

        # Preserve the already validated R4-b path bit-for-bit when no source
        # is supplied. The physical-state branch below is activated only by an
        # explicit rain-rate source tensor.
        if source is None:
            current = initial_field
            predictions, advected_fields = [], []
            for step in range(incremental_flow.shape[1]):
                transport_input = current.detach() if self.stop_gradient else current
                advected = self.warp(transport_input, incremental_flow[:, step])
                current = advected
                advected_fields.append(advected)
                predictions.append(current)
            return {
                'prediction': torch.stack(predictions, dim=1),
                'advected': torch.stack(advected_fields, dim=1),
                'flow': incremental_flow,
                'source': None,
                'advected_rain': None,
                'source_rain': None,
                'evolved_rain': None,
            }

        current = initial_field
        predictions, advected_fields = [], []
        advected_rain_fields, evolved_rain_fields = [], []
        for step in range(incremental_flow.shape[1]):
            transport_input = current.detach() if self.stop_gradient else current
            # self.warp performs advection in rain-rate space. Convert its
            # normalized-dBZ transport result back to rain before adding the
            # physical residual; retaining this boundary round-trip makes a
            # zero-initialized source exactly compatible with R4-b.
            advected = self.warp(transport_input, incremental_flow[:, step])
            advected_rain = normalized_dbz_to_rain(
                advected, value_scale=self.value_scale,
                zr_a=self.zr_a, zr_b=self.zr_b)
            current_rain = (advected_rain + source[:, step]).clamp_min(0.0)
            current = rain_to_normalized_dbz(
                current_rain, value_scale=self.value_scale,
                zr_a=self.zr_a, zr_b=self.zr_b)
            advected_rain_fields.append(advected_rain)
            evolved_rain_fields.append(current_rain)
            advected_fields.append(advected)
            predictions.append(current)
        return {
            'prediction': torch.stack(predictions, dim=1),
            'advected': torch.stack(advected_fields, dim=1),
            'flow': incremental_flow,
            'source': source,
            'advected_rain': torch.stack(advected_rain_fields, dim=1),
            'source_rain': source,
            'evolved_rain': torch.stack(evolved_rain_fields, dim=1),
        }
