import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EvolutionOperator(nn.Module):
    """Autoregressive differentiable advection with an optional source term."""

    def __init__(self, align_corners=True, padding_mode='zeros'):
        super().__init__()
        self.align_corners = align_corners
        self.padding_mode = padding_mode

    def forward(self, initial_field, incremental_flow, source=None):
        if incremental_flow.ndim != 5 or incremental_flow.shape[2] != 2:
            raise ValueError('incremental_flow must be [B,T,2,H,W]')
        if source is not None and source.shape[:2] != incremental_flow.shape[:2]:
            raise ValueError('source and flow time dimensions must match')
        current = initial_field
        predictions, advected_fields = [], []
        for step in range(incremental_flow.shape[1]):
            advected = backward_warp(
                current, incremental_flow[:, step], self.align_corners,
                self.padding_mode)
            current = advected if source is None else advected + source[:, step]
            advected_fields.append(advected)
            predictions.append(current)
        return {
            'prediction': torch.stack(predictions, dim=1),
            'advected': torch.stack(advected_fields, dim=1),
            'flow': incremental_flow,
            'source': source,
        }
