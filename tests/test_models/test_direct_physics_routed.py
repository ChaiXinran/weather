from types import SimpleNamespace

import torch

from openstl.models import DirectPhysicsRouted_Model


def tiny_config():
    return SimpleNamespace(
        in_shape=[10, 1, 16, 16], pre_seq_length=10,
        aft_seq_length=20, total_length=30,
        num_hidden='4', patch_size=2, filter_size=3, stride=1,
        layer_norm=0, reverse_scheduled_sampling=0,
        hybrid_unet_channels=[4, 8, 16, 32],
        hybrid_unet_blocks=[1, 1, 1, 1],
        hybrid_temporal_mix_scales=[0, 1, 2, 3],
        hybrid_convlstm_scales=[], hybrid_temporal_kernel=3,
        hybrid_convlstm_kernel=3, hybrid_fpn_channels=8,
        hybrid_lead_channels=4, hybrid_head_channels=8,
        hybrid_max_residual_displacement=2.0,
        v3a_initial_route_probability=[0.8, 0.15, 0.05],
        v3a_router_temperature=1.0,
        radar_value_scale=50.0, zr_a=200.0, zr_b=1.6)


def test_routed_forward_shapes_and_candidate_bounds():
    model = DirectPhysicsRouted_Model(tiny_config()).eval()
    with torch.no_grad():
        result = model(torch.rand(2, 10, 1, 16, 16), return_aux=True)
    assert result['prediction'].shape == (2, 20, 1, 16, 16)
    assert result['residual_flow'].shape == (2, 20, 2, 16, 16)
    assert result['route_probability'].shape == (2, 20, 3, 16, 16)
    assert torch.allclose(
        result['route_probability'].sum(dim=2),
        torch.ones(2, 20, 16, 16), atol=1e-6)
    assert torch.all(result['decay_rain'] <= result['direct_rain'] + 1e-6)
    assert torch.all(result['decay_rain'] >= 0)
