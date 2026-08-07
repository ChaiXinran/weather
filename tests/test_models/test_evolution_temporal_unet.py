from types import SimpleNamespace

import torch

from openstl.models import EvolutionTemporalUNet_Model


def _config(height=66, width=70):
    return SimpleNamespace(
        in_shape=[10, 1, height, width], pre_seq_length=10,
        aft_seq_length=4, total_length=14,
        temporal_unet_channels=[8, 12, 16, 24],
        temporal_unet_blocks=[1, 1, 1, 1],
        temporal_unet_fpn_channels=12,
        temporal_unet_mix_scales=[1, 2, 3],
        temporal_unet_temporal_kernel=3,
        evolution_head_channels=12,
        evolution_max_displacement=2.0,
        evolution_field_space='normalized_dbz')


def test_temporal_unet_preserves_odd_spatial_sizes_and_starts_persistent():
    model = EvolutionTemporalUNet_Model(_config())
    history = torch.rand(1, 10, 1, 66, 70)
    result = model(history, return_aux=True)
    assert result['prediction'].shape == (1, 4, 1, 66, 70)
    assert result['flow'].shape == (1, 4, 2, 66, 70)
    torch.testing.assert_close(result['flow'], torch.zeros_like(result['flow']))
    torch.testing.assert_close(
        result['prediction'], history[:, -1:].expand(-1, 4, -1, -1, -1),
        atol=1e-4, rtol=0)


def test_temporal_unet_multiscale_shapes_and_fpn_width():
    model = EvolutionTemporalUNet_Model(_config())
    history = torch.rand(1, 10, 1, 66, 70)
    features = model.encode_history(history)
    expected = [(66, 70), (33, 35), (17, 18), (9, 9)]
    assert [tuple(item.shape[-2:]) for item in features['pyramid']] == expected
    assert features['fine'].shape == (1, 12, 66, 70)
    assert features['middle'].shape == (1, 12, 33, 35)
    assert features['coarse'].shape == (1, 12, 17, 18)
    assert features['bottleneck'].shape == (1, 12, 9, 9)


def test_temporal_unet_bottleneck_convlstm_shapes_and_gradients():
    config = _config(height=16, width=18)
    config.temporal_unet_convlstm_scales = [3]
    config.temporal_unet_convlstm_kernel = 3
    model = EvolutionTemporalUNet_Model(config)
    history = torch.rand(2, 4, 1, 16, 18)
    features = model.encode_history(history)
    assert features['pyramid'][-1].shape == (2, 24, 2, 3)
    features['bottleneck'].square().mean().backward()
    mixer = model.backbone.convlstm_mixers['3']
    assert mixer.gates.weight.grad is not None
    assert torch.count_nonzero(mixer.gates.weight.grad) > 0


def test_bottleneck_convlstm_replaces_weighted_fusion_at_same_scale():
    config = _config()
    config.temporal_unet_convlstm_scales = [3]
    model = EvolutionTemporalUNet_Model(config)
    assert '3' in model.backbone.convlstm_mixers
    assert '3' not in model.backbone.mixers


def test_temporal_unet_flow_respects_displacement_bound():
    model = EvolutionTemporalUNet_Model(_config())
    with torch.no_grad():
        model.motion_head.output.bias.fill_(20.0)
    result = model(torch.rand(1, 10, 1, 66, 70), return_aux=True)
    assert result['flow'].abs().max() <= 2.0


def test_temporal_unet_backbone_receives_gradient_after_head_update():
    model = EvolutionTemporalUNet_Model(_config(height=16, width=18))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = torch.rand(1, 4, 1, 16, 18)
    for _ in range(2):
        optimizer.zero_grad()
        result = model(history, return_aux=True)
        loss = (result['prediction'] - 0.5).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    gradients = [parameter.grad for parameter in model.backbone.parameters()
                 if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_temporal_unet_factorized_source_shapes_and_initial_state():
    baseline_config = _config()
    baseline_config.evolution_field_space = 'rain_rate'
    baseline = EvolutionTemporalUNet_Model(baseline_config)
    config = _config()
    config.evolution_field_space = 'rain_rate'
    config.evolution_use_source = True
    config.evolution_source_parameterization = 'factorized_regime'
    config.evolution_source_max_rain = 35.0
    model = EvolutionTemporalUNet_Model(config)
    model.load_state_dict(baseline.state_dict(), strict=False)
    history = torch.rand(1, 10, 1, 66, 70)
    expected = baseline(history, return_aux=True)
    result = model(history, return_aux=True)
    assert result['regime_probability'].shape == (1, 4, 3, 66, 70)
    assert result['growth_fraction'].shape == (1, 4, 1, 66, 70)
    assert result['decay_fraction'].shape == (1, 4, 1, 66, 70)
    assert result['source_rain'].shape == (1, 4, 1, 66, 70)
    torch.testing.assert_close(
        result['prediction'], expected['prediction'], atol=3e-3, rtol=0)


def test_temporal_unet_factorized_source_heads_receive_gradients():
    config = _config(height=16, width=18)
    config.evolution_field_space = 'rain_rate'
    config.evolution_use_source = True
    config.evolution_source_parameterization = 'factorized_regime'
    config.evolution_source_max_rain = 35.0
    model = EvolutionTemporalUNet_Model(config)
    with torch.no_grad():
        model.source_head.regime_head.bias.zero_()
        model.source_head.growth_head.bias.zero_()
        model.source_head.decay_head.bias.zero_()
    history = torch.rand(1, 4, 1, 16, 18)
    target = torch.rand(1, 4, 1, 16, 18)
    result = model(history, return_aux=True, teacher_forcing=target)
    result['evolved_rain'].sum().backward()
    assert torch.count_nonzero(
        model.source_head.regime_head.weight.grad) > 0
    assert torch.count_nonzero(
        model.source_head.growth_head.weight.grad) > 0
    assert torch.count_nonzero(
        model.source_head.decay_head.weight.grad) > 0
