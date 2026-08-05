from types import SimpleNamespace

import torch

from openstl.models import ConvLSTM_Model, EvolutionConvLSTM_Model


def _config():
    return SimpleNamespace(
        in_shape=[3, 1, 8, 10], pre_seq_length=3, aft_seq_length=4,
        total_length=7, patch_size=2, filter_size=3, stride=1,
        layer_norm=0, evolution_head_channels=4, evolution_head_groups=2,
        evolution_max_displacement=2.0)


def test_evolution_model_uses_history_only_and_starts_as_persistence():
    model = EvolutionConvLSTM_Model(2, [4, 4], _config())
    history = torch.rand(2, 3, 1, 8, 10)
    result = model(history, return_aux=True)
    assert result['prediction'].shape == (2, 4, 1, 8, 10)
    assert result['flow'].shape == (2, 4, 2, 8, 10)
    torch.testing.assert_close(result['flow'], torch.zeros_like(result['flow']))
    for step in range(4):
        torch.testing.assert_close(result['prediction'][:, step], history[:, -1])


def test_evolution_flow_gate_starts_at_configured_scale():
    config = _config()
    config.evolution_use_flow_gate = True
    config.evolution_gate_initial = 0.5
    model = EvolutionConvLSTM_Model(2, [4, 4], config)
    history = torch.rand(2, 3, 1, 8, 10)
    result = model(history, return_aux=True)
    assert result['flow_gate'].shape == (2, 4, 1, 8, 10)
    torch.testing.assert_close(
        result['flow_gate'], torch.full_like(result['flow_gate'], 0.5))
    torch.testing.assert_close(result['flow'], result['raw_flow'] * 0.5)


def test_encoder_only_checkpoint_loading_excludes_direct_image_head(tmp_path):
    config = _config()
    baseline = ConvLSTM_Model(2, [4, 4], config)
    checkpoint = tmp_path / 'baseline.ckpt'
    torch.save({'state_dict': {
        **{f'model.{key}': value for key, value in baseline.state_dict().items()}
    }}, checkpoint)
    model = EvolutionConvLSTM_Model(2, [4, 4], config)
    count = model.load_pretrained_encoder(checkpoint)
    assert count == len(baseline.cell_list.state_dict())
    for key, value in baseline.cell_list.state_dict().items():
        torch.testing.assert_close(model.cell_list.state_dict()[key], value)
