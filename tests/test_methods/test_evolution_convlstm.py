import torch
from types import MethodType
from types import SimpleNamespace

from openstl.methods.evolution_convlstm import EvolutionConvLSTM


def test_pixel_weighted_state_loss_uses_one_shared_denominator():
    error = torch.tensor([[[[[100.0, 1.0, 2.0, 3.0]]]]])
    active = torch.tensor([[[[[False, True, True, True]]]]])
    mask16 = torch.tensor([[[[[False, False, True, True]]]]])
    mask32 = torch.tensor([[[[[False, False, False, True]]]]])
    loss, weights = EvolutionConvLSTM._pixel_weighted_state_loss(
        error, active, mask16, mask32)
    torch.testing.assert_close(
        weights, torch.tensor([[[[[0.0, 1.0, 2.0, 3.0]]]]]))
    torch.testing.assert_close(loss, torch.tensor(14.0 / 6.0))


def test_pixel_weighted_state_loss_caps_nested_mask_weight():
    error = torch.ones(1, 1, 1, 1, 1)
    mask = torch.ones_like(error, dtype=torch.bool)
    _, weights = EvolutionConvLSTM._pixel_weighted_state_loss(
        error, mask, mask, mask, increment16=4.0, increment32=8.0,
        max_weight=3.0)
    torch.testing.assert_close(weights, torch.full_like(error, 3.0))


def test_physical_region_masks_and_regime_labels_focus_interior():
    method = SimpleNamespace(
        hparams={'evolution_source_active_threshold': 0.1,
                 'evolution_regime_delta': 0.5},
        _erode_mask=EvolutionConvLSTM._erode_mask,
        _dilate_mask=EvolutionConvLSTM._dilate_mask)
    method._build_physical_region_masks = MethodType(
        EvolutionConvLSTM._build_physical_region_masks, method)
    method._build_regime_labels = MethodType(
        EvolutionConvLSTM._build_regime_labels, method)
    advected = torch.zeros(1, 1, 1, 5, 5)
    target = torch.zeros_like(advected)
    advected[..., 1:4, 1:4] = 10.0
    target[..., 1:4, 1:4] = 10.0
    target[..., 2, 2] = 11.0
    target[..., 0, 0] = 1.0
    masks = method._build_physical_region_masks(advected, target)
    assert masks['interior'][..., 2, 2]
    assert masks['birth'][..., 0, 0]
    labels = method._build_regime_labels(target - advected, masks['interior'])
    assert labels[..., 2, 2] == 0
    assert labels[..., 1, 1] == -100
    assert labels[..., 0, 0] == -100


def test_balanced_regime_loss_is_finite_for_imbalanced_labels():
    logits = torch.zeros(1, 1, 3, 2, 3)
    labels = torch.tensor([[[[[0, 1, 1], [1, 1, 2]]]]])
    loss = EvolutionConvLSTM._balanced_regime_loss(logits, labels)
    assert torch.isfinite(loss)
    assert loss > 0
