import torch

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
