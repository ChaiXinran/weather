import torch

from openstl.modules import EvolutionOperator, backward_warp, warp_field


def test_backward_warp_positive_dx_moves_content_right_exactly():
    field = torch.zeros(1, 1, 5, 6)
    field[0, 0, 2, 1] = 1.0
    flow = torch.zeros(1, 2, 5, 6)
    flow[:, 0] = 2.0
    warped = backward_warp(field, flow)
    expected = torch.zeros_like(field)
    expected[0, 0, 2, 3] = 1.0
    torch.testing.assert_close(warped, expected, atol=1e-6, rtol=0)


def test_evolution_operator_returns_diagnostic_fields():
    initial = torch.rand(2, 1, 4, 5)
    flow = torch.zeros(2, 3, 2, 4, 5)
    result = EvolutionOperator()(initial, flow)
    assert result['prediction'].shape == (2, 3, 1, 4, 5)
    assert result['advected'].shape == result['prediction'].shape
    assert result['source'] is None
    torch.testing.assert_close(result['prediction'][:, 0], initial)


def test_linear_z_warp_preserves_subpixel_peak_better_than_dbz_warp():
    field = torch.zeros(1, 1, 3, 4)
    field[0, 0, 1, 1] = 1.0
    flow = torch.zeros(1, 2, 3, 4)
    flow[:, 0] = 0.34
    dbz_warped = warp_field(field, flow, field_space='normalized_dbz')
    z_warped = warp_field(field, flow, field_space='linear_z')
    assert z_warped.max() > dbz_warped.max()


def test_stop_gradient_blocks_later_leads_from_earlier_flow():
    initial = torch.rand(1, 1, 4, 5)
    flow = torch.zeros(1, 2, 2, 4, 5, requires_grad=True)
    prediction = EvolutionOperator(stop_gradient=True)(initial, flow)['prediction']
    prediction[:, 1].sum().backward()
    torch.testing.assert_close(flow.grad[:, 0], torch.zeros_like(flow.grad[:, 0]))
    assert torch.count_nonzero(flow.grad[:, 1]) > 0
