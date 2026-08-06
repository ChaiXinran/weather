import torch

from openstl.modules import (EvolutionOperator, backward_warp,
                             normalized_dbz_to_rain, rain_to_normalized_dbz,
                             warp_field)


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


def test_rain_rate_source_is_added_in_mm_per_hour_and_clamped_nonnegative():
    initial_rain = torch.full((1, 1, 3, 4), 10.0)
    initial = rain_to_normalized_dbz(initial_rain)
    flow = torch.zeros(1, 2, 2, 3, 4)
    source = torch.stack((
        torch.full_like(initial, 5.0),
        torch.full_like(initial, -20.0),
    ), dim=1)
    result = EvolutionOperator(field_space='rain_rate')(
        initial, flow, source=source)
    torch.testing.assert_close(
        result['advected_rain'][:, 0], initial_rain, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        result['evolved_rain'][:, 0], torch.full_like(initial_rain, 15.0),
        atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        result['evolved_rain'][:, 1], torch.zeros_like(initial_rain),
        atol=1e-4, rtol=0)
    torch.testing.assert_close(
        normalized_dbz_to_rain(result['prediction'][:, 0]),
        torch.full_like(initial_rain, 15.0), atol=1e-4, rtol=1e-4)


def test_no_source_keeps_existing_rain_rate_path_and_diagnostics_empty():
    initial = torch.rand(2, 1, 4, 5)
    flow = torch.randn(2, 3, 2, 4, 5) * 0.1
    operator = EvolutionOperator(field_space='rain_rate')
    expected = []
    current = initial
    for step in range(flow.shape[1]):
        current = operator.warp(current, flow[:, step])
        expected.append(current)
    result = operator(initial, flow)
    torch.testing.assert_close(result['prediction'], torch.stack(expected, dim=1))
    assert result['advected_rain'] is None
    assert result['source_rain'] is None
    assert result['evolved_rain'] is None


def test_physical_source_rejects_non_rain_rate_operator():
    initial = torch.rand(1, 1, 3, 4)
    flow = torch.zeros(1, 1, 2, 3, 4)
    source = torch.zeros(1, 1, 1, 3, 4)
    try:
        EvolutionOperator(field_space='normalized_dbz')(
            initial, flow, source=source)
    except ValueError as error:
        assert 'rain_rate' in str(error)
    else:
        raise AssertionError('Expected a rain-rate source unit validation error')


def test_bounded_source_respects_sink_and_representable_upper_limit():
    operator = EvolutionOperator(field_space='rain_rate')
    advected = torch.tensor([[[[0.0, 10.0, 40.0, operator.max_rain]]]])
    logits = torch.tensor([[[[-20.0, -20.0, 20.0, 20.0]]]])
    source, tendency, capacity = operator.bounded_source(
        advected, logits, source_max_rain=35.0)
    evolved = advected + source
    assert torch.all(source >= -advected)
    assert torch.all(source <= capacity)
    assert torch.all(evolved >= 0.0)
    assert torch.all(evolved <= operator.max_rain + 1e-5)
    torch.testing.assert_close(evolved[0, 0, 0, :2], torch.zeros(2))
    torch.testing.assert_close(evolved[0, 0, 0, -1],
                               torch.tensor(operator.max_rain))
    assert tendency.min() < 0 and tendency.max() > 0


def test_bounded_source_has_gradients_on_both_signed_branches():
    operator = EvolutionOperator(field_space='rain_rate')
    advected = torch.full((1, 1, 2, 2), 10.0)
    logits = torch.tensor([[[[-0.2, -0.1], [0.1, 0.2]]]],
                          requires_grad=True)
    source, _, _ = operator.bounded_source(advected, logits, 35.0)
    source.sum().backward()
    assert torch.all(torch.isfinite(logits.grad))
    assert torch.all(logits.grad != 0)


def test_factorized_regime_probabilities_sum_to_one():
    operator = EvolutionOperator(field_space='rain_rate')
    initial = rain_to_normalized_dbz(torch.full((2, 1, 4, 5), 10.0))
    flow = torch.zeros(2, 2, 4, 5)
    logits = torch.randn(2, 3, 4, 5)
    result = operator.evolve_factorized_step(
        initial, flow, logits, torch.zeros_like(initial),
        torch.zeros_like(initial), torch.full_like(initial, 5.0))
    torch.testing.assert_close(
        result['regime_probability'].sum(dim=1),
        torch.ones(2, 4, 5))


def test_factorized_growth_and_decay_states_are_physically_bounded():
    operator = EvolutionOperator(field_space='rain_rate')
    initial_rain = torch.full((1, 1, 3, 4), 10.0)
    initial = rain_to_normalized_dbz(initial_rain)
    flow = torch.zeros(1, 2, 3, 4)
    regime_logits = torch.zeros(1, 3, 3, 4)
    result = operator.evolve_factorized_step(
        initial, flow, regime_logits,
        torch.full_like(initial, 20.0),
        torch.full_like(initial, 20.0),
        torch.full_like(initial, 7.0))
    assert torch.all(result['growth_state'] >= result['advected_rain'])
    assert torch.all(result['decay_state'] <= result['advected_rain'])
    assert torch.all(result['decay_state'] >= 0.0)
    assert torch.all(result['evolved_rain'] <= operator.max_rain + 1e-5)


def test_factorized_source_is_zero_outside_source_mask():
    operator = EvolutionOperator(field_space='rain_rate')
    initial = rain_to_normalized_dbz(torch.full((1, 1, 2, 3), 10.0))
    flow = torch.zeros(1, 2, 2, 3)
    regime_logits = torch.zeros(1, 3, 2, 3)
    regime_logits[:, 0] = 20.0
    mask = torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]])
    result = operator.evolve_factorized_step(
        initial, flow, regime_logits,
        torch.full_like(initial, 20.0),
        torch.zeros_like(initial), torch.full_like(initial, 5.0),
        source_mask=mask)
    torch.testing.assert_close(
        result['net_source'] * (1.0 - mask),
        torch.zeros_like(result['net_source']))
