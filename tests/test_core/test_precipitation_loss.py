import torch

from openstl.core.precipitation_loss import PrecipitationR2Loss


def test_r2_loss_has_finite_gradient():
    target = torch.rand(2, 20, 1, 8, 8)
    pred = torch.rand_like(target, requires_grad=True)
    loss = PrecipitationR2Loss()(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()


def test_segmented_soft_csi_handles_active_and_empty_samples():
    target = torch.zeros(3, 20, 1, 8, 8)
    target[0, :, :, 2:4, 2:4] = 0.95
    pred = torch.rand_like(target, requires_grad=True)
    loss_fn = PrecipitationR2Loss(
        soft_csi_mode='sample_period',
        segmented_soft_csi_weights=[
            [0.0018, 0.0009],
            [0.00216, 0.00108],
        ],
        empty_event_penalty=0.1,
    )
    loss = loss_fn(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()
    assert {
        'soft_csi_16_0_1h',
        'soft_csi_32_0_1h',
        'soft_csi_16_1_2h',
        'soft_csi_32_1_2h',
    }.issubset(loss_fn.last_components)
