import torch

from openstl.models.simvp_model import SimVP_Model


def test_simvp_direct_output_shape_and_gradient():
    model = SimVP_Model(
        in_shape=(10, 1, 16, 16),
        hid_S=4,
        hid_T=16,
        N_S=2,
        N_T=2,
        model_type='gSTA',
        direct_aft_seq_length=20,
    )
    inputs = torch.randn(1, 10, 1, 16, 16)
    outputs = model(inputs)
    assert outputs.shape == (1, 20, 1, 16, 16)
    outputs.mean().backward()
    assert torch.isfinite(model.latent_time_projection.weight.grad).all()
    assert torch.isfinite(model.skip_time_projection.weight.grad).all()
