from types import SimpleNamespace

import torch
import torch.nn as nn

from openstl.methods.predrnn import PredRNN


class _RecordingCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction = None
        self.target = None

    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return (prediction - target).abs().mean()


class _DummyRecurrentModel(nn.Module):
    def __init__(self, generated):
        super().__init__()
        self.generated = generated

    def forward(self, frames, mask, return_loss=True):
        assert return_loss is False
        return self.generated.to(frames.device), None


class _TestPredRNN(PredRNN):
    @property
    def global_step(self):
        return 0


def test_training_step_uses_shared_criterion_for_future_frames():
    method = _TestPredRNN.__new__(_TestPredRNN)
    nn.Module.__init__(method)
    method._hparams = SimpleNamespace(
        patch_size=1,
        pre_seq_length=2,
        aft_seq_length=3,
        in_shape=(2, 1, 1, 1),
        device='cpu',
        reverse_scheduled_sampling=0,
        scheduled_sampling=0,
        total_length=5,
    )
    method.eta = 1.0
    generated = torch.arange(4.0).reshape(1, 4, 1, 1, 1)
    method.model = _DummyRecurrentModel(generated)
    method.criterion = _RecordingCriterion()
    method.log = lambda *args, **kwargs: None

    batch_x = torch.zeros(1, 2, 1, 1, 1)
    batch_y = torch.zeros(1, 3, 1, 1, 1)
    loss = method.training_step((batch_x, batch_y), 0)

    expected = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1, 1)
    torch.testing.assert_close(method.criterion.prediction, expected)
    assert method.criterion.target is batch_y
    torch.testing.assert_close(loss, expected.mean())
