import torch

from openstl.models import EvolutionConvLSTM_Model
from openstl.modules import backward_warp
from openstl.utils import print_log
from .base_method import Base_method


class EvolutionConvLSTM(Base_method):
    """R4 motion-only ConvLSTM with an explicit differentiable transport step."""

    def __init__(self, **args):
        super().__init__(**args)
        checkpoint = self.hparams.get('evolution_encoder_checkpoint', None)
        if isinstance(checkpoint, str) and checkpoint.lower() in ('none', 'null'):
            checkpoint = None
        if checkpoint:
            count = self.model.load_pretrained_encoder(checkpoint)
            print_log(f'Loaded {count} ConvLSTM encoder tensors from {checkpoint}; '
                      'the direct image head and optimizer state were not loaded.')

    def _build_model(self, **args):
        num_hidden = [int(value) for value in self.hparams.num_hidden.split(',')]
        return EvolutionConvLSTM_Model(len(num_hidden), num_hidden, self.hparams)

    def forward(self, batch_x, batch_y=None, **kwargs):
        # batch_y is deliberately ignored: the physical predictor only sees history.
        return self.model(batch_x)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        frozen = self.current_epoch < int(self.hparams.get(
            'evolution_freeze_encoder_epochs', 0))
        for parameter in self.model.cell_list.parameters():
            parameter.requires_grad_(not frozen)

    def configure_optimizers(self):
        encoder_lr = float(self.hparams.get('evolution_encoder_lr', self.hparams.lr))
        head_lr = float(self.hparams.get('evolution_head_lr', self.hparams.lr))
        optimizer = torch.optim.Adam([
            {'params': self.model.cell_list.parameters(), 'lr': encoder_lr},
            {'params': self.model.motion_head.parameters(), 'lr': head_lr},
        ], weight_decay=float(self.hparams.weight_decay))
        if self.hparams.sched == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=[encoder_lr, head_lr],
                total_steps=self.hparams.epoch * self.hparams.steps_per_epoch,
                final_div_factor=self.hparams.get('final_div_factor', 1e4))
            interval = 'step'
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.epoch,
                eta_min=float(self.hparams.get('min_lr', 1e-6)))
            interval = 'epoch'
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': interval}}

    @staticmethod
    def _spatial_smoothness(flow):
        dx = (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
        dy = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
        return dx + dy

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        result = self.model(batch_x, return_aux=True)
        forecast_loss = self.criterion(result['prediction'], batch_y)

        previous = torch.cat((batch_x[:, -1:], batch_y[:, :-1]), dim=1)
        transported = torch.stack([
            self.model.operator.warp(previous[:, step], result['flow'][:, step])
            for step in range(batch_y.shape[1])
        ], dim=1)
        tf_loss = torch.nn.functional.smooth_l1_loss(transported, batch_y)
        spatial_loss = self._spatial_smoothness(result['flow'])
        if result['flow'].shape[1] > 1:
            temporal_loss = (result['flow'][:, 1:] - result['flow'][:, :-1]).abs().mean()
        else:
            temporal_loss = result['flow'].new_zeros(())

        loss = (forecast_loss
                + float(self.hparams.get('evolution_tf_weight', 0.5)) * tf_loss
                + float(self.hparams.get('evolution_spatial_weight', 1e-3)) * spatial_loss
                + float(self.hparams.get('evolution_temporal_weight', 1e-3)) * temporal_loss)
        values = {'loss': loss, 'forecast': forecast_loss, 'transport': tf_loss,
                  'flow_spatial': spatial_loss, 'flow_temporal': temporal_loss}
        for name, value in values.items():
            self.log(f'train_{name}', value, on_step=name == 'loss', on_epoch=True,
                     prog_bar=name == 'loss')
        for name, value in getattr(self.criterion, 'last_components', {}).items():
            self.log(f'train_{name}', value, on_step=False, on_epoch=True)
        return loss
