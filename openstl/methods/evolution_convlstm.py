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
        motion_checkpoint = self.hparams.get('evolution_motion_checkpoint', None)
        if isinstance(motion_checkpoint, str) and motion_checkpoint.lower() in ('none', 'null'):
            motion_checkpoint = None
        if motion_checkpoint:
            count = self.model.load_pretrained_motion(motion_checkpoint)
            print_log(f'Loaded {count} encoder/motion tensors from {motion_checkpoint}; '
                      'the flow gate and optimizer state start fresh.')

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
        motion_frozen = self.current_epoch < int(self.hparams.get(
            'evolution_freeze_motion_epochs', 0) or 0)
        for parameter in self.model.motion_head.parameters():
            parameter.requires_grad_(not motion_frozen)

    def configure_optimizers(self):
        encoder_lr = float(self.hparams.get('evolution_encoder_lr', self.hparams.lr))
        head_lr = float(self.hparams.get('evolution_head_lr', self.hparams.lr))
        parameter_groups = [
            {'params': self.model.cell_list.parameters(), 'lr': encoder_lr},
            {'params': self.model.motion_head.parameters(), 'lr': head_lr},
        ]
        max_lrs = [encoder_lr, head_lr]
        if getattr(self.model, 'use_flow_gate', False):
            gate_lr = float(self.hparams.get('evolution_gate_lr') or head_lr)
            parameter_groups.append({
                'params': self.model.flow_gate_head.parameters(), 'lr': gate_lr})
            max_lrs.append(gate_lr)
        optimizer = torch.optim.Adam(
            parameter_groups, weight_decay=float(self.hparams.weight_decay))
        if self.hparams.sched == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lrs,
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

    def _gate_oracle_loss(self, previous, target, raw_flow, predicted_gate):
        """Supervise motion necessity using per-pixel teacher-forced scales."""
        candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
        with torch.no_grad():
            candidate_predictions = []
            for scale in candidates:
                candidate_predictions.append(torch.stack([
                    self.model.operator.warp(
                        previous[:, step], scale * raw_flow[:, step])
                    for step in range(target.shape[1])
                ], dim=1))
            candidate_rain = torch.stack([
                self._to_precipitation(value)
                for value in candidate_predictions], dim=0)
            target_rain = self._to_precipitation(target)
            errors = (candidate_rain - target_rain.unsqueeze(0)).abs()
            best_index = errors.argmin(dim=0)
            scale_values = target.new_tensor(candidates)
            oracle_gate = scale_values[best_index]
            previous_rain = self._to_precipitation(previous)
            event_rain = torch.maximum(previous_rain, target_rain)
            active_threshold = float(self.hparams.get(
                'evolution_gate_active_threshold') or 8.0)
            weights = (event_rain >= active_threshold).float()
            weights = weights + 2.0 * (event_rain >= 16.0).float()
            weights = weights + 4.0 * (event_rain >= 32.0).float()
        error = torch.nn.functional.smooth_l1_loss(
            predicted_gate, oracle_gate, reduction='none', beta=0.1)
        return ((error * weights).sum() / weights.sum().clamp_min(1.0),
                oracle_gate, weights)

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

        gate_supervision = result['flow'].new_zeros(())
        gate_target_mean = result['flow'].new_zeros(())
        if (getattr(self.model, 'use_flow_gate', False)
                and float(self.hparams.get(
                    'evolution_gate_supervision_weight') or 0.0) > 0):
            gate_supervision, oracle_gate, gate_weights = self._gate_oracle_loss(
                previous, batch_y, result['raw_flow'], result['flow_gate'])
            gate_target_mean = ((oracle_gate * gate_weights).sum()
                                / gate_weights.sum().clamp_min(1.0))

        if bool(self.hparams.get('evolution_gate_supervision_only', False)):
            loss = gate_supervision
        else:
            loss = (forecast_loss
                    + float(self.hparams.get('evolution_tf_weight', 0.5)) * tf_loss
                    + float(self.hparams.get('evolution_spatial_weight', 1e-3)) * spatial_loss
                    + float(self.hparams.get('evolution_temporal_weight', 1e-3)) * temporal_loss
                    + float(self.hparams.get(
                        'evolution_gate_supervision_weight') or 0.0) * gate_supervision)
        values = {'loss': loss, 'forecast': forecast_loss, 'transport': tf_loss,
                  'flow_spatial': spatial_loss, 'flow_temporal': temporal_loss}
        if getattr(self.model, 'use_flow_gate', False):
            values['gate_mean'] = result['flow_gate'].mean()
            values['gate_supervision'] = gate_supervision
            values['gate_target_mean'] = gate_target_mean
        for name, value in values.items():
            self.log(f'train_{name}', value, on_step=name == 'loss', on_epoch=True,
                     prog_bar=name == 'loss')
        for name, value in getattr(self.criterion, 'last_components', {}).items():
            self.log(f'train_{name}', value, on_step=False, on_epoch=True)
        return loss
