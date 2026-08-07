import torch
import torch.nn.functional as F

from openstl.models import DirectPhysicsHybrid_Model
from .base_method import Base_method


class DirectPhysicsHybrid(Base_method):
    def __init__(self, **args):
        super().__init__(**args)
        count = self.model.load_direct_checkpoint(
            self.hparams.hybrid_direct_checkpoint)
        if self.hparams.get('hybrid_freeze_direct', True):
            self.model.freeze_direct()
        gate_lr_scale = float(self.hparams.get('hybrid_gate_lr_scale', 1.0))
        if not 0.0 < gate_lr_scale <= 1.0:
            raise ValueError('hybrid_gate_lr_scale must be in (0, 1]')
        self.model.blend_logit.register_hook(
            lambda gradient: gradient * gate_lr_scale)
        print(f'Loaded {count} tensors from direct ConvLSTM checkpoint')

    def _build_model(self, **args):
        return DirectPhysicsHybrid_Model(self.hparams)

    def forward(self, batch_x, batch_y=None, **kwargs):
        # Warm-up is a training-only policy. Validation and deployment must
        # always use the learned blend stored in the checkpoint.
        blend_enabled = (not self.training or
                         self.current_epoch >= int(self.hparams.get(
                             'hybrid_blend_warmup_epochs', 0)))
        return self.model(batch_x, blend_enabled=blend_enabled)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        warmup_epochs = int(
            self.hparams.get('hybrid_blend_warmup_epochs', 0))
        blend_enabled = self.current_epoch >= warmup_epochs
        result = self.model(
            batch_x, return_aux=True, blend_enabled=blend_enabled)
        loss = self.criterion(result['prediction'], batch_y)
        # R2d, rather than pixel-only SmoothL1, teaches the warm-up branch to
        # preserve threshold events before it is allowed to alter the baseline.
        aux = self.criterion(result['physics_prediction'], batch_y)
        anchor = F.smooth_l1_loss(
            result['prediction'], result['direct_prediction'].detach(),
            beta=0.02)
        flow_reg = result['residual_flow'].abs().mean()
        source_reg = result['source_rain'].abs().mean()
        aux_weight = float(
            self.hparams.hybrid_warmup_physics_weight if not blend_enabled
            else self.hparams.hybrid_physics_aux_weight)
        total = (loss + aux_weight * aux
                 + float(self.hparams.hybrid_direct_anchor_weight) * anchor
                 + float(self.hparams.hybrid_flow_regularization) * flow_reg
                 + float(self.hparams.hybrid_source_regularization) * source_reg
                 + float(self.hparams.hybrid_alpha_regularization)
                 * result['learned_blend_alpha'].abs().mean())
        self.log('train_loss', total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_fused_r2d', loss, on_epoch=True)
        self.log('train_physics_aux', aux, on_epoch=True)
        self.log('train_direct_anchor', anchor, on_epoch=True)
        self.log('train_blend_enabled', float(blend_enabled), on_epoch=True)
        self.log('train_blend_alpha_abs', result['blend_alpha'].abs().mean(), on_epoch=True)
        self.log('train_learned_alpha_abs', result['learned_blend_alpha'].abs().mean(), on_epoch=True)
        self.log('train_residual_flow_abs', flow_reg, on_epoch=True)
        self.log('train_source_rain_abs', source_reg, on_epoch=True)
        return total
