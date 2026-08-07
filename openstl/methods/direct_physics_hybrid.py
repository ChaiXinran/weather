import torch
import torch.nn.functional as F

from openstl.models import DirectPhysicsHybrid_Model
from openstl.modules.evolution_operator import normalized_dbz_to_rain
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
        for head in (self.model.motion_gate_head, self.model.source_gate_head):
            for parameter in head.parameters():
                parameter.register_hook(lambda gradient: gradient * gate_lr_scale)
        print(f'Loaded {count} tensors from direct ConvLSTM checkpoint')

    def _build_model(self, **args):
        return DirectPhysicsHybrid_Model(self.hparams)

    @staticmethod
    def _flow_smoothness(flow):
        spatial = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
        spatial = spatial + (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
        temporal = flow.new_zeros(())
        if flow.shape[1] > 1:
            temporal = (flow[:, 1:] - flow[:, :-1]).abs().mean()
        return spatial + temporal

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
        flow_reg = self._flow_smoothness(result['residual_flow'])
        source_reg = result['source_rain'].abs().mean()
        aux_weight = float(
            self.hparams.hybrid_warmup_physics_weight if not blend_enabled
            else self.hparams.hybrid_physics_aux_weight)
        direct_rain = result['direct_rain']
        target_rain = normalized_dbz_to_rain(
            batch_y, self.hparams.radar_value_scale,
            self.hparams.zr_a, self.hparams.zr_b)
        physics_residual = result['source_candidate_rain'] - direct_rain
        target_residual = target_rain - direct_rain.detach()
        residual_aux = F.smooth_l1_loss(
            physics_residual / max(float(self.hparams.hybrid_max_source_rain), 1.0),
            target_residual / max(float(self.hparams.hybrid_max_source_rain), 1.0),
            beta=0.05)
        anchor_weight = float(self.hparams.hybrid_direct_anchor_weight)
        if blend_enabled:
            anchor_weight = float(self.hparams.get(
                'hybrid_direct_anchor_after_warmup', 0.02))
        total = (loss + aux_weight * aux
                 + float(self.hparams.get('hybrid_residual_aux_weight', 0.1))
                 * residual_aux
                 + anchor_weight * anchor
                 + float(self.hparams.hybrid_flow_regularization) * flow_reg
                 + float(self.hparams.hybrid_source_regularization) * source_reg
                 + float(self.hparams.get('hybrid_gate_regularization', 0.01))
                 * (result['motion_gate'].abs().mean()
                    + result['source_gate'].abs().mean()))

        # Teach each gate to activate only when its candidate improves the
        # preceding candidate on the current target. The target is detached,
        # so the gate cannot change the candidate it is supervising.
        temperature = float(self.hparams.get('hybrid_gate_temperature', 0.05))
        direct_error = (result['direct_prediction'] - batch_y).abs().mean(2, keepdim=True)
        motion_error = (result['motion_prediction'] - batch_y).abs().mean(2, keepdim=True)
        source_error = (result['source_prediction'] - batch_y).abs().mean(2, keepdim=True)
        motion_target = torch.sigmoid((direct_error - motion_error).detach() / max(temperature, 1e-6))
        source_target = torch.sigmoid((motion_error - source_error).detach() / max(temperature, 1e-6))
        gate_loss = F.binary_cross_entropy(
            result['raw_motion_gate'].clamp(1e-5, 1 - 1e-5), motion_target)
        gate_loss = gate_loss + F.binary_cross_entropy(
            result['raw_source_gate'].clamp(1e-5, 1 - 1e-5), source_target)
        total = total + float(self.hparams.get('hybrid_gate_supervision_weight', 0.1)) * gate_loss

        event_steps = min(10, result['fused_rain'].shape[1])
        direct_event = result['direct_rain'][:, :event_steps]
        fused_event = result['fused_rain'][:, :event_steps]
        target_event = target_rain[:, :event_steps]
        threshold = float(self.hparams.get('hybrid_event_threshold', 16.0))
        margin = float(self.hparams.get('hybrid_event_margin', 0.5))
        event_temperature = float(self.hparams.get(
            'hybrid_event_temperature', 1.0))
        miss_mask = (direct_event < threshold) & (target_event >= threshold)
        false_alarm_mask = ((direct_event >= threshold)
                            & (target_event < threshold))
        miss_penalty = F.softplus(
            (threshold + margin - fused_event) / event_temperature)
        false_alarm_penalty = F.softplus(
            (fused_event - (threshold - margin)) / event_temperature)
        miss_loss = (miss_penalty[miss_mask].mean()
                     if miss_mask.any() else fused_event.new_zeros(()))
        false_alarm_loss = (
            false_alarm_penalty[false_alarm_mask].mean()
            if false_alarm_mask.any() else fused_event.new_zeros(()))
        event_loss = miss_loss + false_alarm_loss
        total = total + float(self.hparams.get(
            'hybrid_event_recovery_weight', 0.0)) * event_loss
        self.log('train_loss', total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_fused_r2d', loss, on_epoch=True)
        self.log('train_physics_aux', aux, on_epoch=True)
        self.log('train_residual_aux', residual_aux, on_epoch=True)
        self.log('train_gate_supervision', gate_loss, on_epoch=True)
        self.log('train_event_miss16', miss_loss, on_epoch=True)
        self.log('train_event_fa16', false_alarm_loss, on_epoch=True)
        self.log('train_event_recovery16', event_loss, on_epoch=True)
        self.log('train_direct_anchor', anchor, on_epoch=True)
        self.log('train_blend_enabled', float(blend_enabled), on_epoch=True)
        self.log('train_blend_alpha_abs', result['blend_alpha'].abs().mean(), on_epoch=True)
        self.log('train_learned_alpha_abs', result['learned_blend_alpha'].abs().mean(), on_epoch=True)
        self.log('train_residual_flow_abs', flow_reg, on_epoch=True)
        self.log('train_source_rain_abs', source_reg, on_epoch=True)
        self.log('train_motion_gate_abs', result['motion_gate'].abs().mean(),
                 on_epoch=True)
        self.log('train_source_gate_abs', result['source_gate'].abs().mean(),
                 on_epoch=True)
        return total
