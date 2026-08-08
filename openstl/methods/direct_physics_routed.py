import torch
import torch.nn.functional as F

from openstl.models import DirectPhysicsRouted_Model
from openstl.modules.evolution_operator import normalized_dbz_to_rain
from openstl.modules.v3a_routing import decode_packed_routing_target
from .base_method import Base_method


class DirectPhysicsRouted(Base_method):
    """Staged V3a training for preserve/motion/decay experts and router."""

    def __init__(self, **args):
        super().__init__(**args)
        restoring_v3a = bool(
            self.hparams.get('init_from_ckpt') or self.hparams.get('ckpt_path'))
        count = (0 if restoring_v3a else
                 self.model.load_direct_checkpoint(
                     self.hparams.hybrid_direct_checkpoint))
        self.model.freeze_direct()
        # A staged continuation will immediately load a complete V3a state in
        # BaseExperiment, so it must not require the original V2 path to exist
        # on the current machine/server.
        if restoring_v3a:
            v2_count, skipped = 0, []
        else:
            v2_count, skipped = self.model.load_v2_correction_checkpoint(
                self.hparams.get('v3a_init_correction_checkpoint', ''))
        self.stage = str(self.hparams.get('v3a_stage', 'joint')).lower()
        if self.stage not in {'expert', 'router', 'joint'}:
            raise ValueError('v3a_stage must be expert, router, or joint')
        self._apply_stage_freezing()
        print(
            f'V3a stage={self.stage}; direct tensors={count}; '
            f'compatible V2 correction tensors={v2_count}; skipped={len(skipped)}')

    def _build_model(self, **args):
        return DirectPhysicsRouted_Model(self.hparams)

    def _apply_stage_freezing(self):
        correction_modules = (
            self.model.features, self.model.decoder, self.model.lead_embedding,
            self.model.head, self.model.flow_head, self.model.decay_head)
        if self.stage == 'router':
            for module in correction_modules:
                module.requires_grad_(False)
            self.model.router_trunk.requires_grad_(True)
            self.model.router_head.requires_grad_(True)
        elif self.stage == 'expert':
            for module in correction_modules:
                module.requires_grad_(True)
            self.model.router_trunk.requires_grad_(False)
            self.model.router_head.requires_grad_(False)
        else:
            for module in correction_modules:
                module.requires_grad_(True)
            self.model.router_trunk.requires_grad_(True)
            self.model.router_head.requires_grad_(True)

    @staticmethod
    def _unpack(batch, require_routing=False):
        if len(batch) == 3:
            return batch
        if require_routing:
            raise RuntimeError(
                'V3a training requires routing_cache_path with train labels')
        return batch[0], batch[1], None

    @staticmethod
    def _masked_mean(values, weight):
        while weight.ndim < values.ndim:
            weight = weight.unsqueeze(2)
        weight = weight.to(values.dtype)
        return (values * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _flow_smoothness(flow):
        spatial = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
        spatial += (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
        temporal = ((flow[:, 1:] - flow[:, :-1]).abs().mean()
                    if flow.shape[1] > 1 else flow.new_zeros(()))
        return spatial + temporal

    def forward(self, batch_x, batch_y=None, **kwargs):
        return self.model(batch_x)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, packed = self._unpack(batch, require_routing=True)
        route_target, valid = decode_packed_routing_target(
            packed, self.hparams.v3a_route_weight16,
            self.hparams.v3a_route_weight32)
        route_target = route_target.to(batch_x.device)
        valid = valid.to(batch_x.device)
        result = self.model(batch_x, return_aux=True)
        target_rain = normalized_dbz_to_rain(
            batch_y, self.hparams.radar_value_scale,
            self.hparams.zr_a, self.hparams.zr_b)
        rain_scale = max(float(self.hparams.v3a_rain_loss_scale), 1.0)

        motion_error = F.smooth_l1_loss(
            result['motion_rain'] / rain_scale,
            target_rain / rain_scale, beta=0.05, reduction='none')
        motion_loss = self._masked_mean(motion_error, route_target[:, :, 1])
        decay_target = ((result['direct_rain'].detach() - target_rain)
                        / result['direct_rain'].detach().clamp_min(1e-3))
        decay_target = decay_target.clamp(0.0, 1.0)
        decay_error = F.smooth_l1_loss(
            result['decay_fraction'], decay_target,
            beta=0.05, reduction='none')
        decay_loss = self._masked_mean(decay_error, route_target[:, :, 2])
        log_probability = torch.log_softmax(
            result['route_logits'] / max(
                float(self.hparams.v3a_router_temperature), 1e-4), dim=2)
        route_error = -(route_target * log_probability).sum(dim=2)
        route_loss = self._masked_mean(route_error, valid)
        preserve_error = F.smooth_l1_loss(
            result['prediction'], result['direct_prediction'].detach(),
            beta=0.02, reduction='none')
        preserve_loss = self._masked_mean(
            preserve_error, route_target[:, :, 0])
        fused_loss = self.criterion(result['prediction'], batch_y)
        flow_loss = self._flow_smoothness(result['residual_flow'])

        if self.stage == 'expert':
            total = (float(self.hparams.v3a_motion_loss_weight) * motion_loss
                     + float(self.hparams.v3a_decay_loss_weight) * decay_loss
                     + float(self.hparams.v3a_flow_regularization) * flow_loss)
        elif self.stage == 'router':
            total = (route_loss
                     + float(self.hparams.v3a_router_fused_weight) * fused_loss)
        else:
            total = (fused_loss
                     + float(self.hparams.v3a_route_loss_weight) * route_loss
                     + float(self.hparams.v3a_motion_loss_weight) * motion_loss
                     + float(self.hparams.v3a_decay_loss_weight) * decay_loss
                     + float(self.hparams.v3a_preserve_loss_weight) * preserve_loss
                     + float(self.hparams.v3a_flow_regularization) * flow_loss)
        self.log('train_loss', total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_v3a_fused_r2d', fused_loss, on_epoch=True)
        self.log('train_v3a_route', route_loss, on_epoch=True)
        self.log('train_v3a_motion', motion_loss, on_epoch=True)
        self.log('train_v3a_decay', decay_loss, on_epoch=True)
        self.log('train_v3a_preserve', preserve_loss, on_epoch=True)
        self.log('train_v3a_flow_smooth', flow_loss, on_epoch=True)
        for index, name in enumerate(('preserve', 'motion', 'decay')):
            self.log(
                f'train_route_{name}',
                result['route_probability'][:, :, index].mean(),
                on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, _ = self._unpack(batch)
        prediction = self(batch_x, batch_y)
        loss = self.validation_criterion(prediction, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self._update_val_precipitation(prediction, batch_y)
        return loss

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            batch = batch[:2]
        return super().test_step(batch, batch_idx)
