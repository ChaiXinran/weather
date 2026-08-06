import torch

from openstl.models import EvolutionConvLSTM_Model
from openstl.modules import backward_warp, normalized_dbz_to_rain
from openstl.utils import print_log
from .base_method import Base_method


class EvolutionPhysicsBase(Base_method):
    """Shared training and diagnostics for explicit physical evolution."""

    def __init__(self, **args):
        super().__init__(**args)
        checkpoint = self.hparams.get('evolution_encoder_checkpoint', None)
        if isinstance(checkpoint, str) and checkpoint.lower() in ('none', 'null'):
            checkpoint = None
        if checkpoint:
            count = self.model.load_pretrained_encoder(checkpoint)
            print_log(f'Loaded {count} evolution encoder tensors from '
                      f'{checkpoint}; optimizer state was not loaded.')
        motion_checkpoint = self.hparams.get('evolution_motion_checkpoint', None)
        if isinstance(motion_checkpoint, str) and motion_checkpoint.lower() in ('none', 'null'):
            motion_checkpoint = None
        if motion_checkpoint:
            count = self.model.load_pretrained_motion(motion_checkpoint)
            print_log(f'Loaded {count} encoder/motion tensors from '
                      f'{motion_checkpoint}; optimizer state starts fresh.')
        source_checkpoint = self.hparams.get(
            'evolution_source_checkpoint', None)
        if isinstance(source_checkpoint, str) and source_checkpoint.lower() in (
                'none', 'null'):
            source_checkpoint = None
        if source_checkpoint:
            count = self.model.load_pretrained_source(source_checkpoint)
            print_log(
                f'Loaded {count} compatible source tensors from '
                f'{source_checkpoint}; newly added mechanism heads start fresh.')

    def _build_model(self, **args):
        raise NotImplementedError

    def forward(self, batch_x, batch_y=None, **kwargs):
        # batch_y is deliberately ignored: the physical predictor only sees history.
        return self.model(batch_x)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        source_only = bool(self.hparams.get('evolution_source_only', False))
        frozen = self.current_epoch < int(self.hparams.get(
            'evolution_freeze_encoder_epochs', 0))
        for parameter in self.model.backbone_parameters():
            parameter.requires_grad_(not (source_only or frozen))
        motion_frozen = self.current_epoch < int(self.hparams.get(
            'evolution_freeze_motion_epochs', 0) or 0)
        for parameter in self.model.motion_parameters():
            parameter.requires_grad_(not (source_only or motion_frozen))

    def configure_optimizers(self):
        source_only = bool(self.hparams.get('evolution_source_only', False))
        if source_only:
            if not getattr(self.model, 'use_source', False):
                raise ValueError('evolution_source_only requires a source model')
            source_lr = float(
                self.hparams.get('evolution_source_lr') or self.hparams.lr)
            parameter_groups = [
                {'params': self.model.source_parameters(), 'lr': source_lr}]
            max_lrs = [source_lr]
        else:
            encoder_lr = float(
                self.hparams.get('evolution_encoder_lr') or self.hparams.lr)
            head_lr = float(
                self.hparams.get('evolution_head_lr') or self.hparams.lr)
            parameter_groups = [
                {'params': self.model.backbone_parameters(), 'lr': encoder_lr},
                {'params': self.model.motion_parameters(), 'lr': head_lr},
            ]
            max_lrs = [encoder_lr, head_lr]
        if getattr(self.model, 'use_flow_gate', False) and not source_only:
            gate_lr = float(self.hparams.get('evolution_gate_lr') or head_lr)
            parameter_groups.append({
                'params': self.model.flow_gate_head.parameters(),
                'lr': gate_lr})
            max_lrs.append(gate_lr)
        if getattr(self.model, 'use_source', False) and not source_only:
            source_lr = float(self.hparams.get('evolution_source_lr') or head_lr)
            parameter_groups.append({
                'params': self.model.source_parameters(), 'lr': source_lr})
            max_lrs.append(source_lr)
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

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _erode_mask(mask, radius=1):
        if radius <= 0:
            return mask
        original_shape = mask.shape
        flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
        kernel = 2 * int(radius) + 1
        eroded = -torch.nn.functional.max_pool2d(
            -flat, kernel_size=kernel, stride=1, padding=radius)
        return (eroded > 0.5).reshape(original_shape)

    @staticmethod
    def _dilate_mask(mask, radius=1):
        if radius <= 0:
            return mask
        original_shape = mask.shape
        flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
        kernel = 2 * int(radius) + 1
        dilated = torch.nn.functional.max_pool2d(
            flat, kernel_size=kernel, stride=1, padding=radius)
        return (dilated > 0.5).reshape(original_shape)

    def _build_physical_region_masks(self, advected_rain, target_rain):
        threshold = float(self.hparams.get(
            'evolution_source_active_threshold', 0.1))
        advected_active = advected_rain >= threshold
        target_active = target_rain >= threshold
        interior = self._erode_mask(advected_active & target_active, radius=1)
        birth = (~advected_active) & target_active
        death = advected_active & (~target_active)
        dilated_union = self._dilate_mask(advected_active | target_active, radius=1)
        edge = dilated_union & ~(interior | birth | death)
        clear = ~(interior | edge | birth | death)
        return {
            'interior': interior,
            'edge': edge,
            'birth': birth,
            'death': death,
            'clear': clear,
        }

    def _build_regime_labels(self, oracle_source, interior):
        delta = float(self.hparams.get('evolution_regime_delta', 0.5))
        labels = torch.full(
            oracle_source.shape, -100, dtype=torch.long,
            device=oracle_source.device)
        labels = torch.where(
            interior & (oracle_source > delta),
            torch.zeros_like(labels), labels)
        labels = torch.where(
            interior & (oracle_source.abs() <= delta),
            torch.ones_like(labels), labels)
        labels = torch.where(
            interior & (oracle_source < -delta),
            torch.full_like(labels, 2), labels)
        return labels

    @staticmethod
    def _balanced_regime_loss(regime_logits, labels):
        logits = regime_logits.permute(0, 1, 3, 4, 2).reshape(-1, 3)
        flat_labels = labels.reshape(-1)
        valid = flat_labels != -100
        if not torch.any(valid):
            return logits.sum() * 0.0
        valid_labels = flat_labels[valid]
        counts = torch.bincount(valid_labels, minlength=3).to(logits.dtype)
        weights = counts.sum() / (3.0 * counts.clamp_min(1.0))
        weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
        return torch.nn.functional.cross_entropy(
            logits[valid], valid_labels, weight=weights)

    @staticmethod
    def _pixel_weighted_state_loss(error, active, mask16, mask32,
                                   increment16=1.0, increment32=1.0,
                                   max_weight=3.0):
        weights = active.to(error.dtype)
        weights = weights + float(increment16) * mask16.to(error.dtype)
        weights = weights + float(increment32) * mask32.to(error.dtype)
        weights = weights.clamp_max(float(max_weight))
        loss = (error * weights).sum() / weights.sum().clamp_min(1.0)
        return loss, weights

    def _factorized_source_terms(self, result, batch_y):
        batch_y = batch_y[:, :result['prediction'].shape[1]]
        operator = self.model.operator
        target_rain = normalized_dbz_to_rain(
            batch_y, value_scale=operator.value_scale,
            zr_a=operator.zr_a, zr_b=operator.zr_b)
        advected_rain = result['advected_rain'].detach()
        oracle_source = target_rain - advected_rain
        masks = self._build_physical_region_masks(advected_rain, target_rain)
        interior = masks['interior']
        labels = self._build_regime_labels(oracle_source, interior)

        regime_loss = self._balanced_regime_loss(result['regime_logits'], labels)
        growth_mask = labels == 0
        steady_mask = labels == 1
        decay_mask = labels == 2
        growth_target = (
            oracle_source / result['positive_capacity'].detach().clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        decay_target = (
            -oracle_source / advected_rain.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        growth_error = torch.nn.functional.smooth_l1_loss(
            result['growth_fraction'], growth_target, reduction='none',
            beta=float(self.hparams.get('evolution_magnitude_huber_beta', 0.05)))
        decay_error = torch.nn.functional.smooth_l1_loss(
            result['decay_fraction'], decay_target, reduction='none',
            beta=float(self.hparams.get('evolution_magnitude_huber_beta', 0.05)))
        magnitude_loss = (
            self._masked_mean(growth_error, growth_mask)
            + self._masked_mean(decay_error, decay_mask))

        masked_source = result['net_source'] * interior.to(result['net_source'].dtype)
        masked_evolved_rain = (result['advected_rain'] + masked_source).clamp_min(0.0)
        state_error = torch.nn.functional.smooth_l1_loss(
            masked_evolved_rain, target_rain, reduction='none',
            beta=float(self.hparams.get('evolution_state_huber_beta', 1.0)))
        event_rain = torch.maximum(advected_rain, target_rain)
        state_active = interior & (event_rain >= float(self.hparams.get(
            'evolution_source_active_threshold', 0.1)))
        state_loss, pixel_weights = self._pixel_weighted_state_loss(
            state_error, state_active, state_active & (event_rain >= 16.0),
            state_active & (event_rain >= 32.0),
            increment16=self.hparams.get('evolution_pixel_16_increment', 1.0),
            increment32=self.hparams.get('evolution_pixel_32_increment', 1.0),
            max_weight=self.hparams.get('evolution_pixel_max_weight', 3.0))
        loss = (float(self.hparams.get('evolution_regime_loss_weight', 1.0))
                * regime_loss
                + float(self.hparams.get('evolution_magnitude_loss_weight', 1.0))
                * magnitude_loss
                + float(self.hparams.get('evolution_state_loss_weight', 0.2))
                * state_loss)
        if 'edge_flow' in result:
            edge_error = torch.nn.functional.smooth_l1_loss(
                result['advected_rain'], target_rain, reduction='none',
                beta=float(self.hparams.get('evolution_edge_huber_beta', 1.0)))
            edge_loss = self._masked_mean(edge_error, masks['edge'])
            edge_smooth = self._spatial_smoothness(result['edge_flow'])
            loss = (loss
                    + float(self.hparams.get('evolution_edge_loss_weight', 0.2))
                    * edge_loss
                    + float(self.hparams.get('evolution_edge_smooth_weight', 0.01))
                    * edge_smooth)
        else:
            edge_loss = loss.new_zeros(())
            edge_smooth = loss.new_zeros(())

        with torch.no_grad():
            predicted_class = result['regime_probability'].argmax(dim=2)
            label_2d = labels.squeeze(2)
            class_masks = [label_2d == index for index in range(3)]
            pred_masks = [predicted_class == index for index in range(3)]
            f1_values = []
            precision_values = []
            recall_values = []
            eps = result['net_source'].new_tensor(1e-6)
            for index in range(3):
                tp = (pred_masks[index] & class_masks[index]).sum().to(eps.dtype)
                fp = (pred_masks[index] & ~class_masks[index]
                      & (label_2d != -100)).sum().to(eps.dtype)
                fn = (~pred_masks[index] & class_masks[index]).sum().to(eps.dtype)
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                precision_values.append(precision)
                recall_values.append(recall)
                f1_values.append(2.0 * precision * recall / (
                    precision + recall + eps))
            predicted_growth = masked_source.clamp_min(0.0)
            predicted_decay = (-masked_source).clamp_min(0.0)
            oracle_growth = oracle_source.clamp_min(0.0)
            oracle_decay = (-oracle_source).clamp_min(0.0)
            values = {
                'loss': loss,
                'regime_loss': regime_loss,
                'magnitude_loss': magnitude_loss,
                'state_loss': state_loss,
                'interior_state_mae': self._masked_mean(
                    (masked_evolved_rain - target_rain).abs(), interior),
                'regime_macro_f1': torch.stack(f1_values).mean(),
                'growth_precision': precision_values[0],
                'growth_recall': recall_values[0],
                'decay_precision': precision_values[2],
                'decay_recall': recall_values[2],
                'growth_source_scale_ratio': (
                    self._masked_mean(predicted_growth, growth_mask)
                    / (self._masked_mean(oracle_growth, growth_mask) + eps)),
                'decay_source_scale_ratio': (
                    self._masked_mean(predicted_decay, decay_mask)
                    / (self._masked_mean(oracle_decay, decay_mask) + eps)),
                'interior_fraction': interior.float().mean(),
                'growth_fraction': growth_mask.float().mean(),
                'steady_fraction': steady_mask.float().mean(),
                'decay_fraction': decay_mask.float().mean(),
                'edge_source_abs': self._masked_mean(
                    result['net_source'].abs(), masks['edge']),
                'birth_source_abs': self._masked_mean(
                    result['net_source'].abs(), masks['birth']),
                'clear_source_abs': self._masked_mean(
                    result['net_source'].abs(), masks['clear']),
                'pixel_weight_mean': self._masked_mean(pixel_weights, state_active),
                'edge_transport_loss': edge_loss,
                'edge_flow_abs': (result['edge_flow'].abs().mean()
                                  if 'edge_flow' in result else edge_loss),
            }
        values.update({
            'loss': loss,
            'regime_loss': regime_loss,
            'magnitude_loss': magnitude_loss,
            'state_loss': state_loss,
            'edge_transport_loss': edge_loss,
            'edge_smooth_loss': edge_smooth,
        })
        return values

    def _bounded_state_terms(self, result, batch_y):
        batch_y = batch_y[:, :result['prediction'].shape[1]]
        operator = self.model.operator
        target_rain = normalized_dbz_to_rain(
            batch_y, value_scale=operator.value_scale,
            zr_a=operator.zr_a, zr_b=operator.zr_b)
        error = torch.nn.functional.smooth_l1_loss(
            result['evolved_rain'], target_rain, reduction='none',
            beta=float(self.hparams.get('evolution_state_huber_beta', 1.0)))
        event_rain = torch.maximum(result['advected_rain'].detach(), target_rain)
        active = event_rain >= float(self.hparams.get(
            'evolution_source_active_threshold', 0.1))
        mask16 = event_rain >= 16.0
        mask32 = event_rain >= 32.0
        losses = {
            'state_all': error.mean(),
            'state_active': self._masked_mean(error, active),
            'state_16': self._masked_mean(error, mask16),
            'state_32': self._masked_mean(error, mask32),
        }
        loss_mode = str(self.hparams.get(
            'evolution_state_loss_mode', 'regional'))
        if loss_mode == 'regional':
            losses['loss'] = (
                losses['state_active']
                + float(self.hparams.get('evolution_state_16_weight', 0.5))
                * losses['state_16']
                + float(self.hparams.get('evolution_state_32_weight', 0.5))
                * losses['state_32'])
        elif loss_mode == 'pixel_weighted':
            losses['loss'], weights = self._pixel_weighted_state_loss(
                error, active, mask16, mask32,
                increment16=self.hparams.get(
                    'evolution_pixel_16_increment', 1.0),
                increment32=self.hparams.get(
                    'evolution_pixel_32_increment', 1.0),
                max_weight=self.hparams.get(
                    'evolution_pixel_max_weight', 3.0))
            losses['pixel_weight_mean'] = self._masked_mean(weights, active)
        else:
            raise ValueError(
                'evolution_state_loss_mode must be regional or pixel_weighted')

        with torch.no_grad():
            source = result['source_rain']
            oracle = target_rain - result['advected_rain']
            threshold = float(self.hparams.get(
                'evolution_source_sign_threshold', 0.1))
            growth = oracle > threshold
            decay = oracle < -threshold
            eps = source.new_tensor(1e-6)
            for name, mask in (
                    ('active', active), ('16', mask16), ('32', mask32),
                    ('growth', growth), ('decay', decay)):
                predicted_scale = self._masked_mean(source.abs(), mask)
                oracle_scale = self._masked_mean(oracle.abs(), mask)
                losses[f'source_abs_{name}_mm_h'] = predicted_scale
                losses[f'source_scale_ratio_{name}'] = (
                    predicted_scale / (oracle_scale + eps))
            losses['source_growth_sign_accuracy'] = self._masked_mean(
                (source > 0).to(source.dtype), growth)
            losses['source_decay_sign_accuracy'] = self._masked_mean(
                (source < 0).to(source.dtype), decay)
            losses['source_positive_fraction'] = (source > 0).float().mean()
            losses['source_negative_fraction'] = (source < 0).float().mean()
            capacity = result['source_positive_capacity']
            losses['source_positive_capacity_fraction'] = (
                (capacity < self.model.source_max_rain - 1e-4).float().mean())
            losses['source_positive_saturation_fraction'] = (
                (source > 0.99 * capacity.clamp_min(1e-6)).float().mean())
            losses['source_sink_clear_fraction'] = (
                result['evolved_rain'] <= 1e-6).float().mean()
            losses['evolved_above_rmax_fraction'] = (
                result['evolved_rain'] > operator.max_rain + 1e-5).float().mean()
            losses['normalized_dbz_at_upper_bound_fraction'] = (
                result['prediction'] >= 1.0 - 1e-6).float().mean()
        return losses

    def _bounded_source_training_step(self, batch_x, batch_y):
        result = self.model(
            batch_x, return_aux=True, teacher_forcing=batch_y)
        values = self._bounded_state_terms(result, batch_y)
        for name, value in values.items():
            self.log(f'train_{name}', value, on_step=name == 'loss',
                     on_epoch=True, prog_bar=name == 'loss')
        return values['loss']

    def _factorized_source_training_step(self, batch_x, batch_y):
        warmup_epochs = int(self.hparams.get(
            'evolution_source_teacher_forcing_epochs', 0) or 0)
        schedule_epochs = int(self.hparams.get(
            'evolution_source_schedule_epochs', 4) or 4)
        if self.current_epoch < warmup_epochs:
            teacher_forcing_ratio = 1.0
        else:
            progress = self.current_epoch - warmup_epochs
            teacher_forcing_ratio = max(
                0.0, 1.0 - progress / max(schedule_epochs, 1))
        use_teacher_forcing = teacher_forcing_ratio > 0.0
        free_rollout = bool(self.hparams.get(
            'evolution_free_rollout_training', False))
        result = self.model(
            batch_x, return_aux=True,
            teacher_forcing=batch_y if use_teacher_forcing else None,
            teacher_forcing_ratio=teacher_forcing_ratio)
        values = self._factorized_source_terms(result, batch_y)
        if free_rollout:
            target = batch_y[:, :result['prediction'].shape[1]]
            rollout_loss = self.validation_criterion(
                result['prediction'], target)
            values['loss'] = (values['loss'] + float(self.hparams.get(
                'evolution_rollout_loss_weight', 1.0)) * rollout_loss)
            values['rollout_loss'] = rollout_loss
        self.log('train_teacher_forcing_ratio', teacher_forcing_ratio,
                 on_step=False, on_epoch=True)
        for name, value in values.items():
            self.log(f'train_{name}', value, on_step=name == 'loss',
                     on_epoch=True, prog_bar=name == 'loss')
        return values['loss']

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        if (getattr(self.model, 'use_source', False)
                and self.model.source_parameterization == 'factorized_regime'):
            return self._factorized_source_training_step(batch_x, batch_y)
        if (getattr(self.model, 'use_source', False)
                and self.model.source_parameterization == 'bounded_state'):
            return self._bounded_source_training_step(batch_x, batch_y)
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

        source_supervision = result['flow'].new_zeros(())
        source_sparse = result['flow'].new_zeros(())
        source_tv = result['flow'].new_zeros(())
        oracle_source_abs_mean = result['flow'].new_zeros(())
        if getattr(self.model, 'use_source', False):
            operator = self.model.operator
            previous_rain = normalized_dbz_to_rain(
                previous, value_scale=operator.value_scale,
                zr_a=operator.zr_a, zr_b=operator.zr_b)
            target_rain = normalized_dbz_to_rain(
                batch_y, value_scale=operator.value_scale,
                zr_a=operator.zr_a, zr_b=operator.zr_b)
            with torch.no_grad():
                advected_rain = torch.stack([
                    backward_warp(
                        previous_rain[:, step], result['flow'][:, step],
                        align_corners=operator.align_corners,
                        padding_mode=operator.padding_mode)
                    for step in range(batch_y.shape[1])
                ], dim=1)
                oracle_source = target_rain - advected_rain
                event_rain = torch.maximum(previous_rain, target_rain)
                active_threshold = float(self.hparams.get(
                    'evolution_source_active_threshold', 0.1))
                active = ((event_rain >= active_threshold)
                          | (oracle_source.abs() >= active_threshold))
                weights = active.float()
                weights = weights + active * 2.0 * (event_rain >= 16.0).float()
                weights = weights + active * 4.0 * (event_rain >= 32.0).float()
                weights = weights + active * (oracle_source.abs() >= 0.5).float()
            source_scale = self.model.source_max_rain
            source_error = torch.nn.functional.smooth_l1_loss(
                result['source_rain'] / source_scale,
                oracle_source / source_scale,
                reduction='none', beta=float(self.hparams.get(
                    'evolution_source_huber_beta', 0.03)))
            source_supervision = ((source_error * weights).sum()
                                  / weights.sum().clamp_min(1.0))
            normalized_source = result['source_rain'] / source_scale
            source_sparse = normalized_source.abs().mean()
            source_tv = self._spatial_smoothness(normalized_source)
            if normalized_source.shape[1] > 1:
                source_tv = source_tv + (
                    normalized_source[:, 1:] - normalized_source[:, :-1]
                ).abs().mean()
            oracle_source_abs_mean = oracle_source.abs().mean()

        if bool(self.hparams.get('evolution_gate_supervision_only', False)):
            loss = gate_supervision
        elif bool(self.hparams.get('evolution_source_supervision_only', False)):
            loss = (source_supervision
                    + float(self.hparams.get(
                        'evolution_source_sparse_weight', 0.01)) * source_sparse
                    + float(self.hparams.get(
                        'evolution_source_tv_weight', 0.001)) * source_tv)
        else:
            loss = (forecast_loss
                    + float(self.hparams.get('evolution_tf_weight', 0.5)) * tf_loss
                    + float(self.hparams.get('evolution_spatial_weight', 1e-3)) * spatial_loss
                    + float(self.hparams.get('evolution_temporal_weight', 1e-3)) * temporal_loss
                    + float(self.hparams.get(
                        'evolution_gate_supervision_weight') or 0.0) * gate_supervision
                    + float(self.hparams.get(
                        'evolution_source_supervision_weight') or 0.0) * source_supervision
                    + float(self.hparams.get(
                        'evolution_source_sparse_weight') or 0.0) * source_sparse
                    + float(self.hparams.get(
                        'evolution_source_tv_weight') or 0.0) * source_tv)
        values = {'loss': loss, 'forecast': forecast_loss, 'transport': tf_loss,
                  'flow_spatial': spatial_loss, 'flow_temporal': temporal_loss}
        if getattr(self.model, 'use_flow_gate', False):
            values['gate_mean'] = result['flow_gate'].mean()
            values['gate_supervision'] = gate_supervision
            values['gate_target_mean'] = gate_target_mean
        if getattr(self.model, 'use_source', False):
            values['source_supervision'] = source_supervision
            values['source_sparse'] = source_sparse
            values['source_tv'] = source_tv
            values['source_abs_mean_mm_h'] = result['source_rain'].abs().mean()
            values['oracle_source_abs_mean_mm_h'] = oracle_source_abs_mean
        for name, value in values.items():
            self.log(f'train_{name}', value, on_step=name == 'loss', on_epoch=True,
                     prog_bar=name == 'loss')
        for name, value in getattr(self.criterion, 'last_components', {}).items():
            self.log(f'train_{name}', value, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if (getattr(self.model, 'use_source', False)
                and self.model.source_parameterization == 'factorized_regime'):
            batch_x, batch_y = batch
            steps = self.model.forecast_steps
            target = batch_y[:, :steps]
            free_rollout = bool(self.hparams.get(
                'evolution_validation_free_rollout', True))
            result = self.model(
                batch_x, return_aux=True,
                teacher_forcing=None if free_rollout else batch_y)
            loss = self.validation_criterion(result['prediction'], target)
            self.log('val_loss', loss, on_step=True, on_epoch=True,
                     prog_bar=False)
            if self.hparams.dataname == 'bth_radar':
                self._update_val_precipitation(result['prediction'], target)
            values = self._factorized_source_terms(result, target)
            for name, value in values.items():
                self.log(f'val_tf_{name}', value, on_step=False,
                         on_epoch=True, prog_bar=False)
            return loss
        if (getattr(self.model, 'use_source', False)
                and self.model.source_parameterization == 'bounded_state'):
            batch_x, batch_y = batch
            steps = self.model.forecast_steps
            target = batch_y[:, :steps]
            if steps == 1:
                result = self.model(batch_x, return_aux=True)
                loss = self.validation_criterion(
                    result['prediction'], target)
            else:
                prediction = self.model(batch_x)
                loss = self.validation_criterion(prediction, target)
                result = self.model(
                    batch_x, return_aux=True, teacher_forcing=batch_y)
            self.log('val_loss', loss, on_step=True, on_epoch=True,
                     prog_bar=False)
            if self.hparams.dataname == 'bth_radar':
                self._update_val_precipitation(
                    result['prediction'] if steps == 1 else prediction,
                    target)
            values = self._bounded_state_terms(result, target)
            for name, value in values.items():
                self.log(f'val_tf_{name}', value, on_step=False,
                         on_epoch=True, prog_bar=False)
            return loss
        return super().validation_step(batch, batch_idx)


class EvolutionConvLSTM(EvolutionPhysicsBase):
    """ConvLSTM history encoder with explicit differentiable evolution."""

    def _build_model(self, **args):
        num_hidden = [
            int(value) for value in self.hparams.num_hidden.split(',')]
        return EvolutionConvLSTM_Model(
            len(num_hidden), num_hidden, self.hparams)
