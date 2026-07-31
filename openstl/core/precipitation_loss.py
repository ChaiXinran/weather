"""Differentiable, event-aware loss for precipitation nowcasting."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrecipitationR2Loss(nn.Module):
    """Normalized weighted Huber plus low-weight soft CSI constraints.

    Inputs and targets are normalized Radar dBZ fields. Rain-rate thresholds
    are converted to normalized dBZ once, avoiding the steep derivative of a
    differentiable Z-R conversion inside the loss.
    """

    def __init__(self,
                 value_scale=50.0,
                 zr_a=200.0,
                 zr_b=1.6,
                 thresholds=(16.0, 32.0),
                 intensity_weights=(2.0, 3.0),
                 soft_csi_weights=(0.005, 0.001),
                 soft_csi_temperature=0.03,
                 huber_beta=0.05,
                 second_hour_weight=1.2,
                 soft_csi_mode='micro',
                 segmented_soft_csi_weights=None,
                 empty_event_penalty=0.1,
                 eps=1e-6):
        super().__init__()
        if len(thresholds) != len(intensity_weights):
            raise ValueError('thresholds and intensity_weights must match')
        if len(thresholds) != len(soft_csi_weights):
            raise ValueError('thresholds and soft_csi_weights must match')
        if value_scale <= 0 or zr_a <= 0 or zr_b <= 0:
            raise ValueError('Z-R parameters and value_scale must be positive')
        if soft_csi_temperature <= 0 or huber_beta <= 0:
            raise ValueError('temperature and huber_beta must be positive')
        if soft_csi_mode not in ('micro', 'sample_period'):
            raise ValueError(
                'soft_csi_mode must be "micro" or "sample_period"')

        normalized_thresholds = [
            (10.0 * math.log10(zr_a)
             + 10.0 * zr_b * math.log10(float(rain_rate))) / value_scale
            for rain_rate in thresholds
        ]
        self.register_buffer(
            'thresholds', torch.tensor(normalized_thresholds))
        self.register_buffer(
            'intensity_weights', torch.tensor(intensity_weights))
        self.register_buffer(
            'soft_csi_weights', torch.tensor(soft_csi_weights))
        self.temperature = float(soft_csi_temperature)
        self.huber_beta = float(huber_beta)
        self.second_hour_weight = float(second_hour_weight)
        self.soft_csi_mode = soft_csi_mode
        self.empty_event_penalty = float(empty_event_penalty)
        if segmented_soft_csi_weights is None:
            segmented_soft_csi_weights = (
                (soft_csi_weights[0], soft_csi_weights[1]),
                (soft_csi_weights[0], soft_csi_weights[1]))
        if (len(segmented_soft_csi_weights) != 2
                or any(len(period) != len(thresholds)
                       for period in segmented_soft_csi_weights)):
            raise ValueError(
                'segmented_soft_csi_weights must be [2, threshold_count]')
        self.register_buffer(
            'segmented_soft_csi_weights',
            torch.tensor(segmented_soft_csi_weights))
        self.eps = float(eps)
        self.last_components = {}

    def _lead_weights(self, target):
        weights = torch.ones(
            target.shape[1], device=target.device, dtype=target.dtype)
        if target.shape[1] > 10:
            weights[10:] = self.second_hour_weight
        return weights.view(1, -1, 1, 1, 1)

    def _micro_soft_csi(self, pred, target):
        terms = []
        for threshold in self.thresholds:
            pred_event = torch.sigmoid(
                (pred - threshold) / self.temperature)
            true_event = (target >= threshold).to(pred.dtype)
            intersection = (pred_event * true_event).sum()
            union = pred_event.sum() + true_event.sum() - intersection
            terms.append(
                1.0 - (intersection + self.eps) / (union + self.eps))
        terms = torch.stack(terms)
        return (terms * self.soft_csi_weights).sum(), terms

    def _sample_period_soft_csi(self, pred, target):
        period_bounds = ((0, min(10, pred.shape[1])),
                         (min(10, pred.shape[1]), pred.shape[1]))
        all_terms = []
        weighted_terms = []
        reduce_dims = (1, 2, 3, 4)
        for period_index, (start, end) in enumerate(period_bounds):
            period_pred = pred[:, start:end]
            period_true = target[:, start:end]
            for threshold_index, threshold in enumerate(self.thresholds):
                pred_event = torch.sigmoid(
                    (period_pred - threshold) / self.temperature)
                true_event = (period_true >= threshold).to(pred.dtype)
                true_sum = true_event.sum(dim=reduce_dims)
                intersection = (
                    pred_event * true_event).sum(dim=reduce_dims)
                union = (
                    pred_event.sum(dim=reduce_dims) + true_sum - intersection)
                sample_loss = 1.0 - (
                    intersection + self.eps) / (union + self.eps)
                active = true_sum > 0
                active_loss = (
                    sample_loss[active].mean()
                    if active.any() else sample_loss.new_zeros(()))
                empty = ~active
                empty_fp = (
                    pred_event[empty].mean()
                    if empty.any() else sample_loss.new_zeros(()))
                term = active_loss + self.empty_event_penalty * empty_fp
                all_terms.append(term)
                weighted_terms.append(
                    term * self.segmented_soft_csi_weights[
                        period_index, threshold_index])
        return torch.stack(weighted_terms).sum(), torch.stack(all_terms)

    def forward(self, pred, target):
        target_for_weights = target.detach()
        pixel_weights = torch.ones_like(target_for_weights)
        for threshold, increment in zip(
                self.thresholds, self.intensity_weights):
            pixel_weights = pixel_weights + increment * (
                target_for_weights >= threshold).to(target.dtype)
        weights = pixel_weights * self._lead_weights(target)
        weights = weights / weights.mean().clamp_min(self.eps)

        huber = F.smooth_l1_loss(
            pred, target, reduction='none', beta=self.huber_beta)
        huber = (huber * weights).mean()

        if self.soft_csi_mode == 'sample_period':
            soft_csi, soft_terms = self._sample_period_soft_csi(
                pred, target_for_weights)
        else:
            soft_csi, soft_terms = self._micro_soft_csi(
                pred, target_for_weights)
        total = huber + soft_csi
        self.last_components = {
            'huber': huber.detach(),
            'soft_csi_weighted': soft_csi.detach(),
        }
        if self.soft_csi_mode == 'sample_period':
            labels = ('16_0_1h', '32_0_1h', '16_1_2h', '32_1_2h')
            self.last_components.update({
                f'soft_csi_{label}': value.detach()
                for label, value in zip(labels, soft_terms)
            })
        else:
            self.last_components.update({
                'soft_csi_16': soft_terms[0].detach(),
                'soft_csi_32': soft_terms[1].detach(),
            })
        return total
