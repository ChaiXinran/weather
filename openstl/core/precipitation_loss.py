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
        self.eps = float(eps)
        self.last_components = {}

    def _lead_weights(self, target):
        weights = torch.ones(
            target.shape[1], device=target.device, dtype=target.dtype)
        if target.shape[1] > 10:
            weights[10:] = self.second_hour_weight
        return weights.view(1, -1, 1, 1, 1)

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

        soft_terms = []
        for threshold in self.thresholds:
            pred_event = torch.sigmoid(
                (pred - threshold) / self.temperature)
            true_event = (target_for_weights >= threshold).to(pred.dtype)
            intersection = (pred_event * true_event).sum()
            union = (
                pred_event.sum() + true_event.sum() - intersection)
            soft_terms.append(1.0 - (
                intersection + self.eps) / (union + self.eps))
        soft_terms = torch.stack(soft_terms)
        soft_csi = (soft_terms * self.soft_csi_weights).sum()
        total = huber + soft_csi
        self.last_components = {
            'huber': huber.detach(),
            'soft_csi_16': soft_terms[0].detach(),
            'soft_csi_32': soft_terms[1].detach(),
            'soft_csi_weighted': soft_csi.detach(),
        }
        return total
