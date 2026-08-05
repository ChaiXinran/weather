"""Overfit one real train batch for the R4-c2 Gate 0 engineering check."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openstl.datasets.dataloader_radar import BTHRadarDataset
from openstl.models import EvolutionConvLSTM_Model
from openstl.modules import normalized_dbz_to_rain
from openstl.utils import load_config


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def state_terms(result, target, operator, config):
    target = target[:, :result['prediction'].shape[1]]
    target_rain = normalized_dbz_to_rain(
        target, value_scale=operator.value_scale,
        zr_a=operator.zr_a, zr_b=operator.zr_b)
    error = F.smooth_l1_loss(
        result['evolved_rain'], target_rain, reduction='none',
        beta=config.evolution_state_huber_beta)
    event = torch.maximum(result['advected_rain'].detach(), target_rain)
    masks = {
        'active': event >= config.evolution_source_active_threshold,
        '16': event >= 16.0,
        '32': event >= 32.0,
    }
    regional = {name: masked_mean(error, mask)
                for name, mask in masks.items()}
    weights = masks['active'].to(error.dtype)
    weights = weights + config.evolution_pixel_16_increment * masks['16']
    weights = weights + config.evolution_pixel_32_increment * masks['32']
    weights = weights.clamp_max(config.evolution_pixel_max_weight)
    loss = (error * weights).sum() / weights.sum().clamp_min(1.0)
    oracle = target_rain - result['advected_rain'].detach()
    growth = oracle > config.evolution_source_sign_threshold
    decay = oracle < -config.evolution_source_sign_threshold
    source = result['source_rain']
    diagnostics = {
        'loss': float(loss.detach()),
        'state_all': float(error.mean()),
        **{f'state_{key}': float(value.detach())
           for key, value in regional.items()},
        'pixel_weight_mean_active': float(masked_mean(
            weights, masks['active'])),
        'growth_pixels': int(growth.sum()),
        'decay_pixels': int(decay.sum()),
        'pixels_16': int(masks['16'].sum()),
        'pixels_32': int(masks['32'].sum()),
        'growth_sign_accuracy': float(masked_mean(
            (source > 0).float(), growth)),
        'decay_sign_accuracy': float(masked_mean(
            (source < 0).float(), decay)),
        'source_abs_mean': float(source.abs().mean()),
        'source_positive_fraction': float((source > 0).float().mean()),
        'source_negative_fraction': float((source < 0).float().mean()),
        'source_positive_saturation_fraction': float((
            source > 0.99 * result['source_positive_capacity'].clamp_min(1e-6)
        ).float().mean()),
        'sink_clear_fraction': float(
            (result['evolved_rain'] <= 1e-6).float().mean()),
        'evolved_above_rmax_fraction': float((
            result['evolved_rain'] > operator.max_rain + 1e-5
        ).float().mean()),
    }
    return loss, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    values = dict(load_config(args.config))
    values.update(
        in_shape=[10, 1, 66, 70], pre_seq_length=10,
        aft_seq_length=20, total_length=30)
    config = SimpleNamespace(**values)
    dataset = BTHRadarDataset(
        data_root=args.data_root, pre_seq_length=10, aft_seq_length=20,
        start_date='2025-05-01', end_date='2025-08-31',
        manifest_path=config.manifest_path, split='train',
        radar_cache_path=config.radar_cache_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    model = EvolutionConvLSTM_Model(4, [64, 64, 64, 64], config).cuda()
    model.load_pretrained_motion(config.evolution_motion_checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_parameters = model.source_parameters()
    for parameter in source_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(source_parameters, lr=args.lr)

    selected = None
    for batch_index, (history, target) in enumerate(loader):
        history, target = history.cuda(), target.cuda()
        with torch.no_grad():
            result = model(history, return_aux=True, teacher_forcing=target)
            _, diagnostics = state_terms(result, target, model.operator, config)
        if (diagnostics['growth_pixels'] > 0
                and diagnostics['decay_pixels'] > 0
                and diagnostics['pixels_16'] > 0
                and diagnostics['pixels_32'] > 0):
            selected = (batch_index, history, target, diagnostics)
            break
    if selected is None:
        raise RuntimeError('No train batch contains growth, decay, 16 and 32 mm/h pixels')

    batch_index, history, target, initial = selected
    history, target = history.detach(), target.detach()
    trace = [initial['loss']]
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        result = model(history, return_aux=True, teacher_forcing=target)
        loss, _ = state_terms(result, target, model.operator, config)
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
    with torch.no_grad():
        result = model(history, return_aux=True, teacher_forcing=target)
        _, final = state_terms(result, target, model.operator, config)
    reduction = 1.0 - final['loss'] / initial['loss']
    report = {
        'status': 'passed' if reduction >= 0.5 else 'failed',
        'config': args.config,
        'checkpoint': config.evolution_motion_checkpoint,
        'train_batch_index': batch_index,
        'batch_size': args.batch_size,
        'steps': args.steps,
        'learning_rate': args.lr,
        'trainable_parameter_count': sum(p.numel() for p in source_parameters),
        'initial': initial,
        'final': final,
        'loss_reduction_fraction': reduction,
        'loss_trace_every_10_steps': trace[::10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
