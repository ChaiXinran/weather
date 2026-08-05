"""Evaluate an R4-c2 checkpoint on the validation split only."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openstl.core.precipitation_metrics import PrecipitationEvaluator
from openstl.datasets.dataloader_radar import BTHRadarDataset
from openstl.models import EvolutionConvLSTM_Model
from openstl.modules import normalized_dbz_to_rain
from openstl.utils import load_config


def load_model(config, checkpoint):
    model = EvolutionConvLSTM_Model(4, [64, 64, 64, 64], config)
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    state = {key[len('model.'):]: value for key, value in state.items()
             if key.startswith('model.')}
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def update_region_stats(stats, name, source, mask):
    mask = mask.expand_as(source)
    count = int(mask.sum())
    if not count:
        return
    values = source[mask]
    item = stats.setdefault(name, {
        'count': 0, 'abs_sum': 0.0, 'signed_sum': 0.0,
        'positive_count': 0, 'negative_count': 0})
    item['count'] += count
    item['abs_sum'] += float(values.abs().sum())
    item['signed_sum'] += float(values.sum())
    item['positive_count'] += int((values > 0).sum())
    item['negative_count'] += int((values < 0).sum())


def finalize_region_stats(stats):
    for item in stats.values():
        count = max(item['count'], 1)
        item['abs_mean_mm_h'] = item['abs_sum'] / count
        item['signed_mean_mm_h'] = item['signed_sum'] / count
        item['positive_fraction'] = item['positive_count'] / count
        item['negative_fraction'] = item['negative_count'] / count
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    values = dict(load_config(args.config))
    values.update(in_shape=[10, 1, 66, 70], pre_seq_length=10,
                  aft_seq_length=20, total_length=30)
    config = SimpleNamespace(**values)
    dataset = BTHRadarDataset(
        data_root=args.data_root, pre_seq_length=10, aft_seq_length=20,
        start_date='2025-05-01', end_date='2025-08-31',
        manifest_path=config.manifest_path, split='val',
        radar_cache_path=config.radar_cache_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    model = load_model(config, args.checkpoint)
    evaluator = PrecipitationEvaluator(
        lead_count=20, thresholds=config.precip_thresholds,
        value_scale=config.radar_value_scale,
        value_unit=config.precip_value_unit,
        lead_minutes=config.lead_minutes,
        clip_range=config.precip_clip_range,
        case_threshold=config.case_threshold, case_count=config.case_count,
        event_id_source='manifest', convert_dbz_to_rain=True,
        zr_a=config.zr_a, zr_b=config.zr_b,
        wet_threshold=config.wet_threshold,
        grid_spacing_km=config.grid_spacing_km,
        neighborhood_windows=config.neighborhood_windows,
        object_iou_threshold=config.object_iou_threshold,
        bootstrap_repetitions=config.bootstrap_repetitions,
        bootstrap_seed=config.bootstrap_seed)
    region_stats = {}
    source_counts = {'pixels': 0, 'above_rmax': 0, 'upper_dbz': 0,
                     'positive_saturated': 0, 'sink_cleared': 0}
    offset = 0
    with torch.inference_mode():
        for history, target in loader:
            history, target = history.cuda(), target.cuda()
            rollout = (None if args.source_only
                       else model(history, return_aux=True))
            teacher = model(history, return_aux=True, teacher_forcing=target)
            batch = history.shape[0]
            indices = range(offset, offset + batch)
            if not args.source_only:
                evaluator.update(
                    rollout['prediction'].cpu().numpy(), target.cpu().numpy(),
                    history.cpu().numpy(),
                    event_ids=[dataset.event_id_for_sample(i) for i in indices],
                    sample_ids=list(indices))
            offset += batch

            target_rain = normalized_dbz_to_rain(
                target, value_scale=config.radar_value_scale,
                zr_a=config.zr_a, zr_b=config.zr_b)
            source = teacher['source_rain']
            advected = teacher['advected_rain']
            oracle = target_rain - advected
            wet_target = target_rain >= config.wet_threshold
            wet_advected = advected >= config.wet_threshold
            kernel = (1, 3, 3)
            eroded = -F.max_pool3d(
                -wet_target.float(), kernel, stride=1,
                padding=(0, 1, 1)) > 0.5
            dilated = F.max_pool3d(
                wet_target.float(), kernel, stride=1, padding=(0, 1, 1)) > 0.5
            regions = {
                'persistent_interior': wet_advected & eroded,
                'object_edge_band': dilated & ~eroded,
                'newborn': ~wet_advected & wet_target,
                'dissipated': wet_advected & ~wet_target,
                'clear_background': ~wet_advected & ~wet_target,
                'growth': oracle > config.evolution_source_sign_threshold,
                'decay': oracle < -config.evolution_source_sign_threshold,
                'rain_16': torch.maximum(advected, target_rain) >= 16.0,
                'rain_32': torch.maximum(advected, target_rain) >= 32.0,
            }
            for name, mask in regions.items():
                update_region_stats(region_stats, name, source, mask)
            capacity = teacher['source_positive_capacity']
            source_counts['pixels'] += source.numel()
            source_counts['above_rmax'] += int((
                teacher['evolved_rain'] > model.operator.max_rain + 1e-5).sum())
            source_counts['upper_dbz'] += int((
                teacher['prediction'] >= 1.0 - 1e-6).sum())
            source_counts['positive_saturated'] += int((
                source > 0.99 * capacity.clamp_min(1e-6)).sum())
            source_counts['sink_cleared'] += int((
                teacher['evolved_rain'] <= 1e-6).sum())

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.source_only:
        evaluator.save(args.output / 'precipitation_evaluation')
    diagnostics = {
        'split': 'val', 'sample_count': len(dataset),
        'checkpoint': args.checkpoint,
        'regions': finalize_region_stats(region_stats),
        'constraint_counts': source_counts,
        'constraint_fractions': {
            key: value / max(source_counts['pixels'], 1)
            for key, value in source_counts.items() if key != 'pixels'},
    }
    (args.output / 'source_diagnostics.json').write_text(
        json.dumps(diagnostics, indent=2), encoding='utf-8')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
