"""R4-d0 oracle-source partition diagnostics for frozen R4-b motion."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openstl.datasets.dataloader_radar import BTHRadarDataset
from openstl.models import EvolutionConvLSTM_Model
from openstl.modules import normalized_dbz_to_rain
from openstl.utils import load_config


REGIONS = ('interior', 'edge', 'birth', 'death', 'clear')
REGIMES = ('growth', 'steady', 'decay')
INTENSITY_BINS = (
    ('0.1-8', 0.1, 8.0),
    ('8-16', 8.0, 16.0),
    ('16-32', 16.0, 32.0),
    ('ge32', 32.0, float('inf')),
)


class RunningStats:
    def __init__(self, max_values=200000, seed=42):
        self.count = 0
        self.sum = 0.0
        self.abs_sum = 0.0
        self.positive = 0
        self.negative = 0
        self.values = []
        self.max_values = int(max_values)
        self.rng = random.Random(seed)

    def update(self, values):
        if values.numel() == 0:
            return
        flat = values.detach().float().cpu().reshape(-1)
        count = int(flat.numel())
        self.count += count
        self.sum += float(flat.sum())
        self.abs_sum += float(flat.abs().sum())
        self.positive += int((flat > 0).sum())
        self.negative += int((flat < 0).sum())
        if self.max_values <= 0:
            return
        for value in flat.tolist():
            if len(self.values) < self.max_values:
                self.values.append(value)
            else:
                index = self.rng.randrange(self.count)
                if index < self.max_values:
                    self.values[index] = value

    def row(self, name):
        if self.values:
            sample = torch.tensor(self.values)
            quantiles = torch.quantile(
                sample, torch.tensor([0.5, 0.9, 0.95, 0.99])).tolist()
        else:
            quantiles = [0.0, 0.0, 0.0, 0.0]
        denom = max(self.count, 1)
        return {
            'name': name,
            'count': self.count,
            'mean': self.sum / denom,
            'abs_mean': self.abs_sum / denom,
            'p50': quantiles[0],
            'p90': quantiles[1],
            'p95': quantiles[2],
            'p99': quantiles[3],
            'positive_fraction': self.positive / denom,
            'negative_fraction': self.negative / denom,
        }


def load_model(config, checkpoint):
    model = EvolutionConvLSTM_Model(4, [64, 64, 64, 64], config)
    payload = torch.load(checkpoint, map_location='cpu')
    state = {
        key[len('model.'):]: value
        for key, value in payload['state_dict'].items()
        if key.startswith('model.')
    }
    model.load_state_dict(state, strict=True)
    return model.cuda().eval() if torch.cuda.is_available() else model.eval()


def erode(mask, radius=1):
    flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
    out = -F.max_pool2d(-flat, 2 * radius + 1, stride=1, padding=radius)
    return (out > 0.5).reshape(mask.shape)


def dilate(mask, radius=1):
    flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
    out = F.max_pool2d(flat, 2 * radius + 1, stride=1, padding=radius)
    return (out > 0.5).reshape(mask.shape)


def build_regions(advected, target, threshold):
    ma = advected >= threshold
    my = target >= threshold
    interior = erode(ma & my)
    birth = (~ma) & my
    death = ma & (~my)
    edge = dilate(ma | my) & ~(interior | birth | death)
    clear = ~(interior | edge | birth | death)
    return {
        'interior': interior,
        'edge': edge,
        'birth': birth,
        'death': death,
        'clear': clear,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_histograms(path, region_stats):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    plt.figure(figsize=(10, 6))
    for name in REGIONS:
        values = region_stats[name].values
        if values:
            plt.hist(values, bins=80, range=(-10, 10), density=True,
                     histtype='step', linewidth=1.4, label=name)
    plt.xlabel('Oracle source S* (mm/h)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=(
        'configs/bth_radar/ConvLSTM_evolution_motion_rainrate_scale05.py'))
    parser.add_argument('--checkpoint', default=(
        'work_dirs/bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0/'
        'checkpoints/val-csi-epoch=01-val_csi_score=0.640662.ckpt'))
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--output', type=Path,
                        default=Path('.research/r4d0_source_partition'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--forecast-steps', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-values-per-group', type=int, default=200000)
    parser.add_argument('--delta', type=float, default=0.5)
    args = parser.parse_args()

    values = dict(load_config(args.config))
    values.update(in_shape=[10, 1, 66, 70], pre_seq_length=10,
                  aft_seq_length=20, total_length=30)
    config = SimpleNamespace(**values)
    dataset = BTHRadarDataset(
        data_root=args.data_root, pre_seq_length=10, aft_seq_length=20,
        start_date='2025-05-01', end_date='2025-08-31',
        manifest_path=config.manifest_path, split=args.split,
        radar_cache_path=config.radar_cache_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    model = load_model(config, args.checkpoint)
    device = next(model.parameters()).device
    steps = min(int(args.forecast_steps), int(config.aft_seq_length))
    threshold = float(getattr(config, 'wet_threshold', 0.1))

    region_stats = {
        name: RunningStats(args.max_values_per_group, seed=42 + index)
        for index, name in enumerate(REGIONS)
    }
    intensity_stats = {
        name: RunningStats(args.max_values_per_group, seed=100 + index)
        for index, (name, _, _) in enumerate(INTENSITY_BINS)
    }
    lead_stats = {
        lead: RunningStats(args.max_values_per_group, seed=200 + lead)
        for lead in range(steps)
    }
    regime_counts = {
        'growth': 0,
        'steady': 0,
        'decay': 0,
    }
    positive_components = []
    birth_near_existing = 0
    birth_total = 0
    processed = 0

    with torch.inference_mode():
        for history, target in loader:
            history = history.to(device)
            target = target.to(device)
            result = model(history, return_aux=True)
            flow = result['flow'][:, :steps]
            previous = torch.cat((history[:, -1:], target[:, :-1]), dim=1)
            advected = torch.stack([
                model.operator.warp(previous[:, step], flow[:, step])
                for step in range(steps)
            ], dim=1)
            advected_rain = normalized_dbz_to_rain(
                advected, value_scale=config.radar_value_scale,
                zr_a=config.zr_a, zr_b=config.zr_b)
            target_rain = normalized_dbz_to_rain(
                target[:, :steps], value_scale=config.radar_value_scale,
                zr_a=config.zr_a, zr_b=config.zr_b)
            oracle = target_rain - advected_rain
            regions = build_regions(advected_rain, target_rain, threshold)

            for name, mask in regions.items():
                region_stats[name].update(oracle[mask])
            for bin_name, low, high in INTENSITY_BINS:
                mask = (advected_rain >= low) & (advected_rain < high)
                intensity_stats[bin_name].update(oracle[mask])
            for lead in range(steps):
                lead_stats[lead].update(oracle[:, lead])

            interior_source = oracle[regions['interior']]
            regime_counts['growth'] += int((interior_source > args.delta).sum())
            regime_counts['steady'] += int((
                interior_source.abs() <= args.delta).sum())
            regime_counts['decay'] += int((interior_source < -args.delta).sum())

            positive_mask = regions['interior'] & (oracle > args.delta)
            flat_positive = positive_mask.reshape(-1, 1, *positive_mask.shape[-2:])
            if flat_positive.any():
                connected = F.max_pool2d(
                    flat_positive.float(), 3, stride=1, padding=1)
                positive_components.append(float(connected.mean()))

            birth = regions['birth']
            near_existing = dilate(advected_rain >= threshold, radius=2)
            birth_near_existing += int((birth & near_existing).sum())
            birth_total += int(birth.sum())

            processed += history.shape[0]
            if args.max_samples and processed >= args.max_samples:
                break

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'region_examples').mkdir(exist_ok=True)
    row_fields = [
        'name', 'count', 'mean', 'abs_mean', 'p50', 'p90', 'p95', 'p99',
        'positive_fraction', 'negative_fraction']
    region_rows = [region_stats[name].row(name) for name in REGIONS]
    intensity_rows = [
        intensity_stats[name].row(name) for name, _, _ in INTENSITY_BINS]
    lead_rows = [
        {**lead_stats[lead].row(str(lead + 1)), 'lead': lead + 1}
        for lead in range(steps)]
    write_csv(args.output / 'source_by_region.csv', region_rows, row_fields)
    write_csv(args.output / 'source_by_intensity.csv', intensity_rows, row_fields)
    write_csv(args.output / 'source_by_lead.csv', lead_rows,
              ['lead'] + row_fields)
    total_region = sum(row['count'] for row in region_rows)
    write_csv(args.output / 'region_counts.csv', [
        {'region': row['name'], 'count': row['count'],
         'fraction': row['count'] / max(total_region, 1)}
        for row in region_rows], ['region', 'count', 'fraction'])
    total_regime = sum(regime_counts.values())
    write_csv(args.output / 'regime_counts.csv', [
        {'delta': args.delta, 'regime': name, 'count': count,
         'fraction': count / max(total_regime, 1)}
        for name, count in regime_counts.items()],
        ['delta', 'regime', 'count', 'fraction'])
    histogram_written = plot_histograms(
        args.output / 'source_histograms.png', region_stats)

    summary = {
        'status': 'complete',
        'split': args.split,
        'processed_samples': min(processed, len(dataset)),
        'dataset_samples': len(dataset),
        'forecast_steps': steps,
        'checkpoint': args.checkpoint,
        'delta': args.delta,
        'region_counts': {
            row['name']: row['count'] for row in region_rows},
        'regime_counts': regime_counts,
        'edge_abs_over_interior_abs': (
            region_stats['edge'].row('edge')['abs_mean']
            / max(region_stats['interior'].row('interior')['abs_mean'], 1e-6)),
        'birth_near_existing_fraction': (
            birth_near_existing / max(birth_total, 1)),
        'positive_source_continuity_proxy': (
            sum(positive_components) / max(len(positive_components), 1)),
        'histogram_written': histogram_written,
        'quantiles_note': (
            'Means/counts are exact over processed pixels; quantiles and '
            'histograms use bounded reservoir samples per group.'),
    }
    (args.output / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    (args.output / 'summary.md').write_text(
        '# R4-d0 Oracle Source Partition\n\n'
        f'- split: {summary["split"]}\n'
        f'- processed samples: {summary["processed_samples"]} / '
        f'{summary["dataset_samples"]}\n'
        f'- forecast steps: {steps}\n'
        f'- edge/interior abs residual ratio: '
        f'{summary["edge_abs_over_interior_abs"]:.4f}\n'
        f'- birth near existing fraction: '
        f'{summary["birth_near_existing_fraction"]:.4f}\n'
        f'- regime counts at delta={args.delta}: {regime_counts}\n',
        encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
