"""R4-c0 no-training audit of the teacher-forced rain-rate source residual."""

import csv
import json
import os.path as osp
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openstl.api import BaseExperiment
from openstl.modules import backward_warp, normalized_dbz_to_rain
from openstl.utils import (check_dir, create_parser, default_parser,
                           load_config, update_config)


MAGNITUDE_EDGES = np.linspace(0.0, 50.0, 5001, dtype=np.float64)
SPARSITY_LEVELS = (0.1, 0.5, 1.0, 2.0, 5.0)


class StreamingSourceStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.abs_sum = 0.0
        self.square_sum = 0.0
        self.minimum = float('inf')
        self.maximum = float('-inf')
        self.positive_count = 0
        self.negative_count = 0
        self.zero_count = 0
        self.abs_histogram = np.zeros(len(MAGNITUDE_EDGES) - 1, dtype=np.int64)
        self.positive_histogram = np.zeros_like(self.abs_histogram)
        self.negative_histogram = np.zeros_like(self.abs_histogram)
        self.above = {value: 0 for value in SPARSITY_LEVELS}

    def update(self, values, mask=None):
        values = np.asarray(values, dtype=np.float32)
        if mask is not None:
            values = values[np.asarray(mask, dtype=bool)]
        else:
            values = values.reshape(-1)
        values = values[np.isfinite(values)]
        if not values.size:
            return
        absolute = np.abs(values)
        positive = values[values > 0]
        negative = -values[values < 0]
        self.count += int(values.size)
        self.sum += float(values.sum(dtype=np.float64))
        self.abs_sum += float(absolute.sum(dtype=np.float64))
        self.square_sum += float(np.square(values, dtype=np.float64).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.positive_count += int(positive.size)
        self.negative_count += int(negative.size)
        self.zero_count += int(values.size - positive.size - negative.size)
        self.abs_histogram += np.histogram(
            np.minimum(absolute, MAGNITUDE_EDGES[-1]), MAGNITUDE_EDGES)[0]
        if positive.size:
            self.positive_histogram += np.histogram(
                np.minimum(positive, MAGNITUDE_EDGES[-1]), MAGNITUDE_EDGES)[0]
        if negative.size:
            self.negative_histogram += np.histogram(
                np.minimum(negative, MAGNITUDE_EDGES[-1]), MAGNITUDE_EDGES)[0]
        for level in SPARSITY_LEVELS:
            self.above[level] += int(np.count_nonzero(absolute >= level))

    @staticmethod
    def _quantile(histogram, probability):
        total = int(histogram.sum())
        if not total:
            return None
        index = int(np.searchsorted(
            np.cumsum(histogram), probability * total, side='left'))
        index = min(index, len(MAGNITUDE_EDGES) - 2)
        return float(MAGNITUDE_EDGES[index + 1])

    def report(self):
        if not self.count:
            return {'count': 0}
        return {
            'count': self.count,
            'mean_mm_h': self.sum / self.count,
            'mean_absolute_mm_h': self.abs_sum / self.count,
            'rmse_mm_h': float(np.sqrt(self.square_sum / self.count)),
            'minimum_mm_h': self.minimum,
            'maximum_mm_h': self.maximum,
            'positive_fraction': self.positive_count / self.count,
            'negative_fraction': self.negative_count / self.count,
            'exact_zero_fraction': self.zero_count / self.count,
            'absolute_percentiles_mm_h': {
                'p90': self._quantile(self.abs_histogram, 0.90),
                'p95': self._quantile(self.abs_histogram, 0.95),
                'p99': self._quantile(self.abs_histogram, 0.99),
            },
            'positive_percentiles_mm_h': {
                'p90': self._quantile(self.positive_histogram, 0.90),
                'p95': self._quantile(self.positive_histogram, 0.95),
                'p99': self._quantile(self.positive_histogram, 0.99),
            },
            'negative_magnitude_percentiles_mm_h': {
                'p90': self._quantile(self.negative_histogram, 0.90),
                'p95': self._quantile(self.negative_histogram, 0.95),
                'p99': self._quantile(self.negative_histogram, 0.99),
            },
            'absolute_exceedance_fraction': {
                str(level): self.above[level] / self.count
                for level in SPARSITY_LEVELS
            },
        }


def _edge_masks(previous_rain, target_rain):
    shape = previous_rain.shape
    wet = previous_rain >= 0.1
    flat = wet.reshape(-1, 1, *shape[-2:]).float()
    dilated = F.max_pool2d(flat, 3, stride=1, padding=1) > 0
    eroded = -F.max_pool2d(-flat, 3, stride=1, padding=1) > 0.5
    dilated = dilated.reshape(shape)
    eroded = eroded.reshape(shape)
    newborn = (~wet) & (target_rain >= 0.1)
    return {
        'all_pixels': torch.ones_like(wet),
        'active_union_0.1': (previous_rain >= 0.1) | (target_rain >= 0.1),
        'existing_interior': eroded,
        'previous_edge_band': dilated & ~eroded,
        'newborn_0.1': newborn,
        'strong_union_16': (previous_rain >= 16.0) | (target_rain >= 16.0),
        'extreme_union_32': (previous_rain >= 32.0) | (target_rain >= 32.0),
        'true_growth': target_rain > previous_rain + 0.1,
        'true_decay': target_rain + 0.1 < previous_rain,
    }


def _write_rows(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = create_parser()
    parser.add_argument('--oracle_source_diagnostic_dir', default=None)
    parser.add_argument('--oracle_source_split', choices=('train', 'val'),
                        default='val')
    args = parser.parse_args()
    if args.config_file is None or args.ckpt_path is None:
        raise ValueError('--config_file and --ckpt_path are required')
    update_config(args.__dict__, load_config(args.config_file),
                  exclude_keys=['method', 'val_batch_size'])
    for attribute, value in default_parser().items():
        if getattr(args, attribute, None) is None:
            setattr(args, attribute, value)

    experiment = BaseExperiment(args)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    experiment.method.load_state_dict(checkpoint['state_dict'])
    device = torch.device('cuda:' + str(args.gpus[0])
                          if torch.cuda.is_available() else 'cpu')
    method = experiment.method.to(device).eval()
    base_loader = (experiment.data.train_loader
                   if args.oracle_source_split == 'train'
                   else experiment.data.valid_loader)
    # A sequential loader is required so event_id_for_sample remains correct;
    # it also prevents the training loader's shuffle/drop_last policy from
    # changing the oracle distribution audit.
    loader = DataLoader(
        base_loader.dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True)
    dataset = loader.dataset
    operator = method.model.operator

    global_stats = StreamingSourceStats()
    lead_stats = [StreamingSourceStats() for _ in range(args.aft_seq_length)]
    event_stats = defaultdict(StreamingSourceStats)
    region_stats = defaultdict(StreamingSourceStats)
    event_region_stats = defaultdict(StreamingSourceStats)
    sample_offset = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_size = batch_x.shape[0]
            event_ids = [str(dataset.event_id_for_sample(sample_offset + index))
                         for index in range(batch_size)]
            sample_offset += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            result = method.model(batch_x, return_aux=True)
            previous = torch.cat((batch_x[:, -1:], batch_y[:, :-1]), dim=1)
            previous_rain = normalized_dbz_to_rain(
                previous, value_scale=operator.value_scale,
                zr_a=operator.zr_a, zr_b=operator.zr_b)
            target_rain = normalized_dbz_to_rain(
                batch_y, value_scale=operator.value_scale,
                zr_a=operator.zr_a, zr_b=operator.zr_b)
            advected_rain = torch.stack([
                backward_warp(
                    previous_rain[:, lead], result['flow'][:, lead],
                    align_corners=operator.align_corners,
                    padding_mode=operator.padding_mode)
                for lead in range(args.aft_seq_length)
            ], dim=1)
            oracle_source = target_rain - advected_rain
            masks = _edge_masks(previous_rain, target_rain)

            source_np = oracle_source.cpu().numpy()
            global_stats.update(source_np)
            for lead in range(args.aft_seq_length):
                lead_stats[lead].update(source_np[:, lead])
            for index, event_id in enumerate(event_ids):
                event_stats[event_id].update(source_np[index])
            for name, mask in masks.items():
                mask_np = mask.cpu().numpy()
                region_stats[name].update(source_np, mask_np)
                for index, event_id in enumerate(event_ids):
                    event_region_stats[(event_id, name)].update(
                        source_np[index], mask_np[index])

    report = {
        'protocol': {
            'checkpoint': args.ckpt_path,
            'sample_windows': sample_offset,
            'split': args.oracle_source_split,
            'lead_count': args.aft_seq_length,
            'lead_minutes': int(getattr(args, 'lead_minutes', 6)),
            'field_space': 'rain_rate',
            'oracle_definition': (
                'true_rain_t - warp(true_rain_t-1, predicted_flow_t)'),
            'units': 'mm/h increment per 6-minute evolution step',
            'teacher_forced': True,
        },
        'global': global_stats.report(),
        'periods': {},
        'regions': {name: stats.report()
                    for name, stats in sorted(region_stats.items())},
        'events': {event: stats.report()
                   for event, stats in sorted(event_stats.items())},
    }
    for name, start, end in (('0_1h', 0, 10), ('1_2h', 10, 20)):
        combined = StreamingSourceStats()
        # Recombine exact streaming accumulators by replaying histogram/count
        # fields without retaining the full validation tensor.
        for stats in lead_stats[start:end]:
            combined.count += stats.count
            combined.sum += stats.sum
            combined.abs_sum += stats.abs_sum
            combined.square_sum += stats.square_sum
            combined.minimum = min(combined.minimum, stats.minimum)
            combined.maximum = max(combined.maximum, stats.maximum)
            combined.positive_count += stats.positive_count
            combined.negative_count += stats.negative_count
            combined.zero_count += stats.zero_count
            combined.abs_histogram += stats.abs_histogram
            combined.positive_histogram += stats.positive_histogram
            combined.negative_histogram += stats.negative_histogram
            for level in SPARSITY_LEVELS:
                combined.above[level] += stats.above[level]
        report['periods'][name] = combined.report()

    active_p99 = report['regions']['active_union_0.1'][
        'absolute_percentiles_mm_h']['p99']
    report['recommended_source_bound'] = {
        'basis': 'P99 absolute oracle source on previous/target >= 0.1 mm/h union',
        'smax_mm_h': active_p99,
        'parameterization': 'source_rain = smax_mm_h * tanh(raw_source)',
    }

    output_dir = (args.oracle_source_diagnostic_dir
                  or osp.join(args.res_dir, args.ex_name, 'saved',
                              'oracle_source_diagnostics'))
    check_dir(output_dir)
    with open(osp.join(output_dir, 'oracle_source_summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    lead_rows = []
    for lead, stats in enumerate(lead_stats, start=1):
        row = {'lead': lead, 'minutes': lead * report['protocol']['lead_minutes']}
        flat = stats.report()
        for key in ('count', 'mean_mm_h', 'mean_absolute_mm_h', 'rmse_mm_h',
                    'positive_fraction', 'negative_fraction'):
            row[key] = flat.get(key)
        for key, value in flat.get('absolute_percentiles_mm_h', {}).items():
            row[f'absolute_{key}_mm_h'] = value
        lead_rows.append(row)
    _write_rows(osp.join(output_dir, 'oracle_source_by_lead.csv'), lead_rows)

    event_rows = []
    for event_id, stats in sorted(event_stats.items()):
        flat = stats.report()
        row = {'event_id': event_id}
        for key in ('count', 'mean_mm_h', 'mean_absolute_mm_h', 'rmse_mm_h',
                    'positive_fraction', 'negative_fraction'):
            row[key] = flat.get(key)
        for key, value in flat.get('absolute_percentiles_mm_h', {}).items():
            row[f'absolute_{key}_mm_h'] = value
        for region in ('strong_union_16', 'extreme_union_32', 'newborn_0.1'):
            region_report = event_region_stats[(event_id, region)].report()
            row[f'{region}_count'] = region_report.get('count', 0)
            row[f'{region}_mae_mm_h'] = region_report.get('mean_absolute_mm_h')
            row[f'{region}_p99_mm_h'] = region_report.get(
                'absolute_percentiles_mm_h', {}).get('p99')
        event_rows.append(row)
    _write_rows(osp.join(output_dir, 'oracle_source_by_event.csv'), event_rows)
    print(json.dumps({
        'output_dir': output_dir,
        'sample_windows': sample_offset,
        'recommended_smax_mm_h': active_p99,
    }, indent=2))


if __name__ == '__main__':
    main()
