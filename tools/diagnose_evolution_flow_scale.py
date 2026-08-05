"""No-training audit of EvolutionConvLSTM flow scale and persistent objects."""

import csv
import json
import math
import os.path as osp
import warnings
from collections import defaultdict, deque

warnings.filterwarnings('ignore')

import numpy as np
import torch

from openstl.api import BaseExperiment
from openstl.modules import warp_field
from openstl.utils import (check_dir, create_parser, default_parser,
                           load_config, update_config)


ALPHAS = (-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
THRESHOLDS = (16.0, 32.0)
MOTION_BINS = (
    ('near_static', 0.0, 0.2),
    ('subpixel', 0.2, 0.5),
    ('moving', 0.5, 1.0),
    ('fast_or_difficult', 1.0, float('inf')),
)


def _components(mask):
    """Return 4-connected components as Nx2 arrays in (y, x) order."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    objects = []
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        points = []
        while queue:
            py, px = queue.popleft()
            points.append((py, px))
            for ny, nx in ((py - 1, px), (py + 1, px),
                           (py, px - 1), (py, px + 1)):
                if (0 <= ny < height and 0 <= nx < width
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        objects.append(np.asarray(points, dtype=np.int16))
    return objects


def _centroid(points, rain):
    values = rain[points[:, 0], points[:, 1]].astype(np.float64)
    weights = np.maximum(values, 1e-6)
    # Return Cartesian image coordinates (x, y).
    return np.asarray([
        np.average(points[:, 1], weights=weights),
        np.average(points[:, 0], weights=weights),
    ])


def _object_record(points, rain):
    return {
        'points': points,
        'pixels': set(map(tuple, points.tolist())),
        'centroid': _centroid(points, rain),
        'area': len(points),
        'energy': float(rain[points[:, 0], points[:, 1]].sum()),
    }


def _persistent_pairs(previous_rain, current_rain, threshold):
    previous = [_object_record(p, previous_rain)
                for p in _components(previous_rain >= threshold)]
    current = [_object_record(p, current_rain)
               for p in _components(current_rain >= threshold)]
    candidates = []
    previous_counts = np.zeros(len(previous), dtype=np.int32)
    current_counts = np.zeros(len(current), dtype=np.int32)
    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            intersection = len(old['pixels'] & new['pixels'])
            union = len(old['pixels'] | new['pixels'])
            iou = intersection / union if union else 0.0
            distance = float(np.linalg.norm(new['centroid'] - old['centroid']))
            if iou >= 0.1 or distance <= 2.0:
                candidates.append((i, j, iou, distance))
                previous_counts[i] += 1
                current_counts[j] += 1
    pairs = []
    for i, j, iou, distance in candidates:
        # Exclude ambiguous split/merge candidates.
        if previous_counts[i] != 1 or current_counts[j] != 1:
            continue
        old, new = previous[i], current[j]
        area_ratio = new['area'] / old['area']
        energy_ratio = new['energy'] / max(old['energy'], 1e-6)
        if not (0.5 <= area_ratio <= 2.0 and 0.5 <= energy_ratio <= 2.0):
            continue
        pairs.append((old, new, iou, distance, area_ratio, energy_ratio))
    return pairs


def _new_field_state(leads):
    return {
        'abs': np.zeros(leads), 'sq': np.zeros(leads),
        'count': np.zeros(leads),
        'hits': np.zeros((len(THRESHOLDS), leads)),
        'false_alarms': np.zeros((len(THRESHOLDS), leads)),
        'misses': np.zeros((len(THRESHOLDS), leads)),
    }


def _update_field(state, prediction, truth, rain_prediction, rain_truth):
    error = prediction - truth
    state['abs'] += error.abs().sum(dim=(0, 2, 3, 4)).cpu().numpy()
    state['sq'] += error.square().sum(dim=(0, 2, 3, 4)).cpu().numpy()
    state['count'] += np.prod([prediction.shape[0], *prediction.shape[2:]])
    for index, threshold in enumerate(THRESHOLDS):
        pred = rain_prediction >= threshold
        true = rain_truth >= threshold
        axes = (0, 2, 3, 4)
        state['hits'][index] += (pred & true).sum(dim=axes).cpu().numpy()
        state['false_alarms'][index] += (pred & ~true).sum(dim=axes).cpu().numpy()
        state['misses'][index] += (~pred & true).sum(dim=axes).cpu().numpy()


def _ratio(a, b):
    return float(a / b) if b else None


def _field_report(state):
    report = {
        'mae': _ratio(state['abs'].sum(), state['count'].sum()),
        'rmse': math.sqrt(_ratio(state['sq'].sum(), state['count'].sum())),
        'lead_mae': np.divide(state['abs'], state['count']).tolist(),
        'thresholds': {},
    }
    for index, threshold in enumerate(THRESHOLDS):
        h, f, m = (state[key][index]
                   for key in ('hits', 'false_alarms', 'misses'))
        lead_csi = np.divide(
            h, h + f + m, out=np.full_like(h, np.nan),
            where=(h + f + m) > 0)
        lead_pod = np.divide(
            h, h + m, out=np.full_like(h, np.nan), where=(h + m) > 0)
        lead_far = np.divide(
            f, h + f, out=np.full_like(h, np.nan), where=(h + f) > 0)
        lead_bias = np.divide(
            h + f, h + m, out=np.full_like(h, np.nan), where=(h + m) > 0)
        report['thresholds'][str(threshold)] = {
            'csi': _ratio(h.sum(), (h + f + m).sum()),
            'pod': _ratio(h.sum(), (h + m).sum()),
            'far': _ratio(f.sum(), (h + f).sum()),
            'bias': _ratio((h + f).sum(), (h + m).sum()),
            'lead_csi': lead_csi.tolist(),
            'lead_pod': lead_pod.tolist(),
            'lead_far': lead_far.tolist(),
            'lead_bias': lead_bias.tolist(),
            'periods': {},
        }
        for label, start, end in (('0_1h', 0, 10), ('1_2h', 10, 20)):
            ph, pf, pm = h[start:end].sum(), f[start:end].sum(), m[start:end].sum()
            report['thresholds'][str(threshold)]['periods'][label] = {
                'csi': _ratio(ph, ph + pf + pm),
                'pod': _ratio(ph, ph + pm),
                'far': _ratio(pf, ph + pf),
                'bias': _ratio(ph + pf, ph + pm),
                'hits': float(ph),
                'false_alarms': float(pf),
                'misses': float(pm),
            }
    report['periods'] = {}
    for label, start, end in (('0_1h', 0, 10), ('1_2h', 10, 20)):
        report['periods'][label] = {
            'mae': _ratio(state['abs'][start:end].sum(),
                          state['count'][start:end].sum()),
            'rmse': math.sqrt(_ratio(state['sq'][start:end].sum(),
                                     state['count'][start:end].sum())),
        }
    return report


def _summarize_objects(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['threshold'], row['motion_bin'])].append(row)
    report = {}
    for (threshold, motion_bin), items in sorted(grouped.items()):
        key = f'{threshold:g}/{motion_bin}'
        cosines = [x['direction_cosine'] for x in items
                   if x['direction_cosine'] is not None]
        ratios = [x['magnitude_ratio'] for x in items
                  if x['magnitude_ratio'] is not None]
        entry = {
            'count': len(items),
            'true_displacement_mean_pixels': float(np.mean(
                [x['true_displacement'] for x in items])),
            'flow_magnitude_mean_pixels': float(np.mean(
                [x['flow_magnitude'] for x in items])),
            'direction_cosine_mean': float(np.mean(cosines)) if cosines else None,
            'direction_cosine_positive_fraction': (
                float(np.mean(np.asarray(cosines) > 0)) if cosines else None),
            'magnitude_ratio_median': float(np.median(ratios)) if ratios else None,
            'endpoint_error_by_alpha': {},
        }
        gate_values = [x['gate_mean'] for x in items
                       if x.get('gate_mean') is not None]
        entry['gate_mean'] = (float(np.mean(gate_values))
                              if gate_values else None)
        for alpha in ALPHAS:
            errors = [x[f'epe_alpha_{alpha:g}'] for x in items]
            entry['endpoint_error_by_alpha'][str(alpha)] = float(np.mean(errors))
        best = min(entry['endpoint_error_by_alpha'],
                   key=entry['endpoint_error_by_alpha'].get)
        entry['best_alpha_by_endpoint_error'] = best
        report[key] = entry
    return report


def main():
    parser = create_parser()
    parser.add_argument('--flow_scale_diagnostic_dir', default=None)
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
    loader = experiment.data.valid_loader
    dataset = loader.dataset
    field_states = {alpha: _new_field_state(args.aft_seq_length)
                    for alpha in ALPHAS}
    event_states = defaultdict(
        lambda: {alpha: _new_field_state(args.aft_seq_length)
                 for alpha in ALPHAS})
    object_rows = []
    sample_offset = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_size = batch_x.shape[0]
            event_ids = [dataset.event_id_for_sample(sample_offset + i)
                         for i in range(batch_size)]
            sample_ids = [f'{event_ids[i]}-{sample_offset + i:06d}'
                          for i in range(batch_size)]
            sample_offset += batch_size
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            model_result = method.model(batch_x, return_aux=True)
            flow = model_result['flow']
            flow_gate = model_result.get('flow_gate')
            previous = torch.cat((batch_x[:, -1:], batch_y[:, :-1]), dim=1)
            rain_truth = method._to_precipitation(batch_y)
            rain_previous = method._to_precipitation(previous)
            scaled_predictions = {}
            for alpha in ALPHAS:
                prediction = torch.stack([
                    warp_field(
                        previous[:, lead], alpha * flow[:, lead],
                        field_space='rain_rate', value_scale=args.radar_value_scale,
                        zr_a=args.zr_a, zr_b=args.zr_b,
                        align_corners=method.model.operator.align_corners,
                        padding_mode=method.model.operator.padding_mode)
                    for lead in range(args.aft_seq_length)], dim=1)
                scaled_predictions[alpha] = prediction
                rain_prediction = method._to_precipitation(prediction)
                _update_field(field_states[alpha], prediction, batch_y,
                              rain_prediction, rain_truth)
                for index, event_id in enumerate(event_ids):
                    _update_field(event_states[event_id][alpha],
                                  prediction[index:index + 1],
                                  batch_y[index:index + 1],
                                  rain_prediction[index:index + 1],
                                  rain_truth[index:index + 1])

            previous_np = rain_previous[:, :, 0].cpu().numpy()
            truth_np = rain_truth[:, :, 0].cpu().numpy()
            flow_np = flow.cpu().numpy()
            gate_np = flow_gate.cpu().numpy() if flow_gate is not None else None
            for sample in range(batch_size):
                for lead in range(args.aft_seq_length):
                    for threshold in THRESHOLDS:
                        pairs = _persistent_pairs(
                            previous_np[sample, lead], truth_np[sample, lead],
                            threshold)
                        for object_index, (old, new, iou, _, area_ratio,
                                           energy_ratio) in enumerate(pairs):
                            points = old['points']
                            weights = previous_np[sample, lead,
                                                  points[:, 0], points[:, 1]]
                            weights = np.maximum(weights, 1e-6)
                            object_flow = np.asarray([
                                np.average(flow_np[sample, lead, 0,
                                                   points[:, 0], points[:, 1]],
                                           weights=weights),
                                np.average(flow_np[sample, lead, 1,
                                                   points[:, 0], points[:, 1]],
                                           weights=weights),
                            ])
                            displacement = new['centroid'] - old['centroid']
                            true_norm = float(np.linalg.norm(displacement))
                            flow_norm = float(np.linalg.norm(object_flow))
                            cosine = None
                            magnitude_ratio = None
                            if true_norm >= 0.2 and flow_norm > 1e-8:
                                cosine = float(np.dot(object_flow, displacement)
                                               / (flow_norm * true_norm))
                                magnitude_ratio = flow_norm / true_norm
                            motion_bin = next(name for name, low, high in MOTION_BINS
                                              if low <= true_norm < high)
                            row = {
                                'sample_id': sample_ids[sample],
                                'event_id': event_ids[sample],
                                'lead_index': lead + 1,
                                'lead_minutes': (lead + 1) * int(args.lead_minutes),
                                'threshold': threshold,
                                'object_index': object_index,
                                'motion_bin': motion_bin,
                                'previous_area': old['area'],
                                'current_area': new['area'],
                                'area_ratio': area_ratio,
                                'energy_ratio': energy_ratio,
                                'match_iou': iou,
                                'true_dx': float(displacement[0]),
                                'true_dy': float(displacement[1]),
                                'true_displacement': true_norm,
                                'flow_dx': float(object_flow[0]),
                                'flow_dy': float(object_flow[1]),
                                'flow_magnitude': flow_norm,
                                'direction_cosine': cosine,
                                'magnitude_ratio': magnitude_ratio,
                                'gate_mean': (float(np.average(
                                    gate_np[sample, lead, 0,
                                            points[:, 0], points[:, 1]],
                                    weights=weights))
                                    if gate_np is not None else None),
                            }
                            for alpha in ALPHAS:
                                row[f'epe_alpha_{alpha:g}'] = float(np.linalg.norm(
                                    alpha * object_flow - displacement))
                            object_rows.append(row)

    report = {
        'experiment': args.ex_name,
        'checkpoint': args.ckpt_path,
        'sample_count': len(dataset),
        'alphas': list(ALPHAS),
        'flow_convention': 'dx>0 right, dy>0 down; pixels per 6 minutes',
        'full_field': {str(alpha): _field_report(field_states[alpha])
                       for alpha in ALPHAS},
        'events': {
            event_id: {str(alpha): _field_report(states[alpha])
                       for alpha in ALPHAS}
            for event_id, states in sorted(event_states.items())
        },
        'persistent_objects': _summarize_objects(object_rows),
        'persistent_object_count': len(object_rows),
    }
    output_dir = check_dir(args.flow_scale_diagnostic_dir or osp.join(
        args.res_dir, args.ex_name, 'saved', 'flow_scale_diagnostics'))
    with open(osp.join(output_dir, 'flow_scale_diagnostics.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    if object_rows:
        with open(osp.join(output_dir, 'persistent_objects.csv'), 'w',
                  newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(object_rows[0]))
            writer.writeheader()
            writer.writerows(object_rows)
    with open(osp.join(output_dir, 'alpha_summary.csv'), 'w', newline='',
              encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['alpha', 'mae', 'csi16', 'csi32'])
        for alpha in ALPHAS:
            values = report['full_field'][str(alpha)]
            writer.writerow([
                alpha, values['mae'],
                values['thresholds']['16.0']['csi'],
                values['thresholds']['32.0']['csi'],
            ])
    print(json.dumps({
        'output_dir': output_dir,
        'persistent_object_count': len(object_rows),
        'alpha_summary': {
            str(alpha): {
                'mae': report['full_field'][str(alpha)]['mae'],
                'csi16': report['full_field'][str(alpha)]['thresholds']['16.0']['csi'],
                'csi32': report['full_field'][str(alpha)]['thresholds']['32.0']['csi'],
            } for alpha in ALPHAS
        },
    }, indent=2))


if __name__ == '__main__':
    main()
