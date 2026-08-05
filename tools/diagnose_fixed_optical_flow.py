"""History-only Farneback R4-a baseline versus zero and learned flow."""

import json
import os.path as osp
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch

from openstl.api import BaseExperiment
from openstl.modules import warp_field
from openstl.utils import (check_dir, create_parser, default_parser,
                           load_config, update_config)
from tools.diagnose_evolution_flow_scale import (
    MOTION_BINS, THRESHOLDS, _field_report, _new_field_state,
    _persistent_pairs, _update_field)


MODES = ('zero', 'farneback_last', 'farneback_median3',
         'learned_0.25', 'learned_0.5', 'learned_1.0')


def _farneback(previous, current):
    previous = np.clip(previous * 255.0, 0, 255).astype(np.uint8)
    current = np.clip(current * 255.0, 0, 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(
        previous, current, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def _history_flows(batch_x):
    """Return last-pair and median-three forward flows as Bx2xHxW tensors."""
    values = batch_x[:, :, 0].detach().cpu().numpy()
    last, median = [], []
    for sample in values:
        flows = [_farneback(sample[index], sample[index + 1])
                 for index in range(sample.shape[0] - 4, sample.shape[0] - 1)]
        last.append(flows[-1].transpose(2, 0, 1))
        median.append(np.median(np.stack(flows), axis=0).transpose(2, 0, 1))
    return (torch.from_numpy(np.stack(last)).to(batch_x.device, batch_x.dtype),
            torch.from_numpy(np.stack(median)).to(batch_x.device, batch_x.dtype))


def _object_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['threshold'], row['motion_bin'], row['mode'])].append(row)
    report = {}
    for (threshold, motion_bin, mode), items in sorted(grouped.items()):
        cosines = [item['direction_cosine'] for item in items
                   if item['direction_cosine'] is not None]
        report[f'{threshold:g}/{motion_bin}/{mode}'] = {
            'count': len(items),
            'endpoint_error_mean': float(np.mean(
                [item['endpoint_error'] for item in items])),
            'flow_magnitude_mean': float(np.mean(
                [item['flow_magnitude'] for item in items])),
            'direction_cosine_mean': float(np.mean(cosines)) if cosines else None,
            'direction_positive_fraction': (
                float(np.mean(np.asarray(cosines) > 0)) if cosines else None),
        }
    return report


def main():
    parser = create_parser()
    parser.add_argument('--fixed_flow_diagnostic_dir', default=None)
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
    states = {mode: _new_field_state(args.aft_seq_length) for mode in MODES}
    event_states = defaultdict(
        lambda: {mode: _new_field_state(args.aft_seq_length) for mode in MODES})
    object_rows = []
    sample_offset = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_size = batch_x.shape[0]
            event_ids = [loader.dataset.event_id_for_sample(sample_offset + i)
                         for i in range(batch_size)]
            sample_offset += batch_size
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            learned = method.model(batch_x, return_aux=True)['flow']
            fixed_last, fixed_median = _history_flows(batch_x)
            flow_modes = {
                'zero': torch.zeros_like(learned),
                'farneback_last': fixed_last[:, None].expand_as(learned),
                'farneback_median3': fixed_median[:, None].expand_as(learned),
                'learned_0.25': 0.25 * learned,
                'learned_0.5': 0.5 * learned,
                'learned_1.0': learned,
            }
            previous = torch.cat((batch_x[:, -1:], batch_y[:, :-1]), dim=1)
            rain_previous = method._to_precipitation(previous)
            rain_truth = method._to_precipitation(batch_y)
            for mode, flow in flow_modes.items():
                prediction = torch.stack([
                    warp_field(
                        previous[:, lead], flow[:, lead],
                        field_space='rain_rate', value_scale=args.radar_value_scale,
                        zr_a=args.zr_a, zr_b=args.zr_b,
                        align_corners=method.model.operator.align_corners,
                        padding_mode=method.model.operator.padding_mode)
                    for lead in range(args.aft_seq_length)], dim=1)
                rain_prediction = method._to_precipitation(prediction)
                _update_field(states[mode], prediction, batch_y,
                              rain_prediction, rain_truth)
                for index, event_id in enumerate(event_ids):
                    _update_field(event_states[event_id][mode],
                                  prediction[index:index + 1],
                                  batch_y[index:index + 1],
                                  rain_prediction[index:index + 1],
                                  rain_truth[index:index + 1])

            previous_np = rain_previous[:, :, 0].cpu().numpy()
            truth_np = rain_truth[:, :, 0].cpu().numpy()
            flow_np = {mode: flow.cpu().numpy()
                       for mode, flow in flow_modes.items()}
            for sample in range(batch_size):
                for lead in range(args.aft_seq_length):
                    for threshold in THRESHOLDS:
                        pairs = _persistent_pairs(
                            previous_np[sample, lead], truth_np[sample, lead],
                            threshold)
                        for old, new, _, _, _, _ in pairs:
                            displacement = new['centroid'] - old['centroid']
                            true_norm = float(np.linalg.norm(displacement))
                            motion_bin = next(name for name, low, high in MOTION_BINS
                                              if low <= true_norm < high)
                            points = old['points']
                            weights = np.maximum(previous_np[
                                sample, lead, points[:, 0], points[:, 1]], 1e-6)
                            for mode in MODES:
                                values = flow_np[mode][sample, lead]
                                object_flow = np.asarray([
                                    np.average(values[0, points[:, 0], points[:, 1]],
                                               weights=weights),
                                    np.average(values[1, points[:, 0], points[:, 1]],
                                               weights=weights),
                                ])
                                flow_norm = float(np.linalg.norm(object_flow))
                                cosine = None
                                if true_norm >= 0.2 and flow_norm > 1e-8:
                                    cosine = float(np.dot(object_flow, displacement)
                                                   / (flow_norm * true_norm))
                                object_rows.append({
                                    'threshold': threshold,
                                    'motion_bin': motion_bin,
                                    'mode': mode,
                                    'flow_magnitude': flow_norm,
                                    'direction_cosine': cosine,
                                    'endpoint_error': float(np.linalg.norm(
                                        object_flow - displacement)),
                                })

    report = {
        'experiment': args.ex_name,
        'checkpoint': args.ckpt_path,
        'sample_count': len(loader.dataset),
        'farneback': {
            'pyr_scale': 0.5, 'levels': 3, 'winsize': 15,
            'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2,
            'history_only': True,
            'last': 'last observed frame pair',
            'median3': 'pixelwise median of last three observed frame-pair flows',
        },
        'full_field': {mode: _field_report(state)
                       for mode, state in states.items()},
        'events': {
            event_id: {mode: _field_report(state)
                       for mode, state in modes.items()}
            for event_id, modes in sorted(event_states.items())
        },
        'persistent_objects': _object_summary(object_rows),
    }
    output_dir = check_dir(args.fixed_flow_diagnostic_dir or osp.join(
        args.res_dir, args.ex_name, 'saved', 'fixed_flow_diagnostics'))
    with open(osp.join(output_dir, 'fixed_flow_diagnostics.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({
        'output_dir': output_dir,
        'modes': {mode: {
            'mae': values['mae'],
            'csi16': values['thresholds']['16.0']['csi'],
            'csi32': values['thresholds']['32.0']['csi'],
        } for mode, values in report['full_field'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
