"""Diagnose learned EvolutionConvLSTM transport independently of source terms."""

import csv
import json
import os.path as osp
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F

from openstl.api import BaseExperiment
from openstl.modules import backward_warp
from openstl.utils import (check_dir, create_parser, default_parser,
                           load_config, update_config)


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def _mode_state(leads, thresholds, windows):
    return {
        'abs': np.zeros(leads), 'sq': np.zeros(leads),
        'count': np.zeros(leads),
        'hits': np.zeros((len(thresholds), leads)),
        'false_alarms': np.zeros((len(thresholds), leads)),
        'misses': np.zeros((len(thresholds), leads)),
        'pred_area': np.zeros((len(thresholds), leads)),
        'true_area': np.zeros((len(thresholds), leads)),
        'fss_error': {window: np.zeros((len(thresholds), leads))
                      for window in windows},
        'fss_reference': {window: np.zeros((len(thresholds), leads))
                          for window in windows},
        'centroid_sum': np.zeros((len(thresholds), leads)),
        'centroid_count': np.zeros((len(thresholds), leads)),
    }


def _update_mode(state, prediction, truth, rain_prediction, rain_truth,
                 thresholds, windows, grid_spacing_km):
    error = prediction - truth
    state['abs'] += error.abs().sum(dim=(0, 2, 3, 4)).cpu().numpy()
    state['sq'] += error.square().sum(dim=(0, 2, 3, 4)).cpu().numpy()
    state['count'] += np.prod([prediction.shape[0], *prediction.shape[2:]])
    batch, leads = prediction.shape[:2]
    for threshold_index, threshold in enumerate(thresholds):
        pred_event = rain_prediction >= threshold
        true_event = rain_truth >= threshold
        axes = (0, 2, 3, 4)
        state['hits'][threshold_index] += (
            pred_event & true_event).sum(dim=axes).cpu().numpy()
        state['false_alarms'][threshold_index] += (
            pred_event & ~true_event).sum(dim=axes).cpu().numpy()
        state['misses'][threshold_index] += (
            ~pred_event & true_event).sum(dim=axes).cpu().numpy()
        state['pred_area'][threshold_index] += pred_event.sum(
            dim=axes).cpu().numpy()
        state['true_area'][threshold_index] += true_event.sum(
            dim=axes).cpu().numpy()
        for window in windows:
            pred_fraction = F.avg_pool2d(
                pred_event.reshape(batch * leads, 1, *prediction.shape[-2:]).float(),
                window, stride=1, padding=window // 2)
            true_fraction = F.avg_pool2d(
                true_event.reshape(batch * leads, 1, *prediction.shape[-2:]).float(),
                window, stride=1, padding=window // 2)
            fss_error = (pred_fraction - true_fraction).square().sum(
                dim=(1, 2, 3)).reshape(batch, leads).sum(dim=0)
            fss_reference = (pred_fraction.square() + true_fraction.square()).sum(
                dim=(1, 2, 3)).reshape(batch, leads).sum(dim=0)
            state['fss_error'][window][threshold_index] += fss_error.cpu().numpy()
            state['fss_reference'][window][threshold_index] += fss_reference.cpu().numpy()
        for sample in range(batch):
            for lead in range(leads):
                pred_points = torch.nonzero(pred_event[sample, lead, 0])
                true_points = torch.nonzero(true_event[sample, lead, 0])
                if len(pred_points) and len(true_points):
                    distance = torch.linalg.vector_norm(
                        pred_points.float().mean(0)
                        - true_points.float().mean(0)).item()
                    state['centroid_sum'][threshold_index, lead] += (
                        distance * grid_spacing_km)
                    state['centroid_count'][threshold_index, lead] += 1


def _report_mode(state, thresholds, windows):
    report = {
        'lead_mae_normalized': (state['abs'] / state['count']).tolist(),
        'lead_rmse_normalized': np.sqrt(state['sq'] / state['count']).tolist(),
        'overall_mae_normalized': _safe_ratio(state['abs'].sum(), state['count'].sum()),
        'overall_rmse_normalized': np.sqrt(
            _safe_ratio(state['sq'].sum(), state['count'].sum())),
        'thresholds': {},
    }
    for index, threshold in enumerate(thresholds):
        hits = state['hits'][index]
        false_alarms = state['false_alarms'][index]
        misses = state['misses'][index]
        entry = {
            'csi': _safe_ratio(hits.sum(), (hits + false_alarms + misses).sum()),
            'pod': _safe_ratio(hits.sum(), (hits + misses).sum()),
            'far': _safe_ratio(false_alarms.sum(), (hits + false_alarms).sum()),
            'bias': _safe_ratio(
                (hits + false_alarms).sum(), (hits + misses).sum()),
            'area_ratio': _safe_ratio(
                state['pred_area'][index].sum(), state['true_area'][index].sum()),
            'lead_centroid_error_km': np.divide(
                state['centroid_sum'][index], state['centroid_count'][index],
                out=np.full_like(state['centroid_sum'][index], np.nan),
                where=state['centroid_count'][index] > 0).tolist(),
        }
        entry['centroid_error_km'] = _safe_ratio(
            state['centroid_sum'][index].sum(),
            state['centroid_count'][index].sum())
        for window in windows:
            error = state['fss_error'][window][index]
            reference = state['fss_reference'][window][index]
            entry[f'fss_{window}x{window}'] = 1.0 - _safe_ratio(
                error.sum(), reference.sum())
            entry[f'lead_fss_{window}x{window}'] = np.divide(
                reference - error, reference,
                out=np.full_like(reference, np.nan), where=reference > 0).tolist()
        report['thresholds'][str(threshold)] = entry
    return report


def main():
    parser = create_parser()
    parser.add_argument('--motion_diagnostic_dir', default=None)
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
    thresholds = [16.0, 32.0]
    windows = [1, 3, 5]
    leads = args.aft_seq_length
    states = {name: _mode_state(leads, thresholds, windows)
              for name in ('recursive', 'cumulative_single_warp',
                           'teacher_forced',
                           'teacher_forced_zero_flow')}
    flow_accumulators = {
        key: np.zeros(leads) for key in (
            'dx_sum', 'dy_sum', 'magnitude_sum', 'magnitude_sq_sum',
            'max_sum', 'p95_sum', 'saturation_90_sum', 'saturation_99_sum',
            'spatial_tv_sum', 'temporal_delta_sum', 'batches')}
    sample = None

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            result = method.model(batch_x, return_aux=True)
            flow = result['flow']
            cumulative_flow = flow.cumsum(dim=1)
            cumulative_prediction = torch.stack([
                backward_warp(batch_x[:, -1], cumulative_flow[:, lead],
                              method.model.operator.align_corners,
                              method.model.operator.padding_mode)
                for lead in range(leads)], dim=1)
            previous_truth = torch.cat((batch_x[:, -1:], batch_y[:, :-1]), dim=1)
            teacher_prediction = torch.stack([
                backward_warp(previous_truth[:, lead], flow[:, lead],
                              method.model.operator.align_corners,
                              method.model.operator.padding_mode)
                for lead in range(leads)], dim=1)
            predictions = {
                'recursive': result['prediction'],
                'cumulative_single_warp': cumulative_prediction,
                'teacher_forced': teacher_prediction,
                'teacher_forced_zero_flow': previous_truth,
            }
            rain_truth = method._to_precipitation(batch_y)
            for name, prediction in predictions.items():
                rain_prediction = method._to_precipitation(prediction)
                _update_mode(
                    states[name], prediction, batch_y, rain_prediction,
                    rain_truth, thresholds, windows,
                    float(args.grid_spacing_km))

            magnitude = torch.linalg.vector_norm(flow, dim=2)
            for lead in range(leads):
                values = magnitude[:, lead].flatten()
                flow_accumulators['dx_sum'][lead] += flow[:, lead, 0].mean().item()
                flow_accumulators['dy_sum'][lead] += flow[:, lead, 1].mean().item()
                flow_accumulators['magnitude_sum'][lead] += values.mean().item()
                flow_accumulators['magnitude_sq_sum'][lead] += values.square().mean().item()
                flow_accumulators['max_sum'][lead] += values.max().item()
                flow_accumulators['p95_sum'][lead] += torch.quantile(values, 0.95).item()
                limit = float(method.model.max_displacement)
                component_max = flow[:, lead].abs().amax(dim=1)
                flow_accumulators['saturation_90_sum'][lead] += (
                    component_max >= 0.9 * limit).float().mean().item()
                flow_accumulators['saturation_99_sum'][lead] += (
                    component_max >= 0.99 * limit).float().mean().item()
                spatial = ((flow[:, lead, :, :, 1:] - flow[:, lead, :, :, :-1]).abs().mean()
                           + (flow[:, lead, :, 1:, :] - flow[:, lead, :, :-1, :]).abs().mean())
                flow_accumulators['spatial_tv_sum'][lead] += spatial.item()
                if lead:
                    flow_accumulators['temporal_delta_sum'][lead] += (
                        flow[:, lead] - flow[:, lead - 1]).abs().mean().item()
                flow_accumulators['batches'][lead] += 1
            if sample is None:
                sample = {
                    'last_input': batch_x[:2, -1].cpu().numpy(),
                    'truth': batch_y[:2].cpu().numpy(),
                    'flow': flow[:2].cpu().numpy(),
                    **{name: value[:2].cpu().numpy()
                       for name, value in predictions.items()},
                }

    count = flow_accumulators.pop('batches')
    flow_report = {}
    means = {}
    for key, values in flow_accumulators.items():
        means[key.removesuffix('_sum')] = (values / count).tolist()
    magnitude_mean = np.asarray(means['magnitude'])
    magnitude_square = flow_accumulators['magnitude_sq_sum'] / count
    means['magnitude_std'] = np.sqrt(
        np.maximum(magnitude_square - magnitude_mean ** 2, 0)).tolist()
    means.pop('magnitude_sq', None)
    flow_report['per_lead'] = means
    flow_report['max_displacement_per_component'] = float(
        method.model.max_displacement)
    flow_report['convention'] = (
        'dx>0 moves content right; dy>0 moves content down; pixels per 6 min')

    report = {
        'experiment': args.ex_name,
        'checkpoint': args.ckpt_path,
        'sample_count': len(loader.dataset),
        'flow': flow_report,
        'modes': {name: _report_mode(state, thresholds, windows)
                  for name, state in states.items()},
        'interpretation': {
            'recursive': '20 sequential resampling operations',
            'cumulative_single_warp': 'sum incremental flow; one resampling from X0 per lead',
            'teacher_forced': 'warp true previous frame; isolates one-step flow quality',
            'teacher_forced_zero_flow': (
                'true previous frame without warp; control for six-minute persistence'),
        },
    }
    output_dir = check_dir(args.motion_diagnostic_dir or osp.join(
        args.res_dir, args.ex_name, 'saved', 'motion_diagnostics'))
    with open(osp.join(output_dir, 'motion_diagnostics.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with open(osp.join(output_dir, 'flow_by_lead.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        keys = list(flow_report['per_lead'])
        writer.writerow(['lead_index', 'lead_minutes', *keys])
        for lead in range(leads):
            writer.writerow([lead + 1, (lead + 1) * int(args.lead_minutes),
                             *[flow_report['per_lead'][key][lead] for key in keys]])
    np.savez_compressed(osp.join(output_dir, 'sample_motion_fields.npz'), **sample)
    print(json.dumps({
        'output_dir': output_dir,
        'flow_mean_magnitude': float(np.mean(magnitude_mean)),
        'flow_max_mean': float(np.mean(means['max'])),
        'saturation_90': float(np.mean(means['saturation_90'])),
        'modes': {name: {
            'mae': values['overall_mae_normalized'],
            'rmse': values['overall_rmse_normalized'],
            'csi16': values['thresholds']['16.0']['csi'],
            'csi32': values['thresholds']['32.0']['csi'],
        } for name, values in report['modes'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
