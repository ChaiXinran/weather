"""Streaming, multi-threshold evaluation for precipitation nowcasting."""

import csv
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


RADAR_COLORS = [
    '#0000f6', '#01a0f6', '#00ecec', '#01ff00', '#00c800',
    '#019000', '#ffff00', '#e7c000', '#ff9000', '#ff0000', '#d60000',
]
RADAR_BOUNDS = np.arange(0, 60, 5)


def _safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


class _MetricState:

    def __init__(self, lead_count, threshold_count):
        self.abs_sum = np.zeros(lead_count, dtype=np.float64)
        self.sq_sum = np.zeros(lead_count, dtype=np.float64)
        self.error_sum = np.zeros(lead_count, dtype=np.float64)
        self.pred_sum = np.zeros(lead_count, dtype=np.float64)
        self.true_sum = np.zeros(lead_count, dtype=np.float64)
        self.valid_count = np.zeros(lead_count, dtype=np.int64)
        shape = (threshold_count, lead_count)
        self.hits = np.zeros(shape, dtype=np.int64)
        self.false_alarms = np.zeros(shape, dtype=np.int64)
        self.misses = np.zeros(shape, dtype=np.int64)
        self.correct_negatives = np.zeros(shape, dtype=np.int64)
        self.sample_count = 0

    def update(self, pred, true, thresholds, valid_mask):
        error = pred - true
        self.abs_sum += np.where(valid_mask, np.abs(error), 0).sum(
            axis=(0, 2, 3, 4))
        self.sq_sum += np.where(valid_mask, error ** 2, 0).sum(
            axis=(0, 2, 3, 4))
        self.error_sum += np.where(valid_mask, error, 0).sum(
            axis=(0, 2, 3, 4))
        self.pred_sum += np.where(valid_mask, pred, 0).sum(
            axis=(0, 2, 3, 4))
        self.true_sum += np.where(valid_mask, true, 0).sum(
            axis=(0, 2, 3, 4))
        self.valid_count += valid_mask.sum(axis=(0, 2, 3, 4))
        for index, threshold in enumerate(thresholds):
            pred_event = (pred >= threshold) & valid_mask
            true_event = (true >= threshold) & valid_mask
            self.hits[index] += (pred_event & true_event).sum(
                axis=(0, 2, 3, 4))
            self.false_alarms[index] += (pred_event & ~true_event).sum(
                axis=(0, 2, 3, 4))
            self.misses[index] += (~pred_event & true_event).sum(
                axis=(0, 2, 3, 4))
            self.correct_negatives[index] += (
                ~pred_event & ~true_event & valid_mask).sum(
                    axis=(0, 2, 3, 4))
        self.sample_count += pred.shape[0]

    def report(self, thresholds):
        mae = _safe_ratio(self.abs_sum, self.valid_count)
        rmse = np.sqrt(_safe_ratio(self.sq_sum, self.valid_count))
        mean_error = _safe_ratio(self.error_sum, self.valid_count)
        intensity_ratio = _safe_ratio(self.pred_sum, self.true_sum)
        report = {
            'sample_count': self.sample_count,
            'lead_time': {
                'mae': mae,
                'rmse': rmse,
                'mean_error': mean_error,
                'intensity_ratio': intensity_ratio,
                'thresholds': {},
            },
            'overall': {
                'mae': _safe_ratio(self.abs_sum.sum(), self.valid_count.sum()),
                'rmse': np.sqrt(
                    _safe_ratio(self.sq_sum.sum(), self.valid_count.sum())),
                'mean_error': _safe_ratio(
                    self.error_sum.sum(), self.valid_count.sum()),
                'intensity_ratio': _safe_ratio(
                    self.pred_sum.sum(), self.true_sum.sum()),
                'thresholds': {},
            },
            'periods': {},
        }
        for index, threshold in enumerate(thresholds):
            hits = self.hits[index]
            false_alarms = self.false_alarms[index]
            misses = self.misses[index]
            correct_negatives = self.correct_negatives[index]
            hss_numerator = 2 * (
                hits * correct_negatives - false_alarms * misses)
            hss_denominator = (
                (hits + misses) * (misses + correct_negatives)
                + (hits + false_alarms)
                * (false_alarms + correct_negatives))
            values = {
                'csi': _safe_ratio(hits, hits + false_alarms + misses),
                'pod': _safe_ratio(hits, hits + misses),
                'far': _safe_ratio(false_alarms, hits + false_alarms),
                'bias': _safe_ratio(hits + false_alarms, hits + misses),
                'hss': _safe_ratio(hss_numerator, hss_denominator),
                'hits': hits,
                'false_alarms': false_alarms,
                'misses': misses,
                'correct_negatives': correct_negatives,
            }
            report['lead_time']['thresholds'][str(threshold)] = values
            total_hits = hits.sum()
            total_false_alarms = false_alarms.sum()
            total_misses = misses.sum()
            total_correct_negatives = correct_negatives.sum()
            total_hss_numerator = 2 * (
                total_hits * total_correct_negatives
                - total_false_alarms * total_misses)
            total_hss_denominator = (
                (total_hits + total_misses)
                * (total_misses + total_correct_negatives)
                + (total_hits + total_false_alarms)
                * (total_false_alarms + total_correct_negatives))
            report['overall']['thresholds'][str(threshold)] = {
                'csi': _safe_ratio(
                    total_hits, total_hits + total_false_alarms + total_misses),
                'pod': _safe_ratio(total_hits, total_hits + total_misses),
                'far': _safe_ratio(
                    total_false_alarms, total_hits + total_false_alarms),
                'bias': _safe_ratio(
                    total_hits + total_false_alarms, total_hits + total_misses),
                'hss': _safe_ratio(
                    total_hss_numerator, total_hss_denominator),
                'hits': total_hits,
                'false_alarms': total_false_alarms,
                'misses': total_misses,
                'correct_negatives': total_correct_negatives,
            }
        period_width = min(10, len(mae))
        for start in range(0, len(mae), period_width):
            end = min(start + period_width, len(mae))
            label = f'lead_{start + 1:02d}_{end:02d}'
            period = {
                'mae': _safe_ratio(
                    self.abs_sum[start:end].sum(),
                    self.valid_count[start:end].sum()),
                'rmse': np.sqrt(_safe_ratio(
                    self.sq_sum[start:end].sum(),
                    self.valid_count[start:end].sum())),
                'mean_error': _safe_ratio(
                    self.error_sum[start:end].sum(),
                    self.valid_count[start:end].sum()),
                'intensity_ratio': _safe_ratio(
                    self.pred_sum[start:end].sum(),
                    self.true_sum[start:end].sum()),
                'thresholds': {},
            }
            for index, threshold in enumerate(thresholds):
                hits = self.hits[index, start:end].sum()
                false_alarms = self.false_alarms[index, start:end].sum()
                misses = self.misses[index, start:end].sum()
                correct_negatives = self.correct_negatives[
                    index, start:end].sum()
                hss_numerator = 2 * (
                    hits * correct_negatives - false_alarms * misses)
                hss_denominator = (
                    (hits + misses) * (misses + correct_negatives)
                    + (hits + false_alarms)
                    * (false_alarms + correct_negatives))
                period['thresholds'][str(threshold)] = {
                    'csi': _safe_ratio(
                        hits, hits + false_alarms + misses),
                    'pod': _safe_ratio(hits, hits + misses),
                    'far': _safe_ratio(false_alarms, hits + false_alarms),
                    'bias': _safe_ratio(
                        hits + false_alarms, hits + misses),
                    'hss': _safe_ratio(hss_numerator, hss_denominator),
                    'hits': hits,
                    'false_alarms': false_alarms,
                    'misses': misses,
                    'correct_negatives': correct_negatives,
                }
            report['periods'][label] = period
        return report


class PrecipitationEvaluator:
    """Accumulate normalized Radar predictions without retaining all samples."""

    def __init__(self,
                 lead_count,
                 thresholds=(20, 30, 35, 40, 45),
                 value_scale=50.0,
                 value_unit='dBZ',
                 lead_minutes=6,
                 clip_range=(0.0, 50.0),
                 case_threshold=35.0,
                 case_count=3,
                 event_id_source='date_proxy',
                 convert_dbz_to_rain=False,
                 zr_a=200.0,
                 zr_b=1.6,
                 wet_threshold=0.1,
                 grid_spacing_km=10.0,
                 neighborhood_windows=(1, 3, 5),
                 object_iou_threshold=0.1,
                 bootstrap_repetitions=2000,
                 bootstrap_seed=42,
                 true_is_rain=False):
        self.lead_count = int(lead_count)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.value_scale = float(value_scale)
        self.value_unit = value_unit
        self.lead_minutes = int(lead_minutes)
        self.clip_range = tuple(float(value) for value in clip_range)
        self.case_threshold = float(case_threshold)
        self.case_count = int(case_count)
        self.event_id_source = str(event_id_source)
        self.convert_dbz_to_rain = bool(convert_dbz_to_rain)
        self.zr_a = float(zr_a)
        self.zr_b = float(zr_b)
        self.wet_threshold = float(wet_threshold)
        self.grid_spacing_km = float(grid_spacing_km)
        self.neighborhood_windows = tuple(
            int(value) for value in neighborhood_windows)
        self.object_iou_threshold = float(object_iou_threshold)
        self.bootstrap_repetitions = int(bootstrap_repetitions)
        self.bootstrap_seed = int(bootstrap_seed)
        self.true_is_rain = bool(true_is_rain)
        if self.convert_dbz_to_rain and self.value_unit != 'mm/h':
            raise ValueError(
                'value_unit must be "mm/h" when convert_dbz_to_rain=True')
        if self.zr_a <= 0 or self.zr_b <= 0:
            raise ValueError('Frozen Z-R parameters zr_a and zr_b must be > 0')
        self.model = _MetricState(self.lead_count, len(self.thresholds))
        self.persistence = _MetricState(self.lead_count, len(self.thresholds))
        self.model_wet = _MetricState(self.lead_count, len(self.thresholds))
        self.persistence_wet = _MetricState(
            self.lead_count, len(self.thresholds))
        self.events = defaultdict(
            lambda: {
                'model': _MetricState(self.lead_count, len(self.thresholds)),
                'persistence': _MetricState(
                    self.lead_count, len(self.thresholds)),
            })
        self._best_cases = []
        self._worst_cases = []
        self._window_rows = []
        self._psd_bins = 20
        self._psd = {
            method: {
                'pred_sum': np.zeros(self._psd_bins, dtype=np.float64),
                'true_sum': np.zeros(self._psd_bins, dtype=np.float64),
                'count': np.zeros(self._psd_bins, dtype=np.int64),
            }
            for method in ('model', 'persistence')
        }

    def _physical(self, values):
        values = np.asarray(values, dtype=np.float32) * self.value_scale
        values = np.clip(values, self.clip_range[0], self.clip_range[1])
        if self.convert_dbz_to_rain:
            reflectivity = np.power(10.0, values / 10.0)
            return np.power(reflectivity / self.zr_a, 1.0 / self.zr_b)
        return values

    def _threshold_dbz(self, rain_rate):
        return (
            10.0 * math.log10(self.zr_a)
            + 10.0 * self.zr_b * math.log10(rain_rate))

    def update(self,
               pred,
               true,
               inputs,
               event_ids=None,
               sample_ids=None,
               valid_mask=None):
        pred = self._physical(pred)
        if self.true_is_rain:
            true = np.clip(
                np.asarray(true, dtype=np.float32) * 35.0, 0.0, 35.0)
        else:
            true = self._physical(true)
        inputs = self._physical(inputs)
        if pred.shape != true.shape or pred.ndim != 5:
            raise ValueError(
                'pred and true must share shape [N, T, C, H, W]')
        if pred.shape[1] != self.lead_count:
            raise ValueError(
                f'Expected {self.lead_count} lead times, got {pred.shape[1]}')
        if inputs.ndim != 5 or inputs.shape[0] != pred.shape[0]:
            raise ValueError('inputs must have shape [N, T_in, C, H, W]')
        persistence = np.repeat(
            inputs[:, -1:, ...], self.lead_count, axis=1)

        if valid_mask is None:
            valid_mask = np.isfinite(pred) & np.isfinite(true)
        else:
            valid_mask = np.broadcast_to(valid_mask, pred.shape)
            valid_mask = valid_mask & np.isfinite(pred) & np.isfinite(true)

        self.model.update(pred, true, self.thresholds, valid_mask)
        self.persistence.update(
            persistence, true, self.thresholds, valid_mask)
        wet_mask = valid_mask & (true > self.wet_threshold)
        self.model_wet.update(pred, true, self.thresholds, wet_mask)
        self.persistence_wet.update(
            persistence, true, self.thresholds, wet_mask)

        batch_size = pred.shape[0]
        if event_ids is None:
            event_ids = ['unassigned'] * batch_size
        if sample_ids is None:
            sample_ids = list(range(batch_size))
        if len(event_ids) != batch_size or len(sample_ids) != batch_size:
            raise ValueError('event_ids and sample_ids must match batch size')

        for index, (event_id, sample_id) in enumerate(
                zip(event_ids, sample_ids)):
            item_mask = valid_mask[index:index + 1]
            event = self.events[str(event_id)]
            event['model'].update(
                pred[index:index + 1], true[index:index + 1],
                self.thresholds, item_mask)
            event['persistence'].update(
                persistence[index:index + 1], true[index:index + 1],
                self.thresholds, item_mask)
            self._consider_case(
                sample_id, event_id, inputs[index], pred[index], true[index],
                persistence[index], item_mask[0])
            self._collect_window_metrics(
                sample_id, event_id, pred[index], true[index],
                persistence[index], item_mask[0])

    def _consider_case(self, sample_id, event_id, inputs, pred, true,
                       persistence, valid_mask):
        true_event = (true >= self.case_threshold) & valid_mask
        if not np.any(true_event):
            return
        model_rmse = np.sqrt(
            np.mean((pred[valid_mask] - true[valid_mask]) ** 2))
        persistence_rmse = np.sqrt(
            np.mean((persistence[valid_mask] - true[valid_mask]) ** 2))
        improvement = float(persistence_rmse - model_rmse)
        pred_event = (pred >= self.case_threshold) & valid_mask
        persistence_event = (
            (persistence >= self.case_threshold) & valid_mask)

        def sample_csi(candidate):
            hits = np.count_nonzero(candidate & true_event)
            false_alarms = np.count_nonzero(candidate & ~true_event & valid_mask)
            misses = np.count_nonzero(~candidate & true_event & valid_mask)
            denominator = hits + false_alarms + misses
            return hits / denominator if denominator else np.nan

        model_csi = sample_csi(pred_event)
        persistence_csi = sample_csi(persistence_event)
        csi_improvement = float(model_csi - persistence_csi)
        case = {
            'sample_id': str(sample_id),
            'event_id': str(event_id),
            'improvement_rmse': improvement,
            'model_rmse': float(model_rmse),
            'persistence_rmse': float(persistence_rmse),
            'case_threshold': self.case_threshold,
            'model_csi': model_csi,
            'persistence_csi': persistence_csi,
            'improvement_csi': csi_improvement,
            'inputs': inputs.copy(),
            'pred': pred.copy(),
            'true': true.copy(),
            'persistence': persistence.copy(),
        }
        ranking_score = csi_improvement + 1e-4 * improvement
        self._push_case(self._best_cases, ranking_score, case)
        self._push_case(self._worst_cases, -ranking_score, case)

    def _push_case(self, heap, score, case):
        item = (score, str(case['sample_id']), case)
        if len(heap) < self.case_count:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    @staticmethod
    def _fractions(binary, valid, window):
        """Return neighborhood event fractions without max-pooling."""
        if window == 1:
            return binary.astype(np.float64), valid.astype(np.float64)
        radius = window // 2
        padded_binary = np.pad(
            binary & valid, radius, mode='constant', constant_values=False)
        padded_valid = np.pad(
            valid, radius, mode='constant', constant_values=False)
        event_count = np.zeros(binary.shape, dtype=np.float64)
        valid_count = np.zeros(binary.shape, dtype=np.float64)
        for row_shift in range(window):
            for column_shift in range(window):
                slices = (
                    slice(row_shift, row_shift + binary.shape[0]),
                    slice(column_shift, column_shift + binary.shape[1]))
                event_count += padded_binary[slices]
                valid_count += padded_valid[slices]
        fractions = _safe_ratio(event_count, valid_count)
        return fractions, valid_count

    @staticmethod
    def _components(binary):
        """Extract 4-connected components as flat-index sets."""
        height, width = binary.shape
        seen = np.zeros_like(binary, dtype=bool)
        components = []
        for row, column in np.argwhere(binary):
            if seen[row, column]:
                continue
            stack = [(int(row), int(column))]
            seen[row, column] = True
            pixels = []
            while stack:
                current_row, current_column = stack.pop()
                pixels.append(current_row * width + current_column)
                for next_row, next_column in (
                        (current_row - 1, current_column),
                        (current_row + 1, current_column),
                        (current_row, current_column - 1),
                        (current_row, current_column + 1)):
                    if (0 <= next_row < height
                            and 0 <= next_column < width
                            and binary[next_row, next_column]
                            and not seen[next_row, next_column]):
                        seen[next_row, next_column] = True
                        stack.append((next_row, next_column))
            components.append(set(pixels))
        return components

    def _object_scores(self, pred_event, true_event,
                       pred_field, true_field):
        predicted = self._components(pred_event)
        observed = self._components(true_event)
        candidates = []
        for pred_index, pred_object in enumerate(predicted):
            for true_index, true_object in enumerate(observed):
                intersection = len(pred_object & true_object)
                union = len(pred_object | true_object)
                iou = intersection / union if union else 0.0
                if iou >= self.object_iou_threshold:
                    candidates.append((iou, pred_index, true_index))
        matched_predicted = set()
        matched_observed = set()
        matched_ious = []
        centroid_errors = []
        area_errors = []
        peak_errors = []
        width = pred_event.shape[1]
        for iou, pred_index, true_index in sorted(candidates, reverse=True):
            if (pred_index not in matched_predicted
                    and true_index not in matched_observed):
                matched_predicted.add(pred_index)
                matched_observed.add(true_index)
                matched_ious.append(iou)
                pred_pixels = predicted[pred_index]
                true_pixels = observed[true_index]
                pred_coordinates = np.asarray([
                    divmod(pixel, width) for pixel in pred_pixels])
                true_coordinates = np.asarray([
                    divmod(pixel, width) for pixel in true_pixels])
                centroid_errors.append(float(np.linalg.norm(
                    pred_coordinates.mean(axis=0)
                    - true_coordinates.mean(axis=0))
                    * self.grid_spacing_km))
                area_errors.append(len(pred_pixels) - len(true_pixels))
                pred_rows, pred_columns = pred_coordinates.T
                true_rows, true_columns = true_coordinates.T
                peak_errors.append(float(
                    pred_field[pred_rows, pred_columns].max()
                    - true_field[true_rows, true_columns].max()))
        object_hits = len(matched_ious)
        object_false_alarms = len(predicted) - object_hits
        object_misses = len(observed) - object_hits
        return {
            'object_hits': object_hits,
            'object_false_alarms': object_false_alarms,
            'object_misses': object_misses,
            'object_pod': _safe_ratio(
                object_hits, object_hits + object_misses),
            'object_far': _safe_ratio(
                object_false_alarms,
                object_hits + object_false_alarms),
            'mean_matched_iou': (
                float(np.mean(matched_ious)) if matched_ious else np.nan),
            'object_centroid_error_km': (
                float(np.mean(centroid_errors))
                if centroid_errors else np.nan),
            'object_area_error_pixels': (
                float(np.mean(area_errors)) if area_errors else np.nan),
            'object_peak_error': (
                float(np.mean(peak_errors)) if peak_errors else np.nan),
        }

    def _collect_window_metrics(self, sample_id, event_id, pred, true,
                                persistence, valid_mask):
        for method, candidate in (
                ('model', pred), ('persistence', persistence)):
            for lead in range(self.lead_count):
                candidate_field = candidate[lead, 0]
                true_field = true[lead, 0]
                mask = valid_mask[lead, 0]
                valid_candidate = candidate_field[mask]
                valid_true = true_field[mask]
                peak_error = (
                    float(np.max(valid_candidate) - np.max(valid_true))
                    if np.any(mask) else np.nan)
                percentile_errors = {}
                for percentile in (95, 99):
                    percentile_errors[f'p{percentile}_error'] = (
                        float(np.percentile(valid_candidate, percentile)
                              - np.percentile(valid_true, percentile))
                        if np.any(mask) else np.nan)
                pred_psd, pred_hf = self._radial_psd(
                    np.where(mask, candidate_field, 0.0))
                true_psd, true_hf = self._radial_psd(
                    np.where(mask, true_field, 0.0))
                finite_psd = np.isfinite(pred_psd) & np.isfinite(true_psd)
                self._psd[method]['pred_sum'][finite_psd] += (
                    pred_psd[finite_psd])
                self._psd[method]['true_sum'][finite_psd] += (
                    true_psd[finite_psd])
                self._psd[method]['count'][finite_psd] += 1
                hf_ratio = _safe_ratio(pred_hf, true_hf)
                for threshold in self.thresholds:
                    pred_event = (candidate_field >= threshold) & mask
                    true_event = (true_field >= threshold) & mask
                    hits = int(np.count_nonzero(pred_event & true_event))
                    false_alarms = int(
                        np.count_nonzero(pred_event & ~true_event & mask))
                    misses = int(
                        np.count_nonzero(~pred_event & true_event & mask))
                    correct_negatives = int(np.count_nonzero(
                        ~pred_event & ~true_event & mask))
                    row = {
                        'sample_id': str(sample_id),
                        'event_id': str(event_id),
                        'method': method,
                        'lead_index': lead + 1,
                        'lead_minutes': (lead + 1) * self.lead_minutes,
                        'threshold': threshold,
                        'csi': _safe_ratio(
                            hits, hits + false_alarms + misses),
                        'pod': _safe_ratio(hits, hits + misses),
                        'far': _safe_ratio(
                            false_alarms, hits + false_alarms),
                        'frequency_bias': _safe_ratio(
                            hits + false_alarms, hits + misses),
                        'hits': hits,
                        'false_alarms': false_alarms,
                        'misses': misses,
                        'correct_negatives': correct_negatives,
                        'area_ratio': _safe_ratio(
                            np.count_nonzero(pred_event),
                            np.count_nonzero(true_event)),
                        'energy_ratio': _safe_ratio(
                            candidate_field[pred_event].sum(),
                            true_field[true_event].sum()),
                        'peak_error': peak_error,
                        'psd_high_frequency_ratio': hf_ratio,
                        **percentile_errors,
                    }
                    hss_numerator = 2 * (
                        hits * correct_negatives
                        - false_alarms * misses)
                    hss_denominator = (
                        (hits + misses) * (misses + correct_negatives)
                        + (hits + false_alarms)
                        * (false_alarms + correct_negatives))
                    row['hss'] = _safe_ratio(
                        hss_numerator, hss_denominator)
                    if np.any(pred_event) and np.any(true_event):
                        pred_centroid = np.argwhere(pred_event).mean(axis=0)
                        true_centroid = np.argwhere(true_event).mean(axis=0)
                        row['centroid_error_km'] = float(
                            np.linalg.norm(pred_centroid - true_centroid)
                            * self.grid_spacing_km)
                    else:
                        row['centroid_error_km'] = np.nan
                    for window in self.neighborhood_windows:
                        pred_fraction, pred_valid = self._fractions(
                            pred_event, mask, window)
                        true_fraction, true_valid = self._fractions(
                            true_event, mask, window)
                        neighborhood_valid = (
                            (pred_valid > 0) & (true_valid > 0))
                        numerator = np.nansum(
                            (pred_fraction[neighborhood_valid]
                             - true_fraction[neighborhood_valid]) ** 2)
                        denominator = np.nansum(
                            pred_fraction[neighborhood_valid] ** 2
                            + true_fraction[neighborhood_valid] ** 2)
                        row[f'fss_{window}x{window}'] = (
                            1.0 - numerator / denominator
                            if denominator else np.nan)
                    row.update(self._object_scores(
                        pred_event, true_event,
                        candidate_field, true_field))
                    self._window_rows.append(row)

    def _radial_psd(self, field):
        centered = field - np.mean(field)
        power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
        rows, columns = field.shape
        row_frequency = np.fft.fftshift(np.fft.fftfreq(rows))
        column_frequency = np.fft.fftshift(np.fft.fftfreq(columns))
        radius = np.sqrt(
            row_frequency[:, None] ** 2
            + column_frequency[None, :] ** 2)
        edges = np.linspace(0.0, radius.max(), self._psd_bins + 1)
        spectrum = np.full(self._psd_bins, np.nan, dtype=np.float64)
        for index in range(self._psd_bins):
            selected = (
                (radius >= edges[index]) & (radius < edges[index + 1]))
            if np.any(selected):
                spectrum[index] = power[selected].mean()
        high_frequency_energy = power[radius >= 0.25].sum()
        return spectrum, high_frequency_energy

    def report(self):
        report = {
            'protocol': {
                'thresholds': self.thresholds,
                'value_unit': self.value_unit,
                'lead_minutes': self.lead_minutes,
                'lead_count': self.lead_count,
                'clip_range': self.clip_range,
                'undefined_policy': 'null when denominator is zero',
                'event_id_source': self.event_id_source,
                'truth_source': (
                    'RAIN_2025_S' if self.true_is_rain
                    else 'future_radar_via_frozen_zr'),
                'strict_pixel_primary': True,
                'neighborhood_windows': self.neighborhood_windows,
                'grid_spacing_km': self.grid_spacing_km,
                'object_iou_threshold': self.object_iou_threshold,
                'wet_threshold': self.wet_threshold,
                'bootstrap': {
                    'unit': 'event',
                    'repetitions': self.bootstrap_repetitions,
                    'confidence': 0.95,
                    'seed': self.bootstrap_seed,
                },
            },
            'model': self.model.report(self.thresholds),
            'persistence': self.persistence.report(self.thresholds),
            'events': {
                event_id: {
                    name: state.report(self.thresholds)
                    for name, state in methods.items()
                }
                for event_id, methods in sorted(self.events.items())
            },
        }
        if self.convert_dbz_to_rain:
            report['protocol']['frozen_zr'] = {
                'a': self.zr_a,
                'b': self.zr_b,
                'formula': 'Z=a*R^b',
                'fit_scope': 'must be training data only',
                'threshold_dbz': {
                    str(threshold): self._threshold_dbz(threshold)
                    for threshold in self.thresholds
                },
            }
        report['model']['wet_pixels'] = self.model_wet.report(
            self.thresholds)
        report['persistence']['wet_pixels'] = self.persistence_wet.report(
            self.thresholds)
        report['event_macro'] = self._event_macro(report['events'])
        report['spatial_object_summary'] = self._spatial_object_summary()
        report['bootstrap_ci'] = self._bootstrap_event_delta(report['events'])
        return report

    def _event_macro(self, events):
        result = {}
        for method in ('model', 'persistence'):
            result[method] = {}
            for threshold in self.thresholds:
                key = str(threshold)
                result[method][key] = {}
                for metric in ('csi', 'pod', 'far', 'hss', 'bias'):
                    values = np.asarray([
                        event[method]['overall']['thresholds'][key][metric]
                        for event in events.values()
                    ], dtype=np.float64)
                    values = values[np.isfinite(values)]
                    result[method][key][metric] = {
                        'count': len(values),
                        'mean': np.mean(values) if len(values) else np.nan,
                        'median': (
                            np.median(values) if len(values) else np.nan),
                        'q25': (
                            np.percentile(values, 25)
                            if len(values) else np.nan),
                        'q75': (
                            np.percentile(values, 75)
                            if len(values) else np.nan),
                    }
        return result

    def _spatial_object_summary(self):
        result = {}
        metrics = [
            'area_ratio', 'energy_ratio', 'centroid_error_km',
            'peak_error', 'p95_error', 'p99_error',
            'psd_high_frequency_ratio',
            'mean_matched_iou', 'object_hits',
            'object_false_alarms', 'object_misses',
            'object_centroid_error_km', 'object_area_error_pixels',
            'object_peak_error', 'object_pod', 'object_far',
        ] + [
            f'fss_{window}x{window}'
            for window in self.neighborhood_windows
        ]
        for method in ('model', 'persistence'):
            result[method] = {}
            for threshold in self.thresholds:
                rows = [
                    row for row in self._window_rows
                    if row['method'] == method
                    and row['threshold'] == threshold
                ]
                summary = {}
                for metric in metrics:
                    values = np.asarray(
                        [row[metric] for row in rows], dtype=np.float64)
                    finite = values[np.isfinite(values)]
                    summary[metric] = {
                        'mean': (
                            np.mean(finite) if len(finite) else np.nan),
                        'median': (
                            np.median(finite) if len(finite) else np.nan),
                        'count': len(finite),
                    }
                result[method][str(threshold)] = summary
        return result

    def _bootstrap_event_delta(self, events):
        rng = np.random.default_rng(self.bootstrap_seed)
        result = {}
        for threshold in self.thresholds:
            key = str(threshold)
            result[key] = {}
            for metric in ('csi', 'pod', 'far', 'hss', 'bias'):
                differences = []
                for event in events.values():
                    model_value = event['model']['overall'][
                        'thresholds'][key][metric]
                    baseline_value = event['persistence']['overall'][
                        'thresholds'][key][metric]
                    if np.isfinite(model_value) and np.isfinite(
                            baseline_value):
                        differences.append(model_value - baseline_value)
                differences = np.asarray(differences, dtype=np.float64)
                if len(differences) == 0:
                    result[key][metric] = {
                        'event_count': 0, 'mean_delta': np.nan,
                        'ci_lower': np.nan, 'ci_upper': np.nan}
                    continue
                indices = rng.integers(
                    0, len(differences),
                    size=(self.bootstrap_repetitions, len(differences)))
                samples = differences[indices].mean(axis=1)
                result[key][metric] = {
                    'event_count': len(differences),
                    'mean_delta': differences.mean(),
                    'ci_lower': np.percentile(samples, 2.5),
                    'ci_upper': np.percentile(samples, 97.5),
                }
        return result

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.report()
        with (output_dir / 'metrics.json').open('w', encoding='utf-8') as stream:
            json.dump(_json_value(report), stream, ensure_ascii=False, indent=2)
        with (output_dir / 'summary.json').open(
                'w', encoding='utf-8') as stream:
            json.dump(_json_value(report), stream, ensure_ascii=False, indent=2)
        self._save_lead_csv(output_dir / 'lead_time_metrics.csv', report)
        self._save_lead_csv(output_dir / 'per_lead_metrics.csv', report)
        self._save_event_csv(output_dir / 'event_metrics.csv', report)
        self._save_event_csv(output_dir / 'per_event_metrics.csv', report)
        self._save_window_csv(output_dir / 'per_window_metrics.csv')
        self._save_object_csv(output_dir / 'per_object_metrics.csv')
        self._save_confusion_csv(
            output_dir / 'confusion_counts.csv', report)
        with (output_dir / 'bootstrap_ci.json').open(
                'w', encoding='utf-8') as stream:
            json.dump(
                _json_value(report['bootstrap_ci']), stream,
                ensure_ascii=False, indent=2)
        self._plot_curves(output_dir / 'lead_time_curves.png', report)
        self._plot_psd(output_dir / 'psd_comparison.png')
        self._save_cases(output_dir)
        return report

    def _plot_psd(self, path):
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        frequencies = np.arange(self._psd_bins)
        for method, style in (('model', '-'), ('persistence', '--')):
            state = self._psd[method]
            pred = _safe_ratio(state['pred_sum'], state['count'])
            true = _safe_ratio(state['true_sum'], state['count'])
            axis.plot(
                frequencies, pred, style, label=f'{method} prediction')
            if method == 'model':
                axis.plot(frequencies, true, ':', label='truth')
        plotted_values = [
            line.get_ydata() for line in axis.get_lines()
        ]
        if any(np.any(np.asarray(values) > 0) for values in plotted_values):
            axis.set_yscale('log')
        axis.set_xlabel('Radial spatial-frequency bin')
        axis.set_ylabel('Mean power spectral density')
        axis.set_title('Spatial PSD diagnostic')
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def _save_window_csv(self, path):
        if not self._window_rows:
            return
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(self._window_rows[0]))
            writer.writeheader()
            writer.writerows([
                _json_value(row) for row in self._window_rows])

    def _save_object_csv(self, path):
        fields = [
            'sample_id', 'event_id', 'method', 'lead_index',
            'lead_minutes', 'threshold', 'object_hits',
            'object_false_alarms', 'object_misses', 'mean_matched_iou',
            'object_centroid_error_km', 'object_area_error_pixels',
            'object_peak_error', 'object_pod', 'object_far']
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self._window_rows:
                writer.writerow({field: _json_value(row[field])
                                 for field in fields})

    def _save_confusion_csv(self, path, report):
        fields = [
            'method', 'lead_index', 'lead_minutes', 'threshold',
            'hits', 'false_alarms', 'misses', 'correct_negatives']
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method in ('model', 'persistence'):
                lead = report[method]['lead_time']['thresholds']
                for threshold in self.thresholds:
                    values = lead[str(threshold)]
                    for index in range(self.lead_count):
                        writer.writerow({
                            'method': method,
                            'lead_index': index + 1,
                            'lead_minutes': (
                                index + 1) * self.lead_minutes,
                            'threshold': threshold,
                            **{
                                field: values[field][index]
                                for field in fields[-4:]
                            },
                        })

    def _save_lead_csv(self, path, report):
        fields = ['method', 'lead_index', 'lead_minutes', 'threshold',
                  'mae', 'rmse', 'mean_error', 'intensity_ratio',
                  'csi', 'pod', 'far', 'hss', 'bias',
                  'hits', 'false_alarms', 'misses', 'correct_negatives']
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method in ('model', 'persistence'):
                lead = report[method]['lead_time']
                for threshold in self.thresholds:
                    categorical = lead['thresholds'][str(threshold)]
                    for index in range(self.lead_count):
                        writer.writerow({
                            'method': method,
                            'lead_index': index + 1,
                            'lead_minutes': (index + 1) * self.lead_minutes,
                            'threshold': threshold,
                            'mae': lead['mae'][index],
                            'rmse': lead['rmse'][index],
                            'mean_error': lead['mean_error'][index],
                            'intensity_ratio': (
                                lead['intensity_ratio'][index]),
                            **{
                                key: categorical[key][index]
                                for key in (
                                    'csi', 'pod', 'far', 'hss', 'bias',
                                    'hits', 'false_alarms', 'misses',
                                    'correct_negatives')
                            },
                        })

    def _save_event_csv(self, path, report):
        fields = ['event_id', 'method', 'sample_count', 'threshold',
                  'mae', 'rmse', 'mean_error', 'intensity_ratio',
                  'csi', 'pod', 'far', 'hss', 'bias',
                  'hits', 'false_alarms', 'misses', 'correct_negatives']
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for event_id, methods in report['events'].items():
                for method, values in methods.items():
                    overall = values['overall']
                    for threshold in self.thresholds:
                        categorical = overall['thresholds'][str(threshold)]
                        writer.writerow({
                            'event_id': event_id,
                            'method': method,
                            'sample_count': values['sample_count'],
                            'threshold': threshold,
                            'mae': overall['mae'],
                            'rmse': overall['rmse'],
                            'mean_error': overall['mean_error'],
                            'intensity_ratio': overall['intensity_ratio'],
                            **categorical,
                        })

    def _plot_curves(self, path, report):
        leads = np.arange(1, self.lead_count + 1) * self.lead_minutes
        figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
        for method, style in (('model', '-'), ('persistence', '--')):
            axes[0, 0].plot(
                leads, report[method]['lead_time']['mae'],
                style, label=method)
            axes[0, 1].plot(
                leads, report[method]['lead_time']['rmse'],
                style, label=method)
        for axis, metric_name in zip(
                axes.flat[2:], ('csi', 'pod', 'far', 'bias')):
            for method, style in (('model', '-'), ('persistence', '--')):
                for threshold in self.thresholds:
                    values = report[method]['lead_time']['thresholds'][
                        str(threshold)][metric_name]
                    axis.plot(
                        leads, values, style,
                        label=f'{method} {threshold:g} {self.value_unit}')
        titles = ('MAE', 'RMSE', 'CSI', 'POD', 'FAR', 'Bias')
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title)
            axis.set_xlabel('Lead time (min)')
            axis.grid(alpha=0.25)
        axes[0, 0].set_ylabel(self.value_unit)
        axes[0, 1].set_ylabel(self.value_unit)
        axes[0, 0].legend()
        axes[0, 1].legend()
        for axis in axes.flat[2:]:
            axis.legend(fontsize=7, ncol=2)
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def _save_cases(self, output_dir):
        cases_dir = output_dir / 'cases'
        cases_dir.mkdir(exist_ok=True)
        metadata = {'success': [], 'failure': []}
        groups = {
            'success': sorted(self._best_cases, reverse=True),
            'failure': sorted(self._worst_cases, reverse=True),
        }
        for kind, cases in groups.items():
            for rank, (_, _, case) in enumerate(cases, start=1):
                filename = (
                    f'{kind}_{rank:02d}_sample_{case["sample_id"]}.png')
                self._plot_case(cases_dir / filename, case)
                metadata[kind].append({
                    key: value for key, value in case.items()
                    if key not in ('inputs', 'pred', 'true', 'persistence')
                } | {'figure': filename})
        with (cases_dir / 'cases.json').open('w', encoding='utf-8') as stream:
            json.dump(_json_value(metadata), stream, ensure_ascii=False, indent=2)

    def _plot_case(self, path, case):
        leads = sorted(set(
            [0, self.lead_count // 4, self.lead_count // 2 - 1,
             3 * self.lead_count // 4 - 1, self.lead_count - 1]))
        cmap = ListedColormap(RADAR_COLORS)
        norm = BoundaryNorm(RADAR_BOUNDS, cmap.N, clip=True)
        figure, axes = plt.subplots(
            3, len(leads), figsize=(2.4 * len(leads), 7.2),
            constrained_layout=True)
        rows = (
            ('Truth', case['true']),
            ('Model', case['pred']),
            ('Persistence', case['persistence']),
        )
        for row_index, (label, values) in enumerate(rows):
            for column_index, lead in enumerate(leads):
                axis = axes[row_index, column_index]
                image = axis.imshow(values[lead, 0], cmap=cmap, norm=norm)
                axis.set_xticks([])
                axis.set_yticks([])
                if column_index == 0:
                    axis.set_ylabel(label)
                axis.set_title(f'+{(lead + 1) * self.lead_minutes} min')
        figure.colorbar(
            image, ax=axes, orientation='horizontal', shrink=0.65,
            label=self.value_unit)
        figure.suptitle(
            f'Event {case["event_id"]} · sample {case["sample_id"]} · '
            f'CSI@{case["case_threshold"]:g} improvement '
            f'{case["improvement_csi"]:+.3f} · RMSE improvement '
            f'{case["improvement_rmse"]:+.2f} {self.value_unit}')
        figure.savefig(path, dpi=180)
        plt.close(figure)
