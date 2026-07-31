"""Event-balanced, zero-aware BTH Radar--Rain calibration."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image


STAMP = '%Y-%m-%d-%H-%M-%S'
RAIN_THRESHOLDS = (0.1, 2.5, 8.0, 16.0, 32.0)
STRATA = (-np.inf, 0.1, 2.5, 8.0, 16.0, 32.0, np.inf)


def index_png(root, prefix):
    candidates = sorted(Path(root).glob(f'{prefix}_*_S'))
    folder = candidates[0] if candidates else Path(root)
    grouped = defaultdict(list)
    for path in folder.rglob('*.png'):
        try:
            grouped[datetime.strptime(path.stem, STAMP)].append(path)
        except ValueError:
            continue
    frames = {stamp: sorted(paths, key=str)[0]
              for stamp, paths in grouped.items()}
    duplicates = {
        stamp.isoformat(): [str(path) for path in paths]
        for stamp, paths in grouped.items() if len(paths) > 1
    }
    return frames, duplicates


def decode(path, scale):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'), dtype=np.float32)
    return (255.0 - pixels) * (scale / 255.0)


def load_events(path, split):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    return [{
        'event_id': event['event_id'],
        'start': datetime.fromisoformat(event['start']),
        'end': datetime.fromisoformat(event['end']),
    } for event in manifest['events'] if event['split'] == split]


def event_timestamps(radar, events, step_frames=30):
    selected = []
    ordered = sorted(radar)
    for event in events:
        stamps = [stamp for stamp in ordered
                  if event['start'] <= stamp <= event['end']]
        selected.extend((event['event_id'], stamp)
                        for stamp in stamps[::step_frames])
    return selected


def shifted_pair(radar_array, rain_array, row_shift, col_shift):
    rows, cols = radar_array.shape
    r0 = max(0, row_shift)
    r1 = min(rows, rows + row_shift)
    c0 = max(0, col_shift)
    c1 = min(cols, cols + col_shift)
    return (radar_array[r0:r1, c0:c1],
            rain_array[r0-row_shift:r1-row_shift,
                       c0-col_shift:c1-col_shift])


def alignment_counts(radar, rain, selected, lag_minutes,
                     row_shift=0, col_shift=0):
    lag = timedelta(minutes=lag_minutes)
    counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    log_stats = defaultdict(lambda: np.zeros(6, dtype=np.float64))
    for event_id, stamp in selected:
        rain_stamp = stamp + lag
        if rain_stamp not in rain:
            continue
        dbz, rate = shifted_pair(
            decode(radar[stamp], 50.0), decode(rain[rain_stamp], 35.0),
            row_shift, col_shift)
        pred_event, obs_event = dbz >= 20.0, rate >= 0.1
        counts[event_id] += (
            np.sum(pred_event & obs_event),
            np.sum(pred_event & ~obs_event),
            np.sum(~pred_event & obs_event),
        )
        positive = obs_event & (dbz > 0)
        if positive.any():
            x = np.log(10.0) * dbz[positive] / 10.0
            y = np.log(rate[positive])
            log_stats[event_id] += (
                x.size, x.sum(), y.sum(), np.square(x).sum(),
                np.square(y).sum(), (x * y).sum())
    return counts, log_stats


def macro_csi(counts, event_ids):
    values = []
    for event_id in event_ids:
        hit, false_alarm, miss = counts[event_id]
        denominator = hit + false_alarm + miss
        if denominator:
            values.append(hit / denominator)
    return float(np.mean(values)) if values else float('nan')


def lag_cross_validation(radar, rain, selected, event_ids,
                         candidates, folds=5):
    by_lag = {}
    for lag in candidates:
        counts, log_stats = alignment_counts(radar, rain, selected, lag)
        by_lag[lag] = {'counts': counts, 'log_stats': log_stats}
    fold_results = []
    for fold in range(folds):
        held_out = {event for index, event in enumerate(event_ids)
                    if index % folds == fold}
        training = [event for event in event_ids if event not in held_out]
        scores = {lag: macro_csi(item['counts'], training)
                  for lag, item in by_lag.items()}
        best = max(scores, key=lambda lag: (scores[lag], -abs(lag)))
        fold_results.append({
            'fold': fold, 'selected_lag_minutes': best,
            'train_macro_csi': scores[best],
            'held_out_macro_csi': macro_csi(
                by_lag[best]['counts'], held_out),
            'held_out_event_ids': sorted(held_out),
        })
    global_scores = {
        lag: macro_csi(item['counts'], event_ids)
        for lag, item in by_lag.items()}
    frozen = max(global_scores, key=lambda lag: (global_scores[lag], -abs(lag)))
    return frozen, global_scores, fold_results


def choose_spatial_shift(radar, rain, selected, event_ids, lag_minutes,
                         radius=2):
    scores = {}
    for row_shift in range(-radius, radius + 1):
        for col_shift in range(-radius, radius + 1):
            counts, _ = alignment_counts(
                radar, rain, selected, lag_minutes,
                row_shift=row_shift, col_shift=col_shift)
            scores[(row_shift, col_shift)] = macro_csi(counts, event_ids)
    best = max(scores, key=lambda shift: (
        scores[shift], -abs(shift[0])-abs(shift[1])))
    return best, scores


def collect_calibration_pairs(radar, rain, selected, lag_minutes,
                              row_shift, col_shift, per_stratum=24,
                              seed=42):
    rng = np.random.default_rng(seed)
    pieces = []
    lag = timedelta(minutes=lag_minutes)
    for event_id, stamp in selected:
        rain_stamp = stamp + lag
        if rain_stamp not in rain:
            continue
        dbz, rate = shifted_pair(
            decode(radar[stamp], 50.0), decode(rain[rain_stamp], 35.0),
            row_shift, col_shift)
        dbz, rate = dbz.ravel(), rate.ravel()
        for stratum in range(len(STRATA) - 1):
            mask = ((rate >= STRATA[stratum])
                    & (rate < STRATA[stratum + 1]))
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue
            if indices.size > per_stratum:
                indices = rng.choice(indices, per_stratum, replace=False)
            pieces.append((
                dbz[indices], rate[indices],
                np.full(indices.size, event_id, dtype=object),
                np.full(indices.size, stratum, dtype=np.int8)))
    if not pieces:
        raise RuntimeError('No calibration pairs were collected')
    return tuple(np.concatenate([piece[index] for piece in pieces])
                 for index in range(4))


def balanced_weights(event_ids, strata):
    keys = list(zip(event_ids.tolist(), strata.tolist()))
    counts = defaultdict(int)
    for key in keys:
        counts[key] += 1
    weights = np.asarray([1.0 / counts[key] for key in keys])
    weights /= weights.sum()
    return weights


def rain_from_dbz(dbz, a, b, z0_dbz):
    predicted = np.zeros_like(dbz, dtype=np.float64)
    wet = dbz >= z0_dbz
    predicted[wet] = np.power(
        np.power(10.0, dbz[wet] / 10.0) / a, 1.0 / b)
    return predicted


def fit_zero_aware(dbz, rate, event_ids, strata):
    weights = balanced_weights(event_ids, strata)
    candidates = []
    for z0 in np.arange(0.0, 35.01, 0.5):
        fit = (rate >= 0.1) & (dbz >= z0)
        if fit.sum() < 100:
            continue
        x = np.column_stack((
            np.ones(fit.sum()), np.log(rate[fit])))
        y = np.log(10.0) * dbz[fit] / 10.0
        sqrt_w = np.sqrt(weights[fit])
        coefficients, _, _, _ = np.linalg.lstsq(
            x * sqrt_w[:, None], y * sqrt_w, rcond=None)
        log_a, b = coefficients
        a = float(np.exp(log_a))
        if not (10.0 <= a <= 1000.0 and 0.5 <= b <= 3.0):
            continue
        predicted = rain_from_dbz(dbz, a, b, z0_dbz=z0)
        residual = np.log1p(predicted) - np.log1p(rate)
        absolute = np.abs(residual)
        huber = np.where(absolute <= 0.5, 0.5 * residual ** 2,
                         0.5 * (absolute - 0.25))
        continuous = float(np.sum(weights * huber))
        categorical = np.mean([
            float(np.sum(weights * (
                (predicted >= threshold) != (rate >= threshold))))
            for threshold in RAIN_THRESHOLDS])
        objective = 0.5 * continuous + 0.5 * categorical
        candidates.append({
            'a': a, 'b': float(b), 'z0_dbz': float(z0),
            'objective': objective,
            'continuous_loss': continuous,
            'categorical_error': categorical,
        })
    return min(candidates, key=lambda item: item['objective']), candidates


def fit_categorical_power_law(dbz, rate, event_ids, strata):
    """Fit the power law to Train-optimal categorical dBZ thresholds."""
    weights = balanced_weights(event_ids, strata)
    dbz_candidates = np.arange(0.0, 50.01, 0.25)
    optimal = {}
    for rain_threshold in RAIN_THRESHOLDS:
        observed = rate >= rain_threshold
        best = None
        for dbz_threshold in dbz_candidates:
            predicted = dbz >= dbz_threshold
            hit = np.sum(weights * (predicted & observed))
            false_alarm = np.sum(weights * (predicted & ~observed))
            miss = np.sum(weights * (~predicted & observed))
            csi = hit / (hit + false_alarm + miss)
            candidate = (csi, -abs(dbz_threshold), dbz_threshold)
            if best is None or candidate > best:
                best = candidate
        optimal[rain_threshold] = {
            'dbz_threshold': float(best[2]), 'balanced_csi': float(best[0])}
    x = 10.0 * np.log10(np.asarray(RAIN_THRESHOLDS))
    y = np.asarray([optimal[value]['dbz_threshold']
                    for value in RAIN_THRESHOLDS])
    design = np.column_stack((np.ones(x.size), x))
    intercept, b = np.linalg.lstsq(design, y, rcond=None)[0]
    relation = {
        'a': float(10.0 ** (intercept / 10.0)),
        'b': float(b),
        'z0_dbz': optimal[0.1]['dbz_threshold'],
    }
    return relation, {str(key): value for key, value in optimal.items()}


def evaluate_relation(radar, rain, selected, relation,
                      lag_minutes, row_shift, col_shift):
    counts = {threshold: np.zeros(3, dtype=np.int64)
              for threshold in RAIN_THRESHOLDS}
    abs_sum = sq_sum = 0.0
    size = 0
    lag = timedelta(minutes=lag_minutes)
    for _, stamp in selected:
        rain_stamp = stamp + lag
        if rain_stamp not in rain:
            continue
        dbz, observed = shifted_pair(
            decode(radar[stamp], 50.0), decode(rain[rain_stamp], 35.0),
            row_shift, col_shift)
        predicted = rain_from_dbz(dbz, **relation)
        error = predicted - observed
        abs_sum += np.abs(error).sum()
        sq_sum += np.square(error).sum()
        size += error.size
        for threshold in RAIN_THRESHOLDS:
            pe, oe = predicted >= threshold, observed >= threshold
            counts[threshold] += (
                np.sum(pe & oe), np.sum(pe & ~oe), np.sum(~pe & oe))
    report = {'mae': abs_sum / size, 'rmse': (sq_sum / size) ** 0.5,
              'thresholds': {}}
    for threshold, (hit, false_alarm, miss) in counts.items():
        report['thresholds'][str(threshold)] = {
            'pod': hit / (hit + miss) if hit + miss else None,
            'csi': hit / (hit + false_alarm + miss)
            if hit + false_alarm + miss else None,
            'far': false_alarm / (hit + false_alarm)
            if hit + false_alarm else None,
            'hits': int(hit), 'false_alarms': int(false_alarm),
            'misses': int(miss),
        }
    return report


def interpolation_audit(rain):
    errors = []
    anchors = sorted(stamp for stamp in rain
                     if stamp.minute == 0 and stamp + timedelta(hours=1) in rain)
    for start in anchors[::max(1, len(anchors) // 200)]:
        first = decode(rain[start], 35.0)
        last = decode(rain[start + timedelta(hours=1)], 35.0)
        for index in range(1, 10):
            stamp = start + timedelta(minutes=6 * index)
            if stamp not in rain:
                continue
            expected = first * (1-index/10) + last * (index/10)
            errors.append(np.mean(np.abs(decode(rain[stamp], 35.0)
                                         - expected)))
    return {
        'tested_intermediate_frames': len(errors),
        'linear_interpolation_mae_mm_h': float(np.mean(errors)),
        'linear_interpolation_max_mae_mm_h': float(np.max(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-2025', required=True)
    parser.add_argument('--manifest-2025', required=True)
    parser.add_argument('--data-2023', required=True)
    parser.add_argument('--manifest-2023', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    radar25, radar25_duplicates = index_png(args.data_2025, 'RADAR')
    rain25, rain25_duplicates = index_png(args.data_2025, 'RAIN')
    radar23, radar23_duplicates = index_png(args.data_2023, 'RADAR')
    rain23, rain23_duplicates = index_png(args.data_2023, 'RAIN')
    train_events = load_events(args.manifest_2025, 'train')
    val_events = load_events(args.manifest_2025, 'val')
    event_ids = [event['event_id'] for event in train_events]

    scan_selected = event_timestamps(radar25, train_events, step_frames=30)
    lag, lag_scores, folds = lag_cross_validation(
        radar25, rain25, scan_selected, event_ids,
        candidates=range(-60, 61, 6))
    shift, shift_scores = choose_spatial_shift(
        radar25, rain25, scan_selected, event_ids, lag)

    calibration_selected = event_timestamps(
        radar25, train_events, step_frames=5)
    dbz, rate, pair_events, strata = collect_calibration_pairs(
        radar25, rain25, calibration_selected, lag, *shift)
    frozen, candidates = fit_zero_aware(dbz, rate, pair_events, strata)
    categorical, categorical_thresholds = fit_categorical_power_law(
        dbz, rate, pair_events, strata)

    relations = {
        'zero_aware_v2': {key: frozen[key]
                          for key in ('a', 'b', 'z0_dbz')},
        'categorical_v3': categorical,
        'positive_only_v1': {
            'a': 42.11573040427291, 'b': 0.9920686176880671,
            'z0_dbz': 0.0},
        'marshall_palmer': {'a': 200.0, 'b': 1.6, 'z0_dbz': 0.0},
    }
    val_selected = event_timestamps(radar25, val_events, step_frames=5)
    validation = {
        name: evaluate_relation(
            radar25, rain25, val_selected, relation, lag, *shift)
        for name, relation in relations.items()
    }

    # 2023 is an external-year diagnostic only; it never selects parameters.
    events23 = load_events(args.manifest_2023, 'train')
    selected23 = event_timestamps(radar23, events23, step_frames=1)
    external_2023 = {
        name: evaluate_relation(
            radar23, rain23, selected23, relation, lag, *shift)
        for name, relation in relations.items()
    }
    report = {
        'schema_version': 2,
        'selection_scope': '2025_train_events_only',
        'test_used_for_selection': False,
        'data_audit': {
            '2025': {
                'radar_unique_frames': len(radar25),
                'rain_unique_frames': len(rain25),
                'radar_duplicate_timestamps': radar25_duplicates,
                'rain_duplicate_timestamps': rain25_duplicates,
                'rain_interpolation': interpolation_audit(rain25),
            },
            '2023': {
                'radar_unique_frames': len(radar23),
                'rain_unique_frames': len(rain23),
                'radar_duplicate_timestamps': radar23_duplicates,
                'rain_duplicate_timestamps': rain23_duplicates,
                'role': 'external_year_diagnostic_only',
            },
        },
        'alignment': {
            'lag_minutes': lag,
            'row_shift': shift[0], 'col_shift': shift[1],
            'lag_macro_csi': {str(key): value
                              for key, value in lag_scores.items()},
            'lag_cross_validation': folds,
            'spatial_macro_csi': {
                f'{row},{col}': score
                for (row, col), score in shift_scores.items()},
        },
        'calibration': {
            'relation': 'R=0 below z0; otherwise (10^(dBZ/10)/a)^(1/b)',
            'frozen': relations['zero_aware_v2'],
            'sample_count': int(dbz.size),
            'objective': frozen['objective'],
            'continuous_loss': frozen['continuous_loss'],
            'categorical_error': frozen['categorical_error'],
            'stratum_counts': {
                str(index): int(np.sum(strata == index))
                for index in range(len(STRATA)-1)},
            'candidate_count': len(candidates),
            'categorical_v3': categorical,
            'train_optimal_thresholds': categorical_thresholds,
        },
        'validation_2025': validation,
        'external_2023': external_2023,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'alignment': report['alignment'],
        'frozen': report['calibration']['frozen'],
        'validation_2025': validation,
        'external_2023': external_2023,
    }, indent=2))


if __name__ == '__main__':
    main()
