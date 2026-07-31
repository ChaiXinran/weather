"""Leakage-safe event manifests and local Z--R calibration for BTH Radar."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S'
FRAME_INTERVAL = timedelta(minutes=6)


def _parse_timestamp(path):
    return datetime.strptime(Path(path).stem, TIMESTAMP_FORMAT)


def _resolve_radar_root(data_root):
    root = Path(data_root)
    for candidate in tuple(sorted(root.glob('RADAR_*_S'))) + (root,):
        if candidate.is_dir() and next(candidate.rglob('*.png'), None):
            return candidate
    raise FileNotFoundError(f'No Radar PNGs found under {root}')


def read_dbz(path):
    """Decode an inverse-grey Radar PNG to dBZ."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'), dtype=np.float64)
    return (255.0 - pixels) * (50.0 / 255.0)


def discover_radar_frames(data_root):
    root = _resolve_radar_root(data_root)
    frames = {}
    for path in root.rglob('*.png'):
        timestamp = _parse_timestamp(path)
        if timestamp in frames:
            raise RuntimeError(f'Duplicate Radar timestamp: {timestamp}')
        frames[timestamp] = path
    return frames


def identify_events(frames, dbz_threshold=10.0, wet_fraction=0.001,
                    max_dry_gap_hours=6.0, padding_minutes=30,
                    scan_workers=8):
    """Identify independent precipitation processes from domain Radar activity.

    Active frames separated by at most ``max_dry_gap_hours`` belong to one
    synoptic process. Padding is applied before samples are generated, and
    overlapping padded processes are merged.
    """
    ordered = sorted(frames.items())

    def is_active(item):
        timestamp, path = item
        dbz = read_dbz(path)
        if np.mean(dbz >= dbz_threshold) >= wet_fraction:
            return timestamp
        return None

    with ThreadPoolExecutor(max_workers=scan_workers) as executor:
        active = [item for item in executor.map(is_active, ordered)
                  if item is not None]
    if not active:
        raise RuntimeError('No active Radar frames satisfy the event threshold')

    gap = timedelta(hours=max_dry_gap_hours)
    raw = [[active[0], active[0]]]
    for timestamp in active[1:]:
        if timestamp - raw[-1][1] <= gap:
            raw[-1][1] = timestamp
        else:
            raw.append([timestamp, timestamp])

    padding = timedelta(minutes=padding_minutes)
    merged = []
    for start, end in raw:
        candidate = [start - padding, end + padding]
        if merged and candidate[0] <= merged[-1][1] + FRAME_INTERVAL:
            merged[-1][1] = max(merged[-1][1], candidate[1])
        else:
            merged.append(candidate)
    return merged


def assign_splits(events, train_end='2025-07-31T23:59:59',
                  val_fraction=0.5):
    """Keep May--July events in train and split later whole events in time."""
    boundary = datetime.fromisoformat(train_end)
    train = [event for event in events if event[1] <= boundary]
    # A process crossing the nominal date boundary is held out in its entirety.
    held_out = [event for event in events if event[1] > boundary]
    cut = int(round(len(held_out) * val_fraction))
    if len(held_out) > 1:
        cut = min(max(cut, 1), len(held_out) - 1)
    return ([(event, 'train') for event in train] +
            [(event, 'val' if index < cut else 'test')
             for index, event in enumerate(held_out)])


def build_manifest(data_root, output_path, pre_seq_length=10,
                   aft_seq_length=20, stride=1,
                   start_date='2025-05-01', end_date='2025-08-31',
                   train_end='2025-07-31T23:59:59',
                   val_fraction=0.5,
                   **event_kwargs):
    """Write an event-first manifest; no sample can cross an event boundary."""
    frames = discover_radar_frames(data_root)
    first = datetime.fromisoformat(start_date)
    last = datetime.fromisoformat(end_date) + timedelta(days=1)
    frames = {timestamp: path for timestamp, path in frames.items()
              if first <= timestamp < last}
    events = assign_splits(
        identify_events(frames, **event_kwargs),
        train_end=train_end, val_fraction=val_fraction)
    year = first.year
    records, samples = [], []
    total = pre_seq_length + aft_seq_length
    for number, ((start, end), split) in enumerate(events, 1):
        event_id = f'bth-{year}-{number:03d}'
        timestamps = [item for item in sorted(frames) if start <= item <= end]
        timestamp_set = set(timestamps)
        event_samples = 0
        for timestamp in timestamps[::stride]:
            sequence = [timestamp + i * FRAME_INTERVAL for i in range(total)]
            if sequence[-1] <= end and all(item in timestamp_set
                                           for item in sequence):
                samples.append({
                    'sample_id': f'{event_id}-{event_samples:06d}',
                    'event_id': event_id,
                    'split': split,
                    'input_start': sequence[0].isoformat(),
                    'target_start': sequence[pre_seq_length].isoformat(),
                    'target_end': sequence[-1].isoformat(),
                })
                event_samples += 1
        records.append({
            'event_id': event_id, 'split': split,
            'start': start.isoformat(), 'end': end.isoformat(),
            'sample_count': event_samples,
        })
    document = {
        'schema_version': 1,
        'radar_root': str(_resolve_radar_root(data_root).resolve()),
        'event_definition': {
            'source': 'radar_domain_activity',
            'independence_rule': (
                'active periods separated by more than max_dry_gap_hours'),
            **event_kwargs,
        },
        'sequence': {'input_frames': pre_seq_length,
                     'target_frames': aft_seq_length,
                     'interval_minutes': 6, 'stride': stride},
        'data_range': {'start_date': start_date, 'end_date': end_date},
        'split_rule': {'train_end': train_end, 'val_fraction': val_fraction},
        'events': records,
        'samples': samples,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2), encoding='utf-8')
    return document


def load_station_pairs(path):
    """Load station-hour pairs with timestamp, row, col and rain_mm columns."""
    with open(path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    required = {'timestamp', 'row', 'col', 'rain_mm'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Station CSV requires columns {sorted(required)}')
    return rows


def _resolve_rain_root(data_root):
    root = Path(data_root)
    for candidate in tuple(sorted(root.glob('RAIN_*_S'))) + (root,):
        if candidate.is_dir() and next(candidate.rglob('*.png'), None):
            return candidate
    raise FileNotFoundError(f'No Rain PNGs found under {root}')


def discover_rain_frames(data_root):
    frames = {}
    for path in _resolve_rain_root(data_root).rglob('*.png'):
        timestamp = _parse_timestamp(path)
        if timestamp in frames:
            raise RuntimeError(f'Duplicate Rain timestamp: {timestamp}')
        frames[timestamp] = path
    return frames


def read_rain_rate(path):
    """Decode inverse-grey Rain PNG to the documented 0--35 mm/h range."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'), dtype=np.float64)
    return (255.0 - pixels) * (35.0 / 255.0)


def fit_local_zr_from_rain(manifest_path, data_root, output_path,
                           min_rain_rate=0.1, min_dbz=0.0,
                           max_pairs=2_000_000, seed=42):
    """Robustly fit log(Z)=log(a)+b*log(R) on unique train grid pairs.

    RAIN PNGs are treated as a gridded gauge-derived calibration product, not
    as independent station observations. Dry pairs are retained for reporting
    but cannot enter the logarithmic power-law fit.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    intervals = [(datetime.fromisoformat(item['start']),
                  datetime.fromisoformat(item['end']))
                 for item in manifest['events'] if item['split'] == 'train']
    radar = discover_radar_frames(data_root)
    rain = discover_rain_frames(data_root)
    timestamps = sorted(
        timestamp for timestamp in radar.keys() & rain.keys()
        if any(start <= timestamp <= end for start, end in intervals))
    if not timestamps:
        raise RuntimeError('No timestamp-matched Radar--Rain frames in train')

    rng = np.random.default_rng(seed)
    log_r_parts, log_z_parts = [], []
    positive_count = dry_count = 0
    for timestamp in timestamps:
        dbz = read_dbz(radar[timestamp]).ravel()
        rate = read_rain_rate(rain[timestamp]).ravel()
        valid = np.isfinite(dbz) & np.isfinite(rate)
        positive = valid & (rate >= min_rain_rate) & (dbz >= min_dbz)
        positive_count += int(positive.sum())
        dry_count += int((valid & (rate < min_rain_rate)).sum())
        if positive.any():
            log_r_parts.append(np.log(rate[positive]))
            log_z_parts.append(np.log(10.0) * dbz[positive] / 10.0)

    log_r = np.concatenate(log_r_parts)
    log_z = np.concatenate(log_z_parts)
    if log_r.size > max_pairs:
        chosen = rng.choice(log_r.size, max_pairs, replace=False)
        log_r, log_z = log_r[chosen], log_z[chosen]
    design = np.column_stack((np.ones(log_r.size), log_r))
    coefficients, _, _, _ = np.linalg.lstsq(design, log_z, rcond=None)
    intercept, b = coefficients
    residual = log_z - design @ coefficients
    # One deterministic robust refit after excluding extreme residuals.
    center = np.median(residual)
    scale = 1.4826 * np.median(np.abs(residual - center))
    keep = np.abs(residual - center) <= max(3.0 * scale, 1e-12)
    coefficients, _, _, _ = np.linalg.lstsq(
        design[keep], log_z[keep], rcond=None)
    intercept, b = coefficients
    a = float(np.exp(intercept))
    fitted = design[keep] @ coefficients
    result = {
        'schema_version': 1, 'frozen': True,
        'fit_scope': 'unique_frames_in_train_events_only',
        'calibration_target': 'RAIN_2025_S_gridded_product',
        'relation': 'Z=a*R^b', 'a': a, 'b': float(b),
        'positive_pair_count': positive_count,
        'dry_pair_count_reported_not_log_fitted': dry_count,
        'sampled_positive_pair_count': int(log_r.size),
        'robust_inlier_pair_count': int(keep.sum()),
        'min_rain_rate_mm_h': min_rain_rate, 'min_dbz': min_dbz,
        'log_rmse': float(np.sqrt(np.mean((fitted-log_z[keep])**2))),
        'manifest': str(Path(manifest_path).resolve()),
        'radar_root': str(_resolve_radar_root(data_root).resolve()),
        'rain_root': str(_resolve_rain_root(data_root).resolve()),
        'warning': (
            'RAIN PNG is a gridded calibration product; it is not independent '
            'station validation data. Values are quantized and capped at 35 mm/h.'),
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def fit_local_zr(manifest_path, station_csv, output_path,
                 a_grid=None, b_grid=None):
    """Fit Z=aR^b using only station-hours belonging to train events."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    train_events = {
        item['event_id']: (datetime.fromisoformat(item['start']),
                           datetime.fromisoformat(item['end']))
        for item in manifest['events'] if item['split'] == 'train'}
    if not train_events:
        raise ValueError('Manifest contains no train events')
    frames = discover_radar_frames(manifest['radar_root'])
    pairs = []
    for row in load_station_pairs(station_csv):
        hour = datetime.fromisoformat(row['timestamp'])
        if not any(start <= hour <= end for start, end in train_events.values()):
            continue
        timestamps = [hour + index * FRAME_INTERVAL for index in range(10)]
        if not all(item in frames for item in timestamps):
            continue
        rr, cc = int(row['row']), int(row['col'])
        z = np.array([10.0 ** (read_dbz(frames[item])[rr, cc] / 10.0)
                      for item in timestamps])
        pairs.append((z, float(row['rain_mm'])))
    if not pairs:
        raise RuntimeError('No complete train-only station--Radar hour pairs')

    a_grid = np.asarray(a_grid if a_grid is not None
                        else np.geomspace(20.0, 1000.0, 160))
    b_grid = np.asarray(b_grid if b_grid is not None
                        else np.linspace(0.8, 3.0, 177))
    observed = np.asarray([item[1] for item in pairs])
    best = None
    for b in b_grid:
        sums = np.asarray([np.sum(z ** (1.0 / b)) * 0.1
                           for z, _ in pairs])
        predicted = sums[:, None] / (a_grid[None, :] ** (1.0 / b))
        mae = np.mean(np.abs(predicted - observed[:, None]), axis=0)
        index = int(np.argmin(mae))
        candidate = (float(mae[index]), float(a_grid[index]), float(b),
                     predicted[:, index])
        if best is None or candidate[0] < best[0]:
            best = candidate
    mae, a, b, predicted = best
    result = {
        'schema_version': 1, 'frozen': True,
        'fit_scope': 'train_events_only',
        'relation': 'Z=a*R^b', 'a': a, 'b': b,
        'pair_count': len(pairs), 'mae_mm_per_hour': mae,
        'rmse_mm_per_hour': float(np.sqrt(np.mean((predicted-observed)**2))),
        'station_csv': str(Path(station_csv).resolve()),
        'manifest': str(Path(manifest_path).resolve()),
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
