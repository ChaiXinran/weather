"""Radar PNG dataset for the BTH precipitation nowcasting task."""

from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader
from openstl.datasets.png_cache import BTHPNGCache
from openstl.datasets.v3a_routing_cache import V3ARoutingCache


TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S'
FRAME_INTERVAL = timedelta(minutes=6)


def _parse_timestamp(path):
    try:
        return datetime.strptime(Path(path).stem, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f'Radar filename must use {TIMESTAMP_FORMAT!r}: {path}') from exc


def _resolve_radar_root(data_root):
    root = Path(data_root)
    candidates = (root / 'RADAR_2025_S', root)
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.rglob('*.png'), None) is not None:
            return candidate
    raise FileNotFoundError(
        f'No Radar PNG files found under {root} or {root / "RADAR_2025_S"}')


class BTHRadarDataset(Dataset):
    """Load strictly continuous Radar sequences with positive intensity semantics.

    PNG values are decoded from the source's inverse grayscale convention:
    ``normalized_reflectivity = (255 - pixel) / 255``. Returned tensors have
    shapes ``[T_in, 1, H, W]`` and ``[T_out, 1, H, W]``.
    """

    def __init__(self,
                 data_root,
                 start_date,
                 end_date,
                 pre_seq_length=10,
                 aft_seq_length=20,
                 stride=1,
                 expected_height=66,
                 expected_width=70,
                 manifest_path=None,
                 split=None,
                 evaluation_truth='radar',
                 radar_cache_path=None,
                 rain_cache_path=None,
                 rain_truth_lag_minutes=0,
                 rain_truth_row_shift=0,
                 rain_truth_col_shift=0,
                 routing_cache_path=None):
        self.data_root = _resolve_radar_root(data_root)
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        if self.end_date < self.start_date:
            raise ValueError('end_date must not precede start_date')
        if pre_seq_length <= 0 or aft_seq_length <= 0:
            raise ValueError('Sequence lengths must be positive')
        if stride <= 0:
            raise ValueError('stride must be positive')

        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.total_length = pre_seq_length + aft_seq_length
        self.expected_size = (expected_width, expected_height)
        self.stride = stride
        self.mean = 0.0
        self.std = 1.0
        self.data_name = 'bth_radar'
        self.event_id_source = 'independent_weather_event' if manifest_path else 'date_proxy'
        self.evaluation_truth = evaluation_truth
        self.rain_truth_lag = timedelta(minutes=rain_truth_lag_minutes)
        self.rain_truth_row_shift = int(rain_truth_row_shift)
        self.rain_truth_col_shift = int(rain_truth_col_shift)
        self.radar_cache_path = None
        if radar_cache_path:
            cache_path = Path(radar_cache_path)
            self.radar_cache_path = (
                cache_path if cache_path.is_absolute()
                else Path(data_root) / cache_path)
        self._radar_cache = None
        self.rain_frames = None
        self.rain_cache = None
        if evaluation_truth == 'rain_png':
            if rain_cache_path:
                cache_path = Path(rain_cache_path)
                cache_path = (cache_path if cache_path.is_absolute()
                              else Path(data_root) / cache_path)
                self.rain_cache = BTHPNGCache(
                    cache_path, 'rain', self.expected_size)
                self.rain_frames = {
                    timestamp: index
                    for timestamp, index in self.rain_cache.frames.items()
                    if self.start_date <= timestamp.date() <= self.end_date}
            else:
                rain_root = Path(data_root) / 'RAIN_2025_S'
                if not rain_root.is_dir():
                    raise FileNotFoundError(f'Rain root not found: {rain_root}')
                self.rain_frames = {
                    _parse_timestamp(path): path
                    for path in rain_root.rglob('*.png')
                    if self.start_date <= _parse_timestamp(path).date()
                    <= self.end_date
                }
        elif evaluation_truth != 'radar':
            raise ValueError('evaluation_truth must be radar or rain_png')

        self.frames = (
            self._discover_cached_frames()
            if self.radar_cache_path else self._discover_frames())
        self.sample_event_ids = None
        if manifest_path:
            if split not in {'train', 'val', 'test'}:
                raise ValueError('split is required when manifest_path is used')
            self.samples, self.sample_event_ids = self._load_manifest(
                manifest_path, split)
        else:
            self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f'No continuous {self.total_length}-frame sequences found from '
                f'{start_date} through {end_date} in {self.data_root}')
        self.routing_cache = None
        if routing_cache_path and split in {'train', 'val'}:
            cache_path = Path(routing_cache_path)
            cache_path = (cache_path if cache_path.is_absolute()
                          else Path(data_root) / cache_path)
            sample_keys = [sample[0].isoformat() for sample in self.samples]
            self.routing_cache = V3ARoutingCache(
                cache_path, split, sample_keys,
                (self.aft_seq_length, self.expected_size[1],
                 self.expected_size[0]))

    def _discover_frames(self):
        frames = {}
        for path in self.data_root.rglob('*.png'):
            timestamp = _parse_timestamp(path)
            if self.start_date <= timestamp.date() <= self.end_date:
                if timestamp in frames:
                    raise RuntimeError(
                        f'Duplicate Radar timestamp {timestamp}: '
                        f'{frames[timestamp]} and {path}')
                frames[timestamp] = path
        return frames

    def _discover_cached_frames(self):
        manifest_path = self.radar_cache_path / 'manifest.json'
        array_path = self.radar_cache_path / 'frames.npy'
        if not manifest_path.is_file() or not array_path.is_file():
            raise FileNotFoundError(
                f'Radar cache requires {manifest_path} and {array_path}')
        document = json.loads(manifest_path.read_text(encoding='utf-8'))
        if document.get('format') != 'bth-radar-uint8-npy-v1':
            raise ValueError(
                f'Unsupported Radar cache format in {manifest_path}')
        expected_shape = [
            len(document['timestamps']),
            self.expected_size[1],
            self.expected_size[0],
        ]
        if document.get('shape') != expected_shape:
            raise ValueError(
                f'Radar cache shape metadata {document.get("shape")} does '
                f'not match expected {expected_shape}')
        frames = {}
        for index, value in enumerate(document['timestamps']):
            timestamp = datetime.fromisoformat(value)
            if self.start_date <= timestamp.date() <= self.end_date:
                if timestamp in frames:
                    raise RuntimeError(
                        f'Duplicate timestamp in Radar cache: {timestamp}')
                frames[timestamp] = index
        if not frames:
            raise RuntimeError(
                f'Radar cache contains no frames from {self.start_date} '
                f'through {self.end_date}')
        self._radar_cache_array_path = array_path
        return frames

    def _build_samples(self):
        samples = []
        timestamps = sorted(self.frames)
        timestamp_set = set(timestamps)
        for timestamp in timestamps[::self.stride]:
            sequence = tuple(
                timestamp + index * FRAME_INTERVAL
                for index in range(self.total_length))
            if all(item in timestamp_set for item in sequence):
                samples.append(sequence)
        return samples

    def _load_manifest(self, manifest_path, split):
        document = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        expected = (self.pre_seq_length + self.aft_seq_length)
        samples, event_ids = [], []
        split_items = [item for item in document['samples']
                       if item['split'] == split]
        for item in split_items[::self.stride]:
            first = datetime.fromisoformat(item['input_start'])
            timestamps = tuple(first + index * FRAME_INTERVAL
                               for index in range(expected))
            if any(timestamps[i + 1] - timestamps[i] != FRAME_INTERVAL
                   for i in range(len(timestamps) - 1)):
                raise ValueError(
                    f"Manifest sample {item['sample_id']} is not continuous")
            if not all(timestamp in self.frames for timestamp in timestamps):
                raise FileNotFoundError(
                    f"Manifest sample {item['sample_id']} references missing frames")
            samples.append(timestamps)
            event_ids.append(item['event_id'])
        return samples, event_ids

    def __len__(self):
        return len(self.samples)

    def _read_frame(self, timestamp):
        source = self.frames[timestamp]
        if self.radar_cache_path:
            if self._radar_cache is None:
                self._radar_cache = np.load(
                    self._radar_cache_array_path, mmap_mode='r')
            pixels = np.asarray(
                self._radar_cache[source], dtype=np.float32)
        else:
            path = source
            with Image.open(path) as image:
                image = image.convert('L')
                if image.size != self.expected_size:
                    raise ValueError(
                        f'Unexpected Radar size {image.size} in {path}; '
                        f'expected {self.expected_size}')
                pixels = np.asarray(image, dtype=np.float32)
        decoded = (255.0 - pixels) / 255.0
        return torch.from_numpy(decoded).unsqueeze(0)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Every DataLoader worker opens its own read-only mmap handle lazily.
        state['_radar_cache'] = None
        if state.get('rain_cache') is not None:
            state['rain_cache']._array = None
        if state.get('routing_cache') is not None:
            state['routing_cache']._array = None
        return state

    def __getitem__(self, index):
        timestamps = self.samples[index]
        sequence = torch.stack(
            [self._read_frame(timestamp) for timestamp in timestamps], dim=0)
        inputs = sequence[:self.pre_seq_length]
        targets = sequence[self.pre_seq_length:]
        if self.routing_cache is None:
            return inputs, targets
        routing = torch.from_numpy(
            self.routing_cache.read(index).astype(np.int64, copy=True))
        return inputs, targets, routing

    def sample_metadata(self, index):
        """Return traceability metadata without reading image pixels."""
        timestamps = self.samples[index]
        return {
            'input_start': timestamps[0].isoformat(),
            'input_end': timestamps[self.pre_seq_length - 1].isoformat(),
            'target_start': timestamps[self.pre_seq_length].isoformat(),
            'target_end': timestamps[-1].isoformat(),
            'files': [
                (f'{self._radar_cache_array_path}#{self.frames[timestamp]}'
                 if self.radar_cache_path else str(self.frames[timestamp]))
                for timestamp in timestamps
            ],
            'event_id': self.event_id_for_sample(index),
            'event_id_source': self.event_id_source,
        }

    def event_id_for_sample(self, index):
        if self.sample_event_ids is not None:
            return self.sample_event_ids[index]
        return self.samples[index][self.pre_seq_length].strftime('%Y-%m-%d')

    def rain_targets(self, indices):
        """Read direct Rain-PNG targets as normalized 0--1 rain rate."""
        if self.rain_frames is None:
            raise RuntimeError('Dataset was not configured with rain_png truth')
        batches = []
        for index in indices:
            timestamps = [
                timestamp + self.rain_truth_lag
                for timestamp in self.samples[index][self.pre_seq_length:]]
            frames = []
            for timestamp in timestamps:
                source = self.rain_frames.get(timestamp)
                if source is None:
                    raise FileNotFoundError(f'Missing Rain frame at {timestamp}')
                if self.rain_cache is not None:
                    pixels = self.rain_cache.read(timestamp)
                else:
                    with Image.open(source) as image:
                        pixels = np.asarray(
                            image.convert('L'), dtype=np.float32)
                decoded = (255.0 - pixels) / 255.0
                aligned = np.full(decoded.shape, np.nan, dtype=np.float32)
                rows, cols = decoded.shape
                dr = self.rain_truth_row_shift
                dc = self.rain_truth_col_shift
                r0, r1 = max(0, dr), min(rows, rows + dr)
                c0, c1 = max(0, dc), min(cols, cols + dc)
                aligned[r0:r1, c0:c1] = decoded[
                    r0-dr:r1-dr, c0-dc:c1-dc]
                frames.append(aligned)
            batches.append(np.stack(frames)[:, None, ...])
        return np.stack(batches)


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              pre_seq_length=10,
              aft_seq_length=20,
              distributed=False,
              use_prefetcher=False,
              drop_last=False,
              train_date_range=('2025-05-01', '2025-07-31'),
              val_date_range=('2025-08-01', '2025-08-15'),
              test_date_range=('2025-08-16', '2025-08-31'),
              sample_stride=1,
              manifest_path=None,
              evaluation_truth='radar',
              radar_cache_path=None,
              rain_cache_path=None,
              rain_truth_lag_minutes=0,
              rain_truth_row_shift=0,
              rain_truth_col_shift=0,
              routing_cache_path=None,
              **kwargs):
    """Build loaders using a provisional, non-overlapping chronological split.

    The August boundary is intentionally configurable and should be replaced by
    the frozen event-based split before reported experiments.
    """
    del kwargs
    common = dict(
        data_root=data_root,
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        stride=sample_stride,
        evaluation_truth=evaluation_truth,
        radar_cache_path=radar_cache_path,
        rain_cache_path=rain_cache_path,
        rain_truth_lag_minutes=rain_truth_lag_minutes,
        rain_truth_row_shift=rain_truth_row_shift,
        rain_truth_col_shift=rain_truth_col_shift,
        routing_cache_path=routing_cache_path,
    )
    if manifest_path:
        full_range = dict(start_date='2025-05-01', end_date='2025-08-31')
        train_set = BTHRadarDataset(
            **full_range, manifest_path=manifest_path, split='train', **common)
        val_set = BTHRadarDataset(
            **full_range, manifest_path=manifest_path, split='val', **common)
        test_set = BTHRadarDataset(
            **full_range, manifest_path=manifest_path, split='test', **common)
    else:
        train_set = BTHRadarDataset(
            start_date=train_date_range[0], end_date=train_date_range[1], **common)
        val_set = BTHRadarDataset(
            start_date=val_date_range[0], end_date=val_date_range[1], **common)
        test_set = BTHRadarDataset(
            start_date=test_date_range[0], end_date=test_date_range[1], **common)

    loader_common = dict(
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        drop_last=drop_last,
        **loader_common)
    val_loader = create_loader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        drop_last=False,
        **loader_common)
    test_loader = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        drop_last=False,
        **loader_common)
    return train_loader, val_loader, test_loader
