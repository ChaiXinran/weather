from datetime import datetime, timedelta
import json

import numpy as np
from PIL import Image

from openstl.datasets.dataloader_radar import BTHRadarDataset
from openstl.datasets.png_cache import CACHE_FORMAT


def _write_sequence(root, length=31, missing=None):
    timestamp = datetime(2025, 5, 1)
    for index in range(length):
        if index == missing:
            continue
        current = timestamp + index * timedelta(minutes=6)
        folder = root / current.strftime('%Y%m') / current.strftime('%Y%m%d')
        folder.mkdir(parents=True, exist_ok=True)
        pixels = np.full((66, 70), 255 - index, dtype=np.uint8)
        Image.fromarray(pixels).save(
            folder / current.strftime('%Y-%m-%d-%H-%M-%S.png'))


def test_radar_dataset_shape_decode_and_metadata(tmp_path):
    _write_sequence(tmp_path)
    dataset = BTHRadarDataset(
        tmp_path, '2025-05-01', '2025-05-01',
        pre_seq_length=10, aft_seq_length=20)

    inputs, targets = dataset[0]
    assert len(dataset) == 2
    assert inputs.shape == (10, 1, 66, 70)
    assert targets.shape == (20, 1, 66, 70)
    assert inputs[0, 0, 0, 0].item() == 0.0
    assert np.isclose(targets[0, 0, 0, 0].item(), 10 / 255)
    assert dataset.sample_metadata(0)['target_end'] == '2025-05-01T02:54:00'


def test_radar_dataset_rejects_windows_crossing_a_missing_frame(tmp_path):
    _write_sequence(tmp_path, length=61, missing=30)
    dataset = BTHRadarDataset(
        tmp_path, '2025-05-01', '2025-05-01',
        pre_seq_length=10, aft_seq_length=20)

    assert len(dataset) == 2
    for sequence in dataset.samples:
        assert datetime(2025, 5, 1, 3, 0) not in sequence


def test_radar_uint8_cache_matches_png_decode(tmp_path):
    _write_sequence(tmp_path)
    png_dataset = BTHRadarDataset(
        tmp_path, '2025-05-01', '2025-05-01',
        pre_seq_length=10, aft_seq_length=20)
    timestamps = sorted(png_dataset.frames)
    cache_dir = tmp_path / 'RADAR_CACHE_UINT8'
    cache_dir.mkdir()
    frames = np.stack([
        np.asarray(Image.open(png_dataset.frames[timestamp]).convert('L'))
        for timestamp in timestamps
    ]).astype(np.uint8)
    np.save(cache_dir / 'frames.npy', frames)
    (cache_dir / 'manifest.json').write_text(json.dumps({
        'format': 'bth-radar-uint8-npy-v1',
        'dtype': 'uint8',
        'shape': list(frames.shape),
        'timestamps': [timestamp.isoformat() for timestamp in timestamps],
    }), encoding='utf-8')

    cached_dataset = BTHRadarDataset(
        tmp_path, '2025-05-01', '2025-05-01',
        pre_seq_length=10, aft_seq_length=20,
        radar_cache_path='RADAR_CACHE_UINT8')
    png_inputs, png_targets = png_dataset[0]
    cache_inputs, cache_targets = cached_dataset[0]

    assert np.array_equal(cache_inputs.numpy(), png_inputs.numpy())
    assert np.array_equal(cache_targets.numpy(), png_targets.numpy())
    assert '#0' in cached_dataset.sample_metadata(0)['files'][0]


def test_rain_uint8_cache_matches_png_truth(tmp_path):
    radar_root = tmp_path / 'RADAR_2025_S'
    rain_root = tmp_path / 'RAIN_2025_S'
    _write_sequence(radar_root)
    _write_sequence(rain_root)

    png_dataset = BTHRadarDataset(
        tmp_path, '2025-05-01', '2025-05-01',
        pre_seq_length=10, aft_seq_length=20,
        evaluation_truth='rain_png')
    timestamps = sorted(png_dataset.rain_frames)
    cache_dir = tmp_path / 'RAIN_CACHE_UINT8'
    cache_dir.mkdir()
    frames = np.stack([
        np.asarray(Image.open(png_dataset.rain_frames[t]).convert('L'))
        for t in timestamps]).astype(np.uint8)
    np.save(cache_dir / 'frames.npy', frames)
    (cache_dir / 'manifest.json').write_text(json.dumps({
        'format': CACHE_FORMAT,
        'variable': 'rain',
        'dtype': 'uint8',
        'shape': list(frames.shape),
        'timestamps': [timestamp.isoformat() for timestamp in timestamps],
    }), encoding='utf-8')

    cached_dataset = BTHRadarDataset(
        tmp_path, '2025-05-01', '2025-05-01',
        pre_seq_length=10, aft_seq_length=20,
        evaluation_truth='rain_png',
        rain_cache_path='RAIN_CACHE_UINT8')

    assert np.array_equal(
        cached_dataset.rain_targets([0]), png_dataset.rain_targets([0]))
