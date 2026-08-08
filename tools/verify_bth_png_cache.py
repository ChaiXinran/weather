"""Verify pixel identity and benchmark a BTH uint8 PNG frame cache."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--samples', type=int, default=600)
    parser.add_argument('--random-checks', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_root = Path(args.cache).resolve()
    document = json.loads(
        (cache_root / 'manifest.json').read_text(encoding='utf-8'))
    source_root = Path(document['source_root'])
    source_paths = {path.stem: path for path in source_root.rglob('*.png')}
    timestamps = document['timestamps']
    if len(source_paths) != len(timestamps):
        raise RuntimeError(
            f'Source/cache count mismatch: {len(source_paths)} vs '
            f'{len(timestamps)}')
    cache = np.load(cache_root / 'frames.npy', mmap_mode='r')

    rng = np.random.default_rng(args.seed)
    anchors = {0, len(timestamps) // 2, len(timestamps) - 1}
    random_count = min(args.random_checks, len(timestamps))
    check_indices = sorted(anchors | set(
        rng.choice(len(timestamps), random_count, replace=False).tolist()))
    for index in check_indices:
        stem = datetime.fromisoformat(timestamps[index]).strftime(
            '%Y-%m-%d-%H-%M-%S')
        with Image.open(source_paths[stem]) as image:
            pixels = np.asarray(image.convert('L'), dtype=np.uint8)
        if not np.array_equal(pixels, cache[index]):
            raise AssertionError(f'Pixel mismatch at index {index}: {stem}')

    count = min(args.samples, len(timestamps))
    benchmark_indices = np.linspace(
        0, len(timestamps) - 1, count, dtype=np.int64)
    started = time.perf_counter()
    for index in benchmark_indices:
        stem = datetime.fromisoformat(timestamps[index]).strftime(
            '%Y-%m-%d-%H-%M-%S')
        with Image.open(source_paths[stem]) as image:
            pixels = np.asarray(image.convert('L'), dtype=np.float32)
        (255.0 - pixels) / 255.0
    png_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for index in benchmark_indices:
        pixels = np.asarray(cache[index], dtype=np.float32)
        (255.0 - pixels) / 255.0
    cache_seconds = time.perf_counter() - started

    print(f"variable: {document['variable']}")
    print(f'frames: {len(timestamps)}')
    print(f'exact pixel checks: {len(check_indices)} passed')
    print(f'benchmark frames: {count}')
    print(f'png seconds: {png_seconds:.6f}')
    print(f'cache seconds: {cache_seconds:.6f}')
    print(f'speedup: {png_seconds / cache_seconds:.2f}x')


if __name__ == '__main__':
    main()
