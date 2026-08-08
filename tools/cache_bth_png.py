"""Build lossless mmap caches for timestamped BTH PWV or Rain PNGs."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S'
CACHE_FORMAT = 'bth-png-uint8-npy-v1'
VARIABLE_METADATA = {
    'pwv': {
        'decode': 'pwv_mm=(255-pixel)*80/255',
        'default_source': 'PWV_2025_S',
        'default_output': 'PWV_CACHE_UINT8',
    },
    'rain': {
        'decode': 'rain_rate_mm_h=(255-pixel)*35/255',
        'default_source': 'RAIN_2025_S',
        'default_output': 'RAIN_CACHE_UINT8',
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', required=True,
                        choices=sorted(VARIABLE_METADATA))
    parser.add_argument(
        '--data-root',
        help='DATA_2025_S root; supplies conventional source/output paths')
    parser.add_argument('--source', help='Override source PNG directory')
    parser.add_argument('--output', help='Override cache directory')
    parser.add_argument('--height', type=int, default=66)
    parser.add_argument('--width', type=int, default=70)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve_paths(args):
    metadata = VARIABLE_METADATA[args.variable]
    if args.data_root:
        root = Path(args.data_root).resolve()
        source = Path(args.source).resolve() if args.source else (
            root / metadata['default_source'])
        output = Path(args.output).resolve() if args.output else (
            root / metadata['default_output'])
    else:
        if not args.source or not args.output:
            raise ValueError(
                'Use --data-root or provide both --source and --output')
        source = Path(args.source).resolve()
        output = Path(args.output).resolve()
    return source, output


def build_cache(variable, source, output, height=66, width=70,
                overwrite=False):
    paths = sorted(source.rglob('*.png'))
    if not paths:
        raise FileNotFoundError(f'No PNG frames found under {source}')

    timestamps = []
    seen = {}
    for path in paths:
        timestamp = datetime.strptime(path.stem, TIMESTAMP_FORMAT)
        if timestamp in seen:
            raise RuntimeError(
                f'Duplicate {variable.upper()} timestamp {timestamp}: '
                f'{seen[timestamp]} and {path}')
        seen[timestamp] = path
        timestamps.append(timestamp)

    output.mkdir(parents=True, exist_ok=True)
    array_path = output / 'frames.npy'
    manifest_path = output / 'manifest.json'
    if (array_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f'Cache already exists in {output}; use --overwrite explicitly')

    temporary_array = output / 'frames.npy.tmp'
    temporary_manifest = output / 'manifest.json.tmp'
    cache = np.lib.format.open_memmap(
        temporary_array, mode='w+', dtype=np.uint8,
        shape=(len(paths), height, width))
    try:
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                image = image.convert('L')
                if image.size != (width, height):
                    raise ValueError(
                        f'Unexpected image size {image.size} in {path}')
                cache[index] = np.asarray(image, dtype=np.uint8)
            if (index + 1) % 1000 == 0 or index + 1 == len(paths):
                print(f'cached {index + 1}/{len(paths)} frames', flush=True)
        cache.flush()
    finally:
        del cache

    document = {
        'format': CACHE_FORMAT,
        'variable': variable,
        'dtype': 'uint8',
        'shape': [len(paths), height, width],
        'timestamps': [value.isoformat() for value in timestamps],
        'source_root': str(source),
        'source_frame_count': len(paths),
        'decode': VARIABLE_METADATA[variable]['decode'],
        'storage': 'lossless_raw_inverse_grayscale_pixels',
    }
    temporary_manifest.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding='utf-8')
    os.replace(temporary_array, array_path)
    os.replace(temporary_manifest, manifest_path)
    return array_path, manifest_path


def main():
    args = parse_args()
    source, output = resolve_paths(args)
    array_path, _ = build_cache(
        args.variable, source, output, args.height, args.width,
        args.overwrite)
    print(f'cache ready: {output}')
    print(f'array bytes: {array_path.stat().st_size}')


if __name__ == '__main__':
    main()
