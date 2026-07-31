"""Build a lossless, memory-mapped uint8 cache from BTH Radar PNG frames."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source', required=True,
        help='RADAR_2025_S directory containing timestamped PNG files')
    parser.add_argument(
        '--output', required=True,
        help='Cache directory; frames.npy and manifest.json are created here')
    parser.add_argument('--height', type=int, default=66)
    parser.add_argument('--width', type=int, default=70)
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Replace an existing complete cache')
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    paths = sorted(source.rglob('*.png'))
    if not paths:
        raise FileNotFoundError(f'No PNG frames found under {source}')

    timestamps = []
    previous = None
    for path in paths:
        timestamp = datetime.strptime(path.stem, TIMESTAMP_FORMAT)
        if timestamp == previous:
            raise RuntimeError(f'Duplicate Radar timestamp: {timestamp}')
        timestamps.append(timestamp)
        previous = timestamp

    output.mkdir(parents=True, exist_ok=True)
    array_path = output / 'frames.npy'
    manifest_path = output / 'manifest.json'
    if ((array_path.exists() or manifest_path.exists())
            and not args.overwrite):
        raise FileExistsError(
            f'Cache already exists in {output}; use --overwrite explicitly')

    temporary_array = output / 'frames.npy.tmp'
    temporary_manifest = output / 'manifest.json.tmp'
    cache = np.lib.format.open_memmap(
        temporary_array, mode='w+', dtype=np.uint8,
        shape=(len(paths), args.height, args.width))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert('L')
            if image.size != (args.width, args.height):
                raise ValueError(
                    f'Unexpected image size {image.size} in {path}')
            cache[index] = np.asarray(image, dtype=np.uint8)
        if (index + 1) % 1000 == 0 or index + 1 == len(paths):
            print(f'cached {index + 1}/{len(paths)} frames', flush=True)
    cache.flush()
    del cache

    document = {
        'format': 'bth-radar-uint8-npy-v1',
        'dtype': 'uint8',
        'shape': [len(paths), args.height, args.width],
        'timestamps': [value.isoformat() for value in timestamps],
        'source_root': str(source),
        'source_frame_count': len(paths),
        'decode': 'normalized_reflectivity=(255-pixel)/255',
    }
    temporary_manifest.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding='utf-8')
    os.replace(temporary_array, array_path)
    os.replace(temporary_manifest, manifest_path)
    print(f'cache ready: {output}')
    print(f'array bytes: {array_path.stat().st_size}')


if __name__ == '__main__':
    main()
