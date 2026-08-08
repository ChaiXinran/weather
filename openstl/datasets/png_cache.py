"""Memory-mapped lossless caches for timestamped BTH PNG fields."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


CACHE_FORMAT = 'bth-png-uint8-npy-v1'


class BTHPNGCache:
    """Validate and lazily read a timestamp-indexed uint8 frame cache."""

    def __init__(self, cache_path, variable, expected_size=(70, 66)):
        self.path = Path(cache_path)
        self.variable = variable.lower()
        self.array_path = self.path / 'frames.npy'
        manifest_path = self.path / 'manifest.json'
        if not manifest_path.is_file() or not self.array_path.is_file():
            raise FileNotFoundError(
                f'{self.variable.upper()} cache requires {manifest_path} '
                f'and {self.array_path}')

        document = json.loads(manifest_path.read_text(encoding='utf-8'))
        if document.get('format') != CACHE_FORMAT:
            raise ValueError(f'Unsupported cache format in {manifest_path}')
        if document.get('variable') != self.variable:
            raise ValueError(
                f"Cache variable {document.get('variable')!r} does not match "
                f'{self.variable!r}')
        expected_shape = [
            len(document['timestamps']), expected_size[1], expected_size[0]]
        if document.get('dtype') != 'uint8':
            raise ValueError(f'Cache dtype must be uint8 in {manifest_path}')
        if document.get('shape') != expected_shape:
            raise ValueError(
                f"Cache shape {document.get('shape')} does not match "
                f'{expected_shape}')

        self.frames = {}
        for index, value in enumerate(document['timestamps']):
            timestamp = datetime.fromisoformat(value)
            if timestamp in self.frames:
                raise RuntimeError(
                    f'Duplicate timestamp in {self.variable} cache: {timestamp}')
            self.frames[timestamp] = index
        self._array = None

    def read(self, timestamp, dtype=np.float32):
        if self._array is None:
            self._array = np.load(self.array_path, mmap_mode='r')
        try:
            index = self.frames[timestamp]
        except KeyError as exc:
            raise FileNotFoundError(
                f'Missing {self.variable.upper()} frame at {timestamp}') from exc
        return np.asarray(self._array[index], dtype=dtype)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Each DataLoader worker creates its own read-only mmap lazily.
        state['_array'] = None
        return state

