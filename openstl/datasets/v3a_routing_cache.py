"""Read-only sample-aligned mmap cache for V3a packed routing labels."""

import json
from pathlib import Path

import numpy as np


class V3ARoutingCache:
    FORMAT = 'bth-v3a-routing-uint8-v1'

    def __init__(self, root, split, sample_keys, expected_shape):
        self.root = Path(root)
        self.split = str(split)
        manifest_path = self.root / 'manifest.json'
        array_path = self.root / f'{self.split}_labels.npy'
        if not manifest_path.is_file() or not array_path.is_file():
            raise FileNotFoundError(
                f'V3a routing cache requires {manifest_path} and {array_path}')
        document = json.loads(manifest_path.read_text(encoding='utf-8'))
        if document.get('format') != self.FORMAT:
            raise ValueError(f'Unsupported V3a routing cache: {manifest_path}')
        split_document = document.get('splits', {}).get(self.split)
        if split_document is None:
            raise KeyError(f'Routing cache has no {self.split!r} split')
        if list(sample_keys) != split_document.get('sample_keys', []):
            raise ValueError(
                f'V3a routing sample order mismatch for split {self.split}')
        shape = [len(sample_keys), *expected_shape]
        if split_document.get('shape') != shape:
            raise ValueError(
                f'V3a routing shape metadata mismatch: expected {shape}, '
                f'got {split_document.get("shape")}')
        self.array_path = array_path
        self.expected_shape = tuple(shape)
        self._array = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_array'] = None
        return state

    def read(self, index):
        if self._array is None:
            self._array = np.load(self.array_path, mmap_mode='r')
            if self._array.dtype != np.uint8:
                raise ValueError('V3a routing cache must use uint8')
            if tuple(self._array.shape) != self.expected_shape:
                raise ValueError('V3a routing cache array shape changed')
        return np.asarray(self._array[index], dtype=np.uint8)
