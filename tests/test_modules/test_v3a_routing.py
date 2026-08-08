import numpy as np
import torch
import json

from openstl.modules.v3a_routing import (
    ROUTE_DECAY, ROUTE_MOTION, build_packed_routing_target,
    decode_packed_routing_target)
from openstl.datasets.v3a_routing_cache import V3ARoutingCache


def test_shifted_object_routes_to_motion():
    direct = np.zeros((1, 12, 12), dtype=np.float32)
    target = np.zeros_like(direct)
    direct[0, 3:7, 3:7] = 20.0
    target[0, 3:7, 4:8] = 20.0
    packed = build_packed_routing_target(direct, target, radius=2.0)
    route16 = packed & 3
    assert np.count_nonzero(route16 == ROUTE_MOTION) > 0


def test_unmatched_direct_object_routes_to_decay():
    direct = np.zeros((1, 12, 12), dtype=np.float32)
    target = np.zeros_like(direct)
    direct[0, 3:7, 3:7] = 35.0
    packed = build_packed_routing_target(direct, target, radius=2.0)
    assert np.all((packed[0, 3:7, 3:7] & 3) == ROUTE_DECAY)
    assert np.all(((packed[0, 3:7, 3:7] >> 2) & 3) == ROUTE_DECAY)


def test_multithreshold_decode_is_soft_and_normalized():
    # 16 says preserve (1), 32 says motion (2).
    packed = torch.tensor([[[1 | (2 << 2)]]], dtype=torch.uint8)
    probability, valid = decode_packed_routing_target(
        packed, weight16=1.0, weight32=1.5)
    assert valid.item()
    assert torch.allclose(probability.sum(dim=2), torch.ones_like(valid.float()))
    assert torch.allclose(
        probability[0, 0, :, 0, 0], torch.tensor([0.4, 0.6, 0.0]))


def test_routing_cache_validates_sample_order(tmp_path):
    labels = np.zeros((2, 20, 4, 5), dtype=np.uint8)
    np.save(tmp_path / 'train_labels.npy', labels)
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'format': 'bth-v3a-routing-uint8-v1',
        'splits': {'train': {
            'shape': [2, 20, 4, 5],
            'sample_keys': ['a', 'b'],
        }},
    }), encoding='utf-8')
    cache = V3ARoutingCache(tmp_path, 'train', ['a', 'b'], (20, 4, 5))
    assert cache.read(1).shape == (20, 4, 5)
