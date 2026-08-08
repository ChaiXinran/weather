"""Build sample-aligned packed V3a routing labels from a frozen direct prior."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from openstl.datasets.dataloader_radar import BTHRadarDataset
from openstl.models import DirectPhysicsHybrid_Model
from openstl.modules.evolution_operator import normalized_dbz_to_rain
from openstl.modules.v3a_routing import build_packed_routing_target
from openstl.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--direct_checkpoint', default=None)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        choices=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--minimum_score', type=float, default=0.1)
    parser.add_argument('--ambiguity_margin', type=float, default=0.05)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_for(split, values, data_root):
    return BTHRadarDataset(
        data_root=data_root, start_date='2025-05-01', end_date='2025-08-31',
        pre_seq_length=int(values.get('pre_seq_length', 10)),
        aft_seq_length=int(values.get('aft_seq_length', 20)), stride=1,
        manifest_path=values.get('manifest_path'), split=split,
        radar_cache_path=values.get('radar_cache_path'),
        evaluation_truth='radar')


def main():
    options = parse_args()
    values = dict(load_config(options.config_file))
    values.update(in_shape=[10, 1, 66, 70], pre_seq_length=10,
                  aft_seq_length=20, total_length=30)
    configs = SimpleNamespace(**values)
    direct_checkpoint = Path(
        options.direct_checkpoint or values['hybrid_direct_checkpoint'])
    if not direct_checkpoint.is_file():
        raise FileNotFoundError(direct_checkpoint)
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)

    model = DirectPhysicsHybrid_Model(configs)
    loaded = model.load_direct_checkpoint(str(direct_checkpoint))
    model.freeze_direct()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f'Loaded {loaded} direct tensors on {device}')

    manifest = {
        'format': 'bth-v3a-routing-uint8-v1',
        'config_file': str(options.config_file),
        'direct_checkpoint': str(direct_checkpoint.resolve()),
        'direct_checkpoint_sha256': sha256(direct_checkpoint),
        'protocol': {
            'thresholds_mm_h': [16.0, 32.0],
            'radius_grid': options.radius,
            'iou_threshold': options.iou_threshold,
            'area_ratio': [0.5, 2.0],
            'iou_weight': 0.5,
            'distance_weight': 0.5,
            'minimum_score': options.minimum_score,
            'ambiguity_margin': options.ambiguity_margin,
            'soft_label_weights': {'16': 1.0, '32': 1.5},
        },
        'splits': {},
    }
    for split in options.splits:
        dataset = dataset_for(split, values, options.data_root)
        loader = DataLoader(
            dataset, batch_size=options.batch_size, shuffle=False,
            num_workers=options.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=options.num_workers > 0)
        shape = (len(dataset), 20, 66, 70)
        final_path = output / f'{split}_labels.npy'
        temporary_path = output / f'{split}_labels.npy.tmp'
        labels = np.lib.format.open_memmap(
            temporary_path, mode='w+', dtype=np.uint8, shape=shape)
        counts = np.zeros((2, 4), dtype=np.int64)
        offset = 0
        with torch.inference_mode():
            for batch_index, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.to(device, non_blocking=True)
                direct = model.direct_forecast(batch_x)
                direct_rain = normalized_dbz_to_rain(
                    direct, configs.radar_value_scale,
                    configs.zr_a, configs.zr_b).cpu().numpy()
                target_rain = normalized_dbz_to_rain(
                    batch_y.to(device, non_blocking=True),
                    configs.radar_value_scale,
                    configs.zr_a, configs.zr_b).cpu().numpy()
                for item in range(batch_x.shape[0]):
                    packed = build_packed_routing_target(
                        direct_rain[item, :, 0], target_rain[item, :, 0],
                        radius=options.radius,
                        iou_threshold=options.iou_threshold,
                        minimum_score=options.minimum_score,
                        ambiguity_margin=options.ambiguity_margin)
                    labels[offset] = packed
                    for threshold_index, route in enumerate(
                            (packed & 3, (packed >> 2) & 3)):
                        counts[threshold_index] += np.bincount(
                            route.reshape(-1), minlength=4)
                    offset += 1
                if (batch_index + 1) % 25 == 0 or offset == len(dataset):
                    print(f'{split}: {offset}/{len(dataset)}', flush=True)
        labels.flush()
        del labels
        os.replace(temporary_path, final_path)
        manifest['splits'][split] = {
            'shape': list(shape),
            'sample_keys': [sample[0].isoformat() for sample in dataset.samples],
            'counts': {
                threshold: dict(zip(
                    ('ignore', 'preserve', 'motion', 'decay'),
                    counts[index].tolist()))
                for index, threshold in enumerate(('16', '32'))
            },
        }
    temporary_manifest = output / 'manifest.json.tmp'
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary_manifest, output / 'manifest.json')
    print(f'V3a routing cache ready: {output.resolve()}')


if __name__ == '__main__':
    main()
