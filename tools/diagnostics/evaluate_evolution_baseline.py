"""Evaluate a source-free EvolutionConvLSTM checkpoint on validation data."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openstl.core.precipitation_metrics import PrecipitationEvaluator
from openstl.datasets.dataloader_radar import BTHRadarDataset
from openstl.models import EvolutionConvLSTM_Model
from openstl.utils import load_config


def load_checkpoint(config, checkpoint):
    model = EvolutionConvLSTM_Model(4, [64, 64, 64, 64], config)
    payload = torch.load(checkpoint, map_location='cpu')
    state = {
        key[len('model.'):]: value
        for key, value in payload['state_dict'].items()
        if key.startswith('model.')
    }
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--allow-source', action='store_true',
                        help='evaluate a source-enabled evolution checkpoint')
    args = parser.parse_args()

    values = dict(load_config(args.config))
    values.update(in_shape=[10, 1, 66, 70], pre_seq_length=10,
                  aft_seq_length=20, total_length=30)
    config = SimpleNamespace(**values)
    if getattr(config, 'evolution_use_source', False) and not args.allow_source:
        raise ValueError('baseline evaluation requires source to be disabled')

    dataset = BTHRadarDataset(
        data_root=args.data_root, pre_seq_length=10, aft_seq_length=20,
        start_date='2025-05-01', end_date='2025-08-31',
        manifest_path=config.manifest_path, split='val',
        radar_cache_path=config.radar_cache_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        persistent_workers=True)
    model = load_checkpoint(config, args.checkpoint)
    evaluator = PrecipitationEvaluator(
        lead_count=20, thresholds=config.precip_thresholds,
        value_scale=config.radar_value_scale,
        value_unit=config.precip_value_unit,
        lead_minutes=config.lead_minutes,
        clip_range=config.precip_clip_range,
        case_threshold=config.case_threshold, case_count=config.case_count,
        event_id_source='manifest', convert_dbz_to_rain=True,
        zr_a=config.zr_a, zr_b=config.zr_b,
        wet_threshold=config.wet_threshold,
        grid_spacing_km=config.grid_spacing_km,
        neighborhood_windows=config.neighborhood_windows,
        object_iou_threshold=config.object_iou_threshold,
        bootstrap_repetitions=config.bootstrap_repetitions,
        bootstrap_seed=config.bootstrap_seed)

    offset = 0
    with torch.inference_mode():
        for history, target in loader:
            prediction = model(history.cuda()).cpu().numpy()
            batch = history.shape[0]
            indices = list(range(offset, offset + batch))
            evaluator.update(
                prediction, target.numpy(), history.numpy(),
                event_ids=[dataset.event_id_for_sample(i) for i in indices],
                sample_ids=indices)
            offset += batch

    args.output.mkdir(parents=True, exist_ok=True)
    evaluator.save(args.output / 'precipitation_evaluation')
    metadata = {
        'status': 'complete',
        'split': 'val',
        'sample_count': len(dataset),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'source_enabled': bool(getattr(config, 'evolution_use_source', False)),
        'forecast_steps': 20,
        'lead_minutes': config.lead_minutes,
        'thresholds_mm_h': config.precip_thresholds,
        'neighborhood_windows': config.neighborhood_windows,
        'event_manifest': config.manifest_path,
    }
    (args.output / 'baseline_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
