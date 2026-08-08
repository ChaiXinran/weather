"""Validation attribution for V3a candidates, protocol oracle, and router."""

import argparse
import json
from pathlib import Path

import torch

from openstl.api import BaseExperiment
from openstl.modules.evolution_operator import rain_to_normalized_dbz
from openstl.modules.v3a_routing import decode_packed_routing_target
from openstl.utils import default_parser
from tools.evaluate_direct_physics_attribution import (
    empty_state, update_state, summarize)


BRANCHES = ('direct', 'motion', 'decay', 'oracle_routed', 'learned_routed')


def write_markdown(document, checkpoint):
    lines = [
        '# V3a Candidate and Routing Attribution', '',
        f'Checkpoint: `{checkpoint}`', '',
        'Validation only; target-derived oracle labels are never inference inputs.',
        '', '## Branch Metrics', '',
        '| Branch | Period | Threshold | CSI | POD | FAR | Bias |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for branch in BRANCHES:
        for period in ('0_1h', '1_2h'):
            for threshold in ('16', '32'):
                row = document[branch][period][threshold]
                lines.append(
                    f'| {branch} | {period} | {threshold} | '
                    f'{row["csi"]:.6f} | {row["pod"]:.6f} | '
                    f'{row["far"]:.6f} | {row["bias"]:.6f} |')
    lines.extend([
        '', '## Transitions Relative to Direct', '',
        '| Branch | Period | Threshold | Miss->Hit | Hit->Miss | FA->Correct | Correct->FA |',
        '|---|---|---:|---:|---:|---:|---:|',
    ])
    for branch in BRANCHES[1:]:
        for period in ('0_1h', '1_2h'):
            for threshold in ('16', '32'):
                row = document[branch][period][threshold]
                lines.append(
                    f'| {branch} | {period} | {threshold} | '
                    f'{row["miss_to_hit"]} | {row["hit_to_miss"]} | '
                    f'{row["fa_to_correct"]} | {row["correct_to_fa"]} |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--val_batch_size', type=int, default=None)
    options = parser.parse_args()
    work_dir = Path(options.work_dir).expanduser().resolve()
    params = json.loads(
        (work_dir / 'model_param.json').read_text(encoding='utf-8'))
    for key, value in default_parser().items():
        params.setdefault(key, value)
        if params[key] is None:
            params[key] = value
    for key in ('data_root', 'batch_size', 'val_batch_size'):
        value = getattr(options, key)
        if value is not None:
            params[key] = value
    checkpoint = (Path(options.ckpt_path).expanduser().resolve()
                  if options.ckpt_path else
                  work_dir / 'checkpoints' / 'best_val_csi.ckpt')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # ckpt_path tells the V3a method that a complete state will be restored,
    # avoiding any dependency on the original V2 initialization path.
    params.update(ckpt_path=str(checkpoint), init_from_ckpt=None, test=False,
                  no_display_method_info=True,
                  ex_name=f'{work_dir.name}_v3a_attribution')
    experiment = BaseExperiment(argparse.Namespace(**params))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    experiment.method.load_state_dict(state['state_dict'], strict=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    method = experiment.method.to(device).eval()
    states = {branch: empty_state() for branch in BRANCHES}
    with torch.inference_mode():
        for batch in experiment.data.valid_loader:
            if len(batch) != 3:
                raise RuntimeError('V3a attribution requires validation routing labels')
            batch_x, batch_y, packed = batch
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            route_target, valid = decode_packed_routing_target(
                packed.to(device), method.hparams.v3a_route_weight16,
                method.hparams.v3a_route_weight32)
            result = method.model(batch_x, return_aux=True)
            preserve = torch.zeros_like(route_target)
            preserve[:, :, 0] = 1.0
            oracle_probability = torch.where(
                valid.to(device)[:, :, None], route_target, preserve)
            candidates = torch.stack([
                result['direct_rain'], result['motion_rain'],
                result['decay_rain']], dim=2)
            oracle_rain = (oracle_probability.unsqueeze(3) * candidates).sum(2)
            oracle_prediction = rain_to_normalized_dbz(
                oracle_rain, method.hparams.radar_value_scale,
                method.hparams.zr_a, method.hparams.zr_b)
            predictions = {
                'direct': method._to_precipitation(result['direct_prediction']),
                'motion': method._to_precipitation(result['motion_prediction']),
                'decay': method._to_precipitation(result['decay_prediction']),
                'oracle_routed': method._to_precipitation(oracle_prediction),
                'learned_routed': method._to_precipitation(result['prediction']),
            }
            update_state(states, predictions, method._to_precipitation(batch_y))
    document = summarize(states)
    output = (Path(options.output_dir).expanduser().resolve()
              if options.output_dir else work_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'v3a_attribution.json').write_text(
        json.dumps(document, indent=2), encoding='utf-8')
    (output / 'v3a_attribution.md').write_text(
        write_markdown(document, checkpoint), encoding='utf-8')
    print(f'V3a attribution written to {output}')


if __name__ == '__main__':
    main()
