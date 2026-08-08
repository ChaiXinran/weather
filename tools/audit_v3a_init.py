"""Read-only V3a checkpoint mapping and full-size forward audit."""

import argparse
from types import SimpleNamespace

import torch

from openstl.models import DirectPhysicsRouted_Model
from openstl.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    options = parser.parse_args()
    values = dict(load_config(options.config_file))
    values.update(in_shape=[10, 1, 66, 70], pre_seq_length=10,
                  aft_seq_length=20, total_length=30)
    configs = SimpleNamespace(**values)
    model = DirectPhysicsRouted_Model(configs)
    direct_count = model.load_direct_checkpoint(
        configs.hybrid_direct_checkpoint)
    v2_count, skipped = model.load_v2_correction_checkpoint(
        getattr(configs, 'v3a_init_correction_checkpoint', ''))
    model.freeze_direct()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    with torch.inference_mode():
        result = model(
            torch.rand(1, 10, 1, 66, 70, device=device), return_aux=True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
    print(f'device={device}')
    print(f'direct_tensors={direct_count}')
    print(f'v2_compatible_tensors={v2_count}')
    print(f'v2_skipped_tensors={len(skipped)}')
    print(f'total_parameters={total}')
    print(f'trainable_parameters_before_stage_freeze={trainable}')
    print(f'prediction_shape={tuple(result["prediction"].shape)}')
    print('initial_route_mean=' + ','.join(
        f'{value:.6f}' for value in result['route_probability'].mean(
            dim=(0, 1, 3, 4)).cpu().tolist()))
    print(f'flow_abs_max={result["residual_flow"].abs().max().item():.6g}')
    print(f'decay_mean={result["decay_fraction"].mean().item():.6g}')


if __name__ == '__main__':
    main()
