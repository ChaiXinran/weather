"""Audit full ConvLSTM loading without modifying a checkpoint."""
import sys
import torch
from types import SimpleNamespace

from openstl.models import DirectPhysicsHybrid_Model
from openstl.utils import load_config


def main(path, config_path=None):
    state = torch.load(path, map_location='cpu').get('state_dict', {})
    recurrent = [k for k in state if k.startswith('model.cell_list.')]
    output = [k for k in state if k.startswith('model.conv_last.')]
    print(f'total_state_tensors={len(state)}')
    print(f'cell_list_tensors={len(recurrent)}')
    print(f'conv_last_tensors={len(output)}')
    if not recurrent or not output:
        raise SystemExit('FAIL: checkpoint is not a complete ConvLSTM')
    print('PASS: complete cell_list + conv_last checkpoint')
    if config_path:
        values = dict(load_config(config_path))
        values.update(in_shape=[10, 1, 66, 70], pre_seq_length=10,
                      aft_seq_length=20, total_length=30)
        model = DirectPhysicsHybrid_Model(SimpleNamespace(**values))
        model.load_direct_checkpoint(path)
        model.freeze_direct()
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        with torch.no_grad():
            result = model(torch.rand(1, 10, 1, 66, 70, device=device),
                           return_aux=True)
        difference = (result['prediction'] - result['direct_prediction']).abs().max()
        print(f'initial_fused_vs_direct_max_abs={difference.item():.9g}')
        print(f'initial_alpha_max_abs={result["blend_alpha"].abs().max().item():.9g}')
        if difference.item() != 0.0:
            raise SystemExit('FAIL: zero-start hybrid does not exactly preserve direct output')
        print('PASS: initial hybrid output exactly equals direct ConvLSTM')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
