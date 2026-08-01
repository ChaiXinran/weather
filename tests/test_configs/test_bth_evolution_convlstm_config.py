from openstl.utils import load_config
from openstl.methods import method_maps


def test_bth_evolution_motion_config_is_motion_only():
    config = load_config('configs/bth_radar/ConvLSTM_evolution_motion.py')
    assert config['method'] == 'EvolutionConvLSTM'
    assert config['num_hidden'] == '64,64,64,64'
    assert config['evolution_encoder_checkpoint'].endswith(
        'val-csi-epoch=02-val_csi_score=0.788316.ckpt')
    assert config['evolution_freeze_encoder_epochs'] == 2
    assert config['evolution_encoder_lr'] < config['evolution_head_lr']
    assert method_maps['evolutionconvlstm'] is method_maps['evolution_convlstm']
