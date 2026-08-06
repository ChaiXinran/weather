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


def test_single_step_source_config_is_an_explicit_rollback():
    config = load_config(
        'configs/bth_radar/ConvLSTM_evolution_source_s1_pixel_weighted.py')
    assert config['evolution_forecast_steps'] == 1
    assert config['evolution_state_loss_mode'] == 'pixel_weighted'
    assert config['evolution_pixel_16_increment'] == 1.0
    assert config['evolution_pixel_32_increment'] == 1.0
    assert config['evolution_pixel_max_weight'] == 3.0


def test_factorized_source_config_is_single_step_source_only():
    config = load_config(
        'configs/bth_radar/ConvLSTM_evolution_factorized_s1.py')
    assert config['evolution_forecast_steps'] == 1
    assert config['evolution_source_parameterization'] == 'factorized_regime'
    assert config['evolution_source_only']
    assert config['evolution_regime_delta'] == 0.5
    assert config['evolution_motion_checkpoint'].endswith(
        'val-csi-epoch=01-val_csi_score=0.640662.ckpt')


def test_factorized_capacity_config_has_four_bins():
    config = load_config(
        'configs/bth_radar/ConvLSTM_evolution_factorized_capacity_s1.py')
    assert config['evolution_source_parameterization'] == 'factorized_regime'
    assert config['evolution_source_capacity_edges'] == [8.0, 16.0, 32.0]
    assert len(config['evolution_source_capacity_values']) == 4
