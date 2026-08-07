from openstl.utils import load_config


def test_temporal_unet_smoke_config_is_motion_only():
    config = load_config(
        'configs/bth_radar/TemporalUNet_evolution_motion_smoke.py')
    assert config['method'] == 'EvolutionTemporalUNet'
    assert config['temporal_unet_mix_scales'] == [1, 2, 3]
    assert config['temporal_unet_fpn_channels'] == 96
    assert not config['evolution_use_source']
    assert config['epoch'] == 1


def test_temporal_unet_formal_config_matches_r4b_transport_controls():
    temporal = load_config(
        'configs/bth_radar/TemporalUNet_evolution_motion.py')
    baseline = load_config('configs/bth_radar/ConvLSTM_evolution_motion.py')
    for key in (
            'evolution_max_displacement', 'evolution_align_corners',
            'evolution_padding_mode', 'evolution_field_space',
            'evolution_stop_gradient', 'evolution_tf_weight',
            'evolution_spatial_weight', 'evolution_temporal_weight'):
        assert temporal[key] == baseline[key]


def test_temporal_unet_bottleneck_convlstm_config_is_motion_only():
    config = load_config(
        'configs/bth_radar/'
        'TemporalUNet_evolution_motion_bottleneck_convlstm.py')
    assert config['temporal_unet_convlstm_scales'] == [3]
    assert config['temporal_unet_convlstm_kernel'] == 3
    assert not config['evolution_use_source']
    assert config['epoch'] == 10


def test_temporal_unet_full_source_config_is_factorized_free_rollout():
    config = load_config(
        'configs/bth_radar/TemporalUNet_evolution_factorized_s20.py')
    assert config['evolution_forecast_steps'] == 20
    assert config['evolution_use_source']
    assert config['evolution_source_parameterization'] == 'factorized_regime'
    assert config['evolution_field_space'] == 'rain_rate'
    assert config['evolution_free_rollout_training']
    assert config['evolution_source_only']
    assert config['evolution_source_capacity_values'] == [4.0, 6.0, 8.0, 10.0]
    assert config['evolution_motion_checkpoint'].endswith(
        'val-csi-epoch=06-val_csi_score=0.439920.ckpt')
