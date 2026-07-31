from openstl.utils import load_config


def test_bth_convlstm_r2d_config_is_self_contained():
    config = load_config('configs/bth_radar/ConvLSTM_r2d.py')

    assert config['method'] == 'ConvLSTM'
    assert config['num_hidden'] == '64,64,64,64'
    assert config['loss_type'] == 'precipitation_r2'
    assert config['r2_soft_csi_mode'] == 'sample_period'
    assert config['r2_segmented_soft_csi_weights'] == [
        [0.00180, 0.00090],
        [0.00216, 0.00108],
    ]
