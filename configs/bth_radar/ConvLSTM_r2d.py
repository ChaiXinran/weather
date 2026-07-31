method = 'ConvLSTM'

# Model: a deliberately modest classic recurrent baseline.
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 0

# Standard ConvLSTM autoregressive training with scheduled sampling.
reverse_scheduled_sampling = 0
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# Match the R3 optimization protocol.
lr = 1e-3
batch_size = 8
val_batch_size = 8
sched = 'onecycle'
epoch = 50

# Match the R3 data manifest and precipitation evaluation protocol.
manifest_path = '.research/bth_2025_events.json'
radar_cache_path = 'RADAR_CACHE_UINT8'
precip_thresholds = [0.1, 2.5, 8.0, 16.0, 32.0]
val_precip_thresholds = [16.0, 32.0]
radar_value_scale = 50.0
precip_value_unit = 'mm/h'
precip_clip_range = [0.0, 50.0]
convert_dbz_to_rain = True
zr_a = 200.0
zr_b = 1.6
zr_fit_artifact = '.research/local_zr_v2.json'
zr_selection = 'marshall_palmer_validation_winner'
wet_threshold = 0.1
grid_spacing_km = 10.0
neighborhood_windows = [1, 3, 5]
object_iou_threshold = 0.1
bootstrap_repetitions = 2000
bootstrap_seed = 42
lead_minutes = 6
case_threshold = 32.0
case_count = 3

# R2d: normalized weighted Huber plus sample/period Soft CSI. This must be
# evaluated by Base_method.criterion, not ConvLSTM_Model's internal MSE.
loss_type = 'precipitation_r2'
r2_thresholds = [16.0, 32.0]
r2_intensity_weights = [2.0, 3.0]
r2_soft_csi_weights = [0.005, 0.001]
r2_soft_csi_temperature = 0.03
r2_huber_beta = 0.05
r2_second_hour_weight = 1.2
r2_soft_csi_mode = 'sample_period'
r2_segmented_soft_csi_weights = [
    [0.00180, 0.00090],
    [0.00216, 0.00108],
]
r2_empty_event_penalty = 0.1
