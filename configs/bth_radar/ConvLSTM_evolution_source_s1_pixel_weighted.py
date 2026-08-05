"""Single-step rollback smoke: bounded source with pixel-weighted state loss."""

method = 'EvolutionConvLSTM'
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 0

# Keep the 20-flow R4-b head checkpoint-compatible, but execute only lead 1.
evolution_head_channels = 64
evolution_head_groups = 8
evolution_max_displacement = 1.0
evolution_forecast_steps = 1
evolution_align_corners = True
evolution_padding_mode = 'zeros'
evolution_field_space = 'rain_rate'
evolution_use_flow_gate = False

evolution_use_source = True
evolution_source_parameterization = 'bounded_state'
evolution_source_max_rain = 35.0
evolution_source_lead_dim = 8
evolution_source_only = True
evolution_source_lr = 2e-4
evolution_source_active_threshold = 0.1
evolution_source_sign_threshold = 0.1
evolution_state_huber_beta = 1.0

# Nested masks yield per-pixel weights 1 / 2 / 3 for active / 16 / 32.
evolution_state_loss_mode = 'pixel_weighted'
evolution_pixel_16_increment = 1.0
evolution_pixel_32_increment = 1.0
evolution_pixel_max_weight = 3.0

evolution_encoder_checkpoint = None
evolution_motion_checkpoint = (
    'work_dirs/bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0/'
    'checkpoints/val-csi-epoch=01-val_csi_score=0.640662.ckpt'
)
evolution_freeze_encoder_epochs = 999
evolution_freeze_motion_epochs = 999

batch_size = 8
val_batch_size = 8
lr = 2e-4
sched = 'onecycle'
epoch = 3

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
wet_threshold = 0.1
grid_spacing_km = 10.0
neighborhood_windows = [1, 3, 5]
object_iou_threshold = 0.1
bootstrap_repetitions = 2000
bootstrap_seed = 42
lead_minutes = 6
case_threshold = 32.0
case_count = 3
loss_type = 'mse'
