"""Formal R4-c1 source-only training configuration for BTH radar."""

method = 'EvolutionConvLSTM'

num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 0

# Frozen R4-b transport skeleton.
evolution_head_channels = 64
evolution_head_groups = 8
evolution_max_displacement = 1.0
evolution_align_corners = True
evolution_padding_mode = 'zeros'
evolution_field_space = 'rain_rate'
evolution_use_flow_gate = False
evolution_gate_supervision_weight = 0.0
evolution_gate_supervision_only = False
evolution_tf_weight = 0.5
evolution_spatial_weight = 1e-3
evolution_temporal_weight = 1e-3

# R4-c1 signed source in mm/h per six-minute evolution step. The symmetric
# bound is the rounded training-set P99 for |oracle source| in >=32-mm/h areas
# (33.35 mm/h; positive P99 34.41), independently confirmed on validation.
evolution_use_source = True
evolution_source_max_rain = 35.0
evolution_source_lr = 2e-4
evolution_source_supervision_weight = 1.0
evolution_source_sparse_weight = 0.01
evolution_source_tv_weight = 0.001
evolution_source_huber_beta = 0.03
evolution_source_active_threshold = 0.1
evolution_source_supervision_only = False

# Load only the validated encoder and motion head. The new source head remains
# zero-initialized, so epoch zero starts from the R4-b prediction.
evolution_encoder_checkpoint = None
evolution_motion_checkpoint = (
    'work_dirs/bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0/'
    'checkpoints/val-csi-epoch=01-val_csi_score=0.640662.ckpt'
)
evolution_freeze_encoder_epochs = 3
evolution_freeze_motion_epochs = 3
evolution_encoder_lr = 1e-6
evolution_head_lr = 1e-6

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

loss_type = 'precipitation_r2'
r2_thresholds = [16.0, 32.0]
r2_intensity_weights = [2.0, 3.0]
r2_soft_csi_weights = [0.005, 0.001]
r2_soft_csi_temperature = 0.03
r2_huber_beta = 0.05
r2_second_hour_weight = 1.2
r2_soft_csi_mode = 'sample_period'
r2_segmented_soft_csi_weights = [[0.00180, 0.00090], [0.00216, 0.00108]]
r2_empty_event_penalty = 0.1
