"""Formal R4-b motion-only configuration for BTH radar.

This keeps rain-rate evolution and applies the selected global 0.5 flow
calibration by reducing the per-step displacement bound from 2.0 to 1.0 pixel.
No learned flow gate or source/sink head is used.
"""

method = 'EvolutionConvLSTM'

# History encoder: structurally identical to the CSI 0.788 ConvLSTM model.
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 0

# R4-b: history-only motion head followed by autoregressive rain-rate warping.
evolution_head_channels = 64
evolution_head_groups = 8
evolution_max_displacement = 1.0  # 0.5 x the validated 2-pixel raw-flow bound
evolution_align_corners = True
evolution_padding_mode = 'zeros'
evolution_field_space = 'rain_rate'

# Keep the simplified motion-only path explicit. Gate/oracle/source mechanisms
# are deliberately excluded from this formal configuration.
evolution_use_flow_gate = False
evolution_gate_supervision_weight = 0.0
evolution_gate_supervision_only = False

# Motion training losses retained from the controlled R4-b experiments.
evolution_tf_weight = 0.5
evolution_spatial_weight = 1e-3
evolution_temporal_weight = 1e-3

# Fresh R4-b training starts from the reproduced CSI 0.788 history encoder.
# For continuation/evaluation of an existing R4-b checkpoint, pass
# --init_from_ckpt or --ckpt_path on the command line instead.
evolution_encoder_checkpoint = (
    'work_dirs/bth_convlstm_r2d_ft3ep_seed0/checkpoints/'
    'val-csi-epoch=02-val_csi_score=0.788316.ckpt'
)
evolution_freeze_encoder_epochs = 2
evolution_encoder_lr = 1e-4
evolution_head_lr = 5e-4

batch_size = 8
val_batch_size = 8
lr = 5e-4
sched = 'onecycle'
epoch = 50

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

# Existing R2 precipitation objective; this changes only training supervision,
# not the motion-only prediction architecture.
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
