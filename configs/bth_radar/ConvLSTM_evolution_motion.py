method = 'EvolutionConvLSTM'

# The encoder exactly matches the CSI-score 0.788 ConvLSTM run.
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 0

# R4b: shallow bounded motion head + differentiable autoregressive transport.
evolution_head_channels = 64
evolution_head_groups = 8
evolution_max_displacement = 2.0  # pixels per 6-minute step; calibrate statistically
evolution_align_corners = True
evolution_padding_mode = 'zeros'
evolution_field_space = 'normalized_dbz'
evolution_stop_gradient = False
evolution_tf_weight = 0.5
evolution_spatial_weight = 1e-3
evolution_temporal_weight = 1e-3

# Score-max initialization requested for R4. Only model.cell_list.* is loaded.
# Set to None for the required scratch ablation. The project audit also retains
# the 0.775037 checkpoint as a lower-FAR/lower-bias Pareto initialization.
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
