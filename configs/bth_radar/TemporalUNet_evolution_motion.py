"""Formal phase-one Temporal U-Net motion comparison against R4-b."""

method = 'EvolutionTemporalUNet'

temporal_unet_channels = [32, 64, 128, 192]
temporal_unet_blocks = [2, 2, 2, 2]
temporal_unet_fpn_channels = 96
temporal_unet_mix_scales = [1, 2, 3]
temporal_unet_temporal_kernel = 3

evolution_head_channels = 128
evolution_max_displacement = 2.0
evolution_align_corners = True
evolution_padding_mode = 'zeros'
evolution_field_space = 'normalized_dbz'
evolution_stop_gradient = False
evolution_use_source = False
evolution_tf_weight = 0.5
evolution_spatial_weight = 1e-3
evolution_temporal_weight = 1e-3

# The new backbone starts from scratch; both groups use the same learning rate
# so phase one measures architecture rather than checkpoint transfer policy.
evolution_freeze_encoder_epochs = 0
evolution_encoder_lr = 2e-4
evolution_head_lr = 2e-4

batch_size = 4
val_batch_size = 4
lr = 2e-4
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
