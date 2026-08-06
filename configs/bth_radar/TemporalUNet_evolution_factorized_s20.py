"""Full 20-step factorized source rollout for EvolutionTemporalUNet."""

method = 'EvolutionTemporalUNet'

temporal_unet_channels = [32, 64, 128, 192]
temporal_unet_blocks = [2, 2, 2, 2]
temporal_unet_fpn_channels = 96
temporal_unet_mix_scales = [1, 2, 3]
temporal_unet_temporal_kernel = 3
temporal_unet_source_channels = 32
temporal_unet_source_hidden_channels = 64

evolution_head_channels = 128
evolution_max_displacement = 2.0
evolution_forecast_steps = 20
evolution_align_corners = True
evolution_padding_mode = 'zeros'
evolution_field_space = 'rain_rate'
evolution_stop_gradient = False
evolution_use_flow_gate = False

evolution_use_source = True
evolution_source_parameterization = 'factorized_regime'
evolution_source_max_rain = 35.0
evolution_source_capacity_edges = [8.0, 16.0, 32.0]
evolution_source_capacity_values = [4.0, 6.0, 8.0, 10.0]
evolution_factorized_mask_advected_inference = True
evolution_source_active_threshold = 0.1
evolution_regime_delta = 0.5
evolution_effective_source_huber_beta = 0.05
evolution_state_normalized_huber_beta = 0.02
evolution_effective_loss_weight = 1.0
evolution_state_loss_weight = 1.0
evolution_steady_loss_weight = 0.25
evolution_guard_loss_weight = 0.5
evolution_regime_loss_weight = 0.05
evolution_pixel_16_increment = 1.0
evolution_pixel_32_increment = 1.0
evolution_pixel_max_weight = 3.0

evolution_free_rollout_training = True
evolution_rollout_horizon = 3
evolution_rollout_state_loss_weight = 0.5
evolution_soft_csi_16_loss_weight = 0.05
evolution_soft_csi_32_loss_weight = 0.10
evolution_area_loss_weight = 0.02
evolution_budget_loss_weight = 0.05
evolution_soft_csi_temperature = 2.0
evolution_source_only = True
evolution_source_lr = 1e-5
evolution_motion_checkpoint = (
    'work_dirs/bth_temporal_unet_motion_10ep_seed0/checkpoints/'
    'val-csi-epoch=06-val_csi_score=0.439920.ckpt'
)
evolution_encoder_checkpoint = None
evolution_source_checkpoint = None
evolution_freeze_encoder_epochs = 999
evolution_freeze_motion_epochs = 999

bth_mechanism_checkpoint_monitor = 'val_source_gain_vs_zero'
bth_mechanism_checkpoint_mode = 'max'

batch_size = 4
val_batch_size = 4
lr = 2e-4
sched = 'onecycle'
epoch = 10

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

loss_type = 'mse'
