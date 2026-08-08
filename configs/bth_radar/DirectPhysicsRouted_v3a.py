"""V3a expert pretraining: object-routed motion and decay candidates."""

from configs.bth_radar.ConvLSTM_r2d import *

method = 'DirectPhysicsRouted'
hybrid_direct_checkpoint = (
    'work_dirs/bth_convlstm_r2d_ft3ep_seed0/checkpoints/best_val_csi.ckpt')

# Server V2 initialization. Set to an empty string only for a scratch ablation.
v3a_init_correction_checkpoint = (
    'work_dirs/bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0/'
    'checkpoints/best_val_csi.ckpt')
routing_cache_path = 'V3A_ROUTING_CACHE'
v3a_stage = 'expert'

# Keep the successful lightweight V2 feature pyramid; the new prior-context
# head, experts, and router change the correction mechanism substantially.
hybrid_unet_channels = [32, 64, 128, 192]
hybrid_unet_blocks = [1, 1, 2, 2]
hybrid_temporal_mix_scales = [0, 1, 2, 3]
hybrid_convlstm_scales = []
hybrid_temporal_kernel = 3
hybrid_convlstm_kernel = 3
hybrid_fpn_channels = 96
hybrid_lead_channels = 8
hybrid_head_channels = 64
hybrid_max_residual_displacement = 2.0

v3a_initial_route_probability = [0.80, 0.15, 0.05]
v3a_router_temperature = 1.0
v3a_route_weight16 = 1.0
v3a_route_weight32 = 1.5
v3a_rain_loss_scale = 50.0
v3a_motion_loss_weight = 1.0
v3a_decay_loss_weight = 1.0
v3a_route_loss_weight = 0.30
v3a_preserve_loss_weight = 0.05
v3a_router_fused_weight = 0.25
v3a_flow_regularization = 1e-4

lr = 1e-4
batch_size = 4
val_batch_size = 4
epoch = 3
