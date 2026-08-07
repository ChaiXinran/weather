"""0.788 direct ConvLSTM plus zero-start motion/growth/decay correction."""

from configs.bth_radar.ConvLSTM_r2d import *

method = 'DirectPhysicsHybrid'
hybrid_direct_checkpoint = (
    'work_dirs/bth_convlstm_r2d_ft3ep_seed0/checkpoints/best_val_csi.ckpt')
hybrid_freeze_direct = True

hybrid_unet_channels = [32, 64, 128, 192]
hybrid_unet_blocks = [1, 1, 2, 2]
hybrid_temporal_mix_scales = [0, 1, 2]
hybrid_convlstm_scales = [3]
hybrid_temporal_kernel = 3
hybrid_convlstm_kernel = 3
hybrid_fpn_channels = 96
hybrid_lead_channels = 8
hybrid_head_channels = 64

# Residual corrections are deliberately bounded. blend_logit starts at zero,
# so the first forward pass is exactly the loaded direct ConvLSTM forecast.
hybrid_alpha_max = 0.08
hybrid_max_residual_displacement = 2.0
hybrid_max_source_rain = 12.0
hybrid_physics_aux_weight = 0.10
hybrid_warmup_physics_weight = 1.0
hybrid_blend_warmup_epochs = 3
hybrid_gate_lr_scale = 0.05
hybrid_direct_anchor_weight = 0.10
hybrid_alpha_regularization = 0.01
hybrid_flow_regularization = 1e-4
hybrid_source_regularization = 1e-5
hybrid_residual_aux_weight = 0.1

lr = 1e-4
batch_size = 4
val_batch_size = 4
epoch = 10
