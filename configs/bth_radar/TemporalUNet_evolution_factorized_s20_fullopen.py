"""Fully trainable scratch factorized source experiment."""

from configs.bth_radar.TemporalUNet_evolution_factorized_s20_warmup import *

# Train the motion backbone and factorized source jointly from scratch. Keep
# the stable source losses and short free rollout from the warmup protocol.
evolution_motion_checkpoint = None
evolution_encoder_checkpoint = None
evolution_source_only = False
evolution_freeze_encoder_epochs = 0
evolution_freeze_motion_epochs = 0
evolution_encoder_lr = 2e-4
evolution_head_lr = 2e-4
evolution_source_lr = 5e-5
evolution_motion_loss_weight = 1.0
evolution_rollout_horizon_schedule = [3, 3, 6, 6, 10, 10] + [20] * 44
evolution_rollout_last_step_weight = 2.0
evolution_gradient_loss_weight = 0.1
evolution_soft_csi_16_loss_weight = 0.3
evolution_soft_csi_32_loss_weight = 0.6
evolution_pixel_16_increment = 2.0
evolution_pixel_32_increment = 5.0
evolution_pixel_max_weight = 8.0
epoch = 50
