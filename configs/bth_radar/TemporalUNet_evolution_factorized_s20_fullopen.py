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
epoch = 10
