"""Parameter-matched Temporal U-Net with bottleneck ConvLSTM fusion."""

from configs.bth_radar.TemporalUNet_evolution_motion import *

# Replace only the deepest weighted temporal fusion. At 192 channels this adds
# about 2.65 M parameters, bringing the complete model close to the 3.7 M
# ConvLSTM reference while preserving all spatial and physical controls.
temporal_unet_convlstm_scales = [3]
temporal_unet_convlstm_kernel = 3

# This is a new backbone and must first pass the motion-only gate from scratch.
evolution_use_source = False
evolution_freeze_encoder_epochs = 0
evolution_encoder_lr = 2e-4
evolution_head_lr = 2e-4
batch_size = 4
val_batch_size = 4
lr = 2e-4
epoch = 10
