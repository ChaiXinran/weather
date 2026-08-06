"""Stable source protocol: teacher-forced warm-up, then free rollout."""

from configs.bth_radar.TemporalUNet_evolution_factorized_s20 import *

# Source is first fitted against true previous frames. Free rollout starts only
# after the source heads have learned the one-step growth/steady/decay mapping.
evolution_source_teacher_forcing_epochs = 3
evolution_validation_free_rollout = True
evolution_free_rollout_training = True
evolution_rollout_loss_weight = 0.25
evolution_source_lr = 5e-5
epoch = 10
