"""Stable source protocol: low-LR 3-step pure-free regularization."""

from configs.bth_radar.TemporalUNet_evolution_factorized_s20 import *

# Mechanism supervision is always teacher-forced. The rollout branch stays
# pure-free and is kept short/low-weight so it regularizes recursive behavior
# without turning the source head into a generic error-correction branch.
evolution_validation_free_rollout = True
evolution_free_rollout_training = True
evolution_rollout_horizon = 3
evolution_rollout_state_loss_weight = 0.25
evolution_source_lr = 5e-5
epoch = 10
