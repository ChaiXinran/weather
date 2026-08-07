"""Ablation: replace the correction U-Net's deep ConvLSTM with temporal mixing."""

from configs.bth_radar.DirectPhysicsHybrid_r2d import *

hybrid_temporal_mix_scales = [0, 1, 2, 3]
hybrid_convlstm_scales = []

# Use date-range split (no event manifest available in this environment).
manifest_path = ''

hybrid_motion_alpha_max = 0.5
hybrid_source_alpha_max = 0.25
hybrid_gate_supervision_weight = 0.1
hybrid_gate_temperature = 0.05
hybrid_gate_regularization = 0.002
hybrid_flow_regularization = 1e-4
hybrid_direct_anchor_after_warmup = 0.02
hybrid_blend_warmup_epochs = 0
manifest_path = '.research/bth_2025_events.json'
