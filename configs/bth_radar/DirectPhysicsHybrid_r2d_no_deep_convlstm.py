"""Ablation: replace the correction U-Net's deep ConvLSTM with temporal mixing."""

from configs.bth_radar.DirectPhysicsHybrid_r2d import *

hybrid_temporal_mix_scales = [0, 1, 2, 3]
hybrid_convlstm_scales = []

