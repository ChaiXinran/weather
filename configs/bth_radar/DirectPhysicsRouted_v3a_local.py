"""Local-WSL V3a stage A using the available no-deep-ConvLSTM checkpoint."""

from configs.bth_radar.DirectPhysicsRouted_v3a import *

v3a_init_correction_checkpoint = (
    'work_dirs/bth_direct_physics_hybrid_no_deep_convlstm_10ep_seed0/'
    'checkpoints/best_val_csi.ckpt')
