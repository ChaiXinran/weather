"""Zero-source free-rollout validation of a motion checkpoint."""

from configs.bth_radar.TemporalUNet_evolution_factorized_s20 import *

evolution_use_source = True
evolution_source_checkpoint = None
evolution_source_only = True
evolution_motion_checkpoint = (
    'work_dirs/bth_temporal_unet_motion_10ep_seed0/checkpoints/'
    'val-csi-epoch=06-val_csi_score=0.439920.ckpt'
)
evolution_validation_free_rollout = True
evolution_free_rollout_training = False

manifest_path = '.research/bth_2025_events.json'
radar_cache_path = 'RADAR_CACHE_UINT8'
epoch = 1
