method = 'SimVP'

spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 128
N_S = 2
N_T = 4

lr = 1e-3
batch_size = 8
val_batch_size = 8
drop_path = 0.0
sched = 'onecycle'
epoch = 50

manifest_path = '.research/bth_2025_events.json'
radar_cache_path = 'RADAR_CACHE_UINT8'

precip_thresholds = [0.1, 2.5, 8.0, 16.0, 32.0]
val_precip_thresholds = [16.0, 32.0]
radar_value_scale = 50.0
precip_value_unit = 'mm/h'
precip_clip_range = [0.0, 50.0]
convert_dbz_to_rain = True
zr_a = 200.0
zr_b = 1.6
zr_fit_artifact = '.research/local_zr_v2.json'
zr_selection = 'marshall_palmer_validation_winner'
wet_threshold = 0.1
grid_spacing_km = 10.0
neighborhood_windows = [1, 3, 5]
object_iou_threshold = 0.1
bootstrap_repetitions = 2000
bootstrap_seed = 42
lead_minutes = 6
case_threshold = 32.0
case_count = 3

# R2 changes only the training objective. Validation remains pure MSE plus the
# R1 strong-precipitation metrics and dual-checkpoint protocol.
loss_type = 'precipitation_r2'
r2_thresholds = [16.0, 32.0]
r2_intensity_weights = [2.0, 3.0]
r2_soft_csi_weights = [0.005, 0.001]
r2_soft_csi_temperature = 0.03
r2_huber_beta = 0.05
r2_second_hour_weight = 1.2
