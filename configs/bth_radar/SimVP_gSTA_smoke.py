method = 'SimVP'

spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 128
N_S = 2
N_T = 4

lr = 1e-3
batch_size = 16
val_batch_size = 16
drop_path = 0.0
sched = 'onecycle'
epoch = 10

# Smoke-test protocol: retain complete date coverage but use every tenth
# overlapping 10->20 window to reduce redundant training and evaluation.
sample_stride = 10
radar_cache_path = 'RADAR_CACHE_UINT8'
manifest_path = '.research/bth_2025_events.json'

# Same frozen train-only local Z-R relation as the formal configuration.
precip_thresholds = [0.1, 2.5, 8.0, 16.0, 32.0]
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
