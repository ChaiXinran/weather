method = 'SimVP'

# Radar frames are 66 x 70. N_S=2 keeps encoder/decoder scaling compatible
# with these even dimensions.
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 128
N_S = 2
N_T = 4

# Provisional V0 training defaults.
lr = 1e-3
batch_size = 8
val_batch_size = 8
drop_path = 0.0
sched = 'onecycle'
epoch = 50

# Event-first split: windows are read from this frozen manifest instead of
# being cut by date and subsequently treated as independent observations.
manifest_path = '.research/bth_2025_events.json'
# Lossless uint8 NPY cache relative to data_root. Build once with
# tools/cache_bth_radar.py; remove this option to fall back to PNG decoding.
radar_cache_path = 'RADAR_CACHE_UINT8'

# Frozen evaluation interface. Fitted once from unique timestamp-matched
# Radar--RAIN grid pairs in train events; validation/test never refit Z-R.
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
