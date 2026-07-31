import runpy
globals().update({
    key: value
    for key, value in runpy.run_path(
        r'{{ fileDirname }}/SimVP_gSTA_r2.py').items()
    if not key.startswith('__')
})
del runpy

# R2d: sample-wise, period-aware Soft CSI. The four weights retain almost the
# same total scale as R2c while mildly emphasizing the second hour.
r2_soft_csi_mode = 'sample_period'
r2_segmented_soft_csi_weights = [
    [0.00180, 0.00090],
    [0.00216, 0.00108],
]
r2_empty_event_penalty = 0.1
