import runpy
globals().update({
    key: value
    for key, value in runpy.run_path(
        r'{{ fileDirname }}/SimVP_gSTA_r2.py').items()
    if not key.startswith('__')
})
del runpy

# Freeze the selected R2d loss baseline without nested config inheritance.
r2_soft_csi_mode = 'sample_period'
r2_segmented_soft_csi_weights = [
    [0.00180, 0.00090],
    [0.00216, 0.00108],
]
r2_empty_event_penalty = 0.1

# R3 changes only the output mechanism: one true-history encoding directly
# produces all 20 future latent/skip frames.
direct_aft_seq_length = 20
