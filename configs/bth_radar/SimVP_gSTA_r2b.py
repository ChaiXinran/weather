import runpy
globals().update({
    key: value
    for key, value in runpy.run_path(
        r'{{ fileDirname }}/SimVP_gSTA_r2.py').items()
    if not key.startswith('__')
})
del runpy

# R2b: normalized strong-pixel Weighted Huber without Soft CSI.
r2_intensity_weights = [2.0, 3.0]
r2_soft_csi_weights = [0.0, 0.0]
