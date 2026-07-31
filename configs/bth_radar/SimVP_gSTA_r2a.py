import runpy
globals().update({
    key: value
    for key, value in runpy.run_path(
        r'{{ fileDirname }}/SimVP_gSTA_r2.py').items()
    if not key.startswith('__')
})
del runpy

# R2a: plain Huber. No strong-pixel weighting and no Soft CSI.
r2_intensity_weights = [0.0, 0.0]
r2_soft_csi_weights = [0.0, 0.0]
