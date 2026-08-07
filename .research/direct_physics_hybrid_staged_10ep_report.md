# DirectPhysicsHybrid Staged 10-Epoch Validation Report

Date: 2026-08-07

## 1. Executive Summary

Experiment:

```text
bth_direct_physics_hybrid_staged_10ep_seed0
```

Configuration:

```text
configs/bth_radar/DirectPhysicsHybrid_r2d.py
```

Best checkpoint:

```text
work_dirs/bth_direct_physics_hybrid_staged_10ep_seed0/checkpoints/
val-csi-epoch=08-val_csi_score=0.790404.ckpt
```

Frozen Validation reproduced:

```text
val_csi_score = 0.7904037653
val_loss      = 0.0110839745
```

The run used Train and Validation only. `--skip_test_after_train` was enabled;
the Test split was not evaluated.

The architecture is stable and avoids the source-head collapse observed in
the previous EvolutionTemporalUNet experiments. Its gain over the frozen
direct ConvLSTM baseline is real but small:

```text
direct baseline CSI = 0.7880069641
hybrid epoch 8 CSI  = 0.7904037653
absolute gain       = 0.0023968012
relative gain       = 0.304%
```

Validation MSE improves by approximately 2.86%, from `0.01141047` to
`0.01108397`.

## 2. Architecture and Training Protocol

The model consists of:

```text
frozen direct ConvLSTM forecast
        +
Temporal U-Net/FPN condition encoder
        +
bounded residual flow and rain-rate source
        +
per-lead bounded blend gate
```

The direct checkpoint is:

```text
work_dirs/bth_convlstm_r2d_ft3ep_seed0/checkpoints/best_val_csi.ckpt
```

The direct branch is frozen. The correction branch contains 3.7M trainable
parameters, while the frozen direct branch contains 3.7M parameters. Total
model size is 7.5M parameters, approximately 29.9 MB.

The correction is deliberately bounded:

```text
maximum blend alpha             = 0.08
maximum residual displacement   = 2 pixels
maximum rain-rate source        = 12 mm/h
blend warm-up                   = 3 epochs
```

The first three epochs train the physics branch through its auxiliary loss
while the deployable prediction remains exactly the frozen direct forecast.
The blend is enabled from epoch 3 onward.

Training loss:

```text
L = fused precipitation-R2 loss
  + w_aux * physics precipitation-R2 loss
  + 0.10  * direct-anchor loss
  + 1e-4  * residual-flow magnitude
  + 1e-5  * source magnitude
  + 0.01  * blend-alpha magnitude
```

During warm-up, `w_aux=1.0`; after blend activation, `w_aux=0.1`.

## 3. Training Curve

| Epoch | Blend state | Validation CSI | Validation loss |
|---:|---|---:|---:|
| 0 | disabled | 0.788007 | 0.011410 |
| 1 | disabled | 0.788007 | 0.011410 |
| 2 | disabled | 0.788007 | 0.011410 |
| 3 | enabled | 0.788486 | 0.011314 |
| 4 | enabled | 0.789167 | 0.011231 |
| 5 | enabled | 0.789934 | 0.011161 |
| 6 | enabled | 0.790310 | 0.011117 |
| 7 | enabled | 0.790135 | 0.011090 |
| 8 | enabled | **0.790404** | 0.011084 |
| 9 | enabled | 0.790313 | **0.011082** |

The curve is stable: there is no abrupt post-update collapse. Epoch 8 is the
correct CSI-selected checkpoint; epoch 9 is the MSE-selected checkpoint.

## 4. Frozen Validation Metrics

### 4.1 Period Metrics

| Metric | First hour | Second hour |
|---|---:|---:|
| CSI at 16 mm/h | 0.335208 | 0.109633 |
| CSI at 32 mm/h | 0.236154 | 0.054704 |
| POD at 16 mm/h | 0.480812 | 0.211979 |
| POD at 32 mm/h | 0.458670 | 0.186123 |
| FAR at 16 mm/h | 0.474626 | 0.814948 |
| FAR at 32 mm/h | 0.672593 | 0.928096 |
| Area ratio at 16 mm/h | 0.915181 | 1.145514 |
| Area ratio at 32 mm/h | 1.400918 | 2.588487 |
| Intensity ratio | 0.904797 | 0.905287 |

### 4.2 Selected Lead Metrics

| Lead | CSI16 | CSI32 | Bias16 | Bias32 | Intensity ratio |
|---:|---:|---:|---:|---:|---:|
| 6 min | 0.713359 | 0.631435 | 0.944837 | 0.999270 | 0.968118 |
| 30 min | 0.323847 | 0.231770 | 0.882057 | 1.313101 | 0.896860 |
| 60 min | 0.171406 | 0.106865 | 0.959049 | 1.866040 | 0.868986 |
| 90 min | 0.110328 | 0.056624 | 1.111685 | 2.497909 | 0.889357 |
| 120 min | 0.078647 | 0.031977 | 1.363052 | 3.270096 | 0.974082 |

## 5. Comparison with the Direct Baseline

The independent direct baseline was reproduced by disabling the blend:

| Metric | Direct baseline | Hybrid epoch 8 | Change |
|---|---:|---:|---:|
| Aggregate CSI | 0.788007 | 0.790404 | +0.002397 |
| Validation MSE | 0.011410 | 0.011084 | -0.000326 |
| CSI16 first hour | 0.334686 | 0.335208 | +0.000523 |
| CSI16 second hour | 0.109829 | 0.109633 | -0.000196 |
| CSI32 first hour | 0.234803 | 0.236154 | +0.001351 |
| CSI32 second hour | 0.054345 | 0.054704 | +0.000360 |
| 32 mm/h area ratio, second hour | 2.884694 | 2.588487 | improved |
| Intensity ratio, second hour | 0.958811 | 0.905287 | more under-intense |

The hybrid correction mainly improves first-hour 32 mm/h CSI and reduces the
severe second-hour 32 mm/h over-coverage. It does not materially improve
second-hour CSI16, and it increases the overall second-hour intensity deficit.

## 6. Comparison with EvolutionTemporalUNet

| Model | Validation CSI | Stability |
|---|---:|---|
| Temporal U-Net motion-only | 0.439920 | stable but weak |
| Motion plus strict zero source | 0.564899 | stable reference |
| Warm-up source epoch 0 snapshot | 0.602775 | near-zero source, not learned mechanism |
| Warm-up source epoch 1 | 0.406984 | collapsed after source update |
| Scheduled source epoch 1 | 0.411575 | collapsed after source update |
| Frozen direct ConvLSTM baseline | 0.788007 | stable |
| DirectPhysicsHybrid epoch 8 | **0.790404** | stable |

The new architecture succeeds primarily because it treats physics as a small,
bounded residual around a strong direct predictor. It does not replace the
stable direct forecast with a recursively updated source trajectory.

## 7. Evaluation Bug Found and Fixed

The first standalone checkpoint validation returned exactly the direct
baseline score `0.788007`, despite loading the epoch 8 checkpoint. The cause
was that blend activation depended only on `current_epoch`. A fresh standalone
Validation starts at epoch 0, so it incorrectly disabled the learned blend.

The inference rule was corrected to:

```text
disable blend only during training warm-up;
always enable learned blend during validation and deployment.
```

After this correction, the epoch 8 checkpoint reproduced its training-time
score exactly:

```text
checkpoint filename: 0.790404
frozen Validation:    0.7904037653
```

This fix changes evaluation/deployment semantics only; it does not modify the
trained checkpoint weights.

## 8. Assessment

The experiment passes the engineering stability criterion and establishes a
new best Validation CSI in the experiments considered here. It does not yet
establish a large modeling improvement over the direct ConvLSTM baseline.

Main strengths:

1. No second-epoch collapse.
2. Exact preservation of the strong direct baseline during warm-up.
3. Bounded residual correction limits damage from the physics branch.
4. Best checkpoint is reproducible under standalone Validation.

Remaining limitations:

1. Aggregate CSI gain is only 0.0024.
2. FAR remains very high in the second hour, especially at 32 mm/h.
3. The second-hour 32 mm/h area ratio remains over-predicted by 2.59 times.
4. Mean intensity is under-predicted by roughly 9.5% in both hours.
5. A single seed and one validation split are insufficient for a robust claim.

## 9. Recommended Next Steps

1. Keep epoch 8 as the CSI-selected checkpoint and epoch 9 as the MSE-selected
   checkpoint; do not use `last.ckpt` as the primary model.
2. Run frozen diagnostics for both checkpoints and compare object/FSS metrics.
3. Add per-lead logging for blend alpha, residual flow, and source magnitude to
   determine where the small CSI gain originates.
4. Test at least three deterministic seeds before accepting the 0.0024 gain as
   robust.
5. Tune for second-hour false alarms and 32 mm/h area bias, rather than simply
   increasing the maximum blend or source capacity.
