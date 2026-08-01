# R4-b rain-rate/no-stop full and motion diagnostic

## Scope

- Candidate: C, rain-rate evolution without stop-gradient.
- Experiment: `bth_r4b_rain_nostop_5ep_seed0`.
- Checkpoint: epoch-4 `best_val_csi.ckpt` (`val_csi_score=0.642496`).
- Protocol: frozen 2025 validation split, 4 events / 932 windows, seed 0,
  10 input frames to 20 forecast frames, Marshall--Palmer conversion, and the
  existing full spatial/object/event/bootstrap evaluator.
- Full-evaluation artifacts:
  `work_dirs/bth_r4b_rain_nostop_best_valdiag/saved/precipitation_evaluation/`.
- Motion artifacts:
  `work_dirs/bth_r4b_rain_nostop_motiondiag/saved/motion_diagnostics/`.

## Full-rollout result

| Metric | A: normalized/no-stop | C: rain/no-stop | Change |
|---|---:|---:|---:|
| MAE (mm/h) | 0.3224 | **0.3217** | -0.0007 |
| RMSE (mm/h) | 1.9537 | **1.9205** | -0.0332 |
| Intensity ratio | 0.7712 | **0.7910** | +0.0198 |
| CSI16 | 0.2111 | **0.2202** | +0.0091 |
| CSI32 | 0.1022 | **0.1052** | +0.0030 |
| FSS3@16 | 0.3148 | **0.3194** | +0.0046 |
| FSS3@32 | 0.1446 | **0.1458** | +0.0012 |
| Field centroid@16 (km) | 68.89 | **64.93** | -3.96 |
| Field centroid@32 (km) | 56.73 | **52.14** | -4.59 |
| Object POD@16 | 0.1586 | **0.1642** | +0.0056 |
| Object POD@32 | 0.0910 | **0.0951** | +0.0041 |
| Object FAR@16 | 0.3072 | **0.2929** | -0.0143 |
| Object FAR@32 | 0.3377 | **0.3093** | -0.0284 |

Rain-rate evolution produces a consistent but modest improvement over A. It
does not recover the ConvLSTM 0.788 strong-rain skill: C remains below its
CSI16/32 (0.2050/0.1191 gives mixed CSI behavior), FSS3@16/32
(0.4113/0.2991), and object POD (0.2530/0.2520), especially at 32 mm/h.

The mass diagnostics remain the central failure:

| Threshold | A area / energy | C area / energy |
|---:|---:|---:|
| 16 mm/h | 0.5028 / 0.4176 | **0.4882 / 0.4019** |
| 32 mm/h | 0.1966 / 0.1745 | **0.1843 / 0.1637** |

Although C improves intensity ratio averaged over every pixel, it retains even
less thresholded strong-rain area and energy than A. Its lower centroid error
and FAR therefore still describe a smaller, selectively surviving object set.
They are not sufficient evidence of a correct motion field.

Period metrics confirm that the second-hour extreme-rain failure persists:
CSI32 is 0.1869 in hour one but only 0.00966 in hour two; second-hour Bias32 is
0.0619. C improves erosion on average but does not solve long-lead convective
survival.

## Strict teacher-forced motion gate

The relevant learned-flow mode is `teacher_forced_rain_rate`: each true
previous frame is warped once in rain-rate space by C's predicted incremental
flow. `teacher_forced_zero_flow` uses that same true previous frame without
motion and is the strict six-minute persistence control.

| Metric | Predicted rain-rate flow | Zero flow | Gate |
|---|---:|---:|---|
| MAE normalized | **0.01472** | 0.01541 | pass |
| RMSE normalized | **0.03761** | 0.03933 | pass |
| CSI16 | 0.6155 | **0.6201** | fail |
| CSI32 | 0.4599 | **0.5519** | fail |
| FSS3@16 | 0.9191 | **0.9238** | fail |
| FSS5@16 | 0.9493 | **0.9554** | fail |
| FSS3@32 | 0.8073 | **0.8751** | fail |
| FSS5@32 | 0.8478 | **0.9138** | fail |
| Centroid@16 (km) | 33.37 | **27.30** | fail |
| Centroid@32 (km) | 42.61 | **32.14** | fail |
| Area ratio@16 | 0.9010 | **1.0027** | fail |
| Area ratio@32 | 0.6338 | **1.0049** | fail |

The learned flow reduces background-dominated continuous error but degrades all
predeclared strong-rain motion criteria. The failure is especially large at
32 mm/h. This repeats the original R4-b mechanism result after physical-space
retraining, so operator space was a real error source but not the root cause of
the learned-motion error.

The flow remains numerically stable: mean magnitude is 0.320 pixels per six
minutes, mean batch maximum is 1.204 pixels, and no pixels reach 90% of the
configured displacement limit. The issue is supervision/identifiability, not
flow saturation or numerical divergence.

## Decision

**No-go for R4-c.** C is retained as the best R4-b implementation, but it does
not pass the motion gate and should not receive a source/intensity head yet.
Adding that head now could compensate for, and conceal, an incorrect flow.

Do not continue the completed five-epoch OneCycle run. The next controlled
stage should change motion supervision rather than add epochs:

1. Add strong-rain/event weighting to per-step teacher-forced transport loss.
2. Add rain-weighted spatial-gradient/edge supervision rather than uniform TV
   alone.
3. Calibrate displacement direction and scale against fixed optical-flow or
   centroid-shift diagnostics.
4. Re-run this exact predicted-flow versus zero-flow gate. Promote to R4-c only
   when CSI16/32, FSS16/32, and centroid errors improve together.
