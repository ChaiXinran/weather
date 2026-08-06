# R4-d2 Edge Residual Flow + Factorized Source Report

Date: 2026-08-06

## 1. Executive Summary

R4-d2 directly combines the edge residual-flow mechanism with the factorized
growth/steady/decay source mechanism and trains with a 20-step free rollout.
The selected checkpoint is epoch 4, selected by validation aggregate CSI:

```text
work_dirs/bth_r4d2_edge_factorized_s20_20ep_seed0/checkpoints/
val-csi-epoch=04-val_csi_score=0.593111.ckpt
```

The mechanism improves intensity preservation substantially. The overall
intensity ratio is 1.0204, compared with 0.8151 for R4-b. First-hour
intensity ratio is 0.9889, compared with 0.8915 for R4-b. However, this does
not translate into a clear strong-echo skill improvement: overall CSI16 is
0.2140 versus 0.2184 for R4-b, and CSI32 is 0.0907 versus 0.1057. The main
failure is long-lead strong-echo coverage and false alarms.

The current result is therefore a useful mechanism integration result, but
not yet a successful end-to-end improvement over R4-b.

## 2. Fixed Protocol

| Item | Setting |
|---|---|
| Split | Event-level validation only |
| Samples | 932 |
| Input/output | 10 history frames -> 20 forecast frames |
| Interval | 6 minutes/frame |
| Field | Rain rate |
| Z-R | Frozen `Z=200R^1.6` |
| Thresholds | 0.1, 2.5, 8, 16, 32 mm/h |
| FSS windows | 1x1, 3x3, 5x5 |
| Model | R4-b frozen motion + edge residual flow + factorized source |
| Training | 10 actual epochs; config default is 20, overridden by CLI |
| Seed | 0 |
| Test split | Not accessed |

The evaluator metadata currently reports `source_enabled=false` due to a
legacy hard-coded field. This is a metadata bug only: the checkpoint and
configuration use `evolution_use_source=True` and
`evolution_use_edge_residual_flow=True`.

## 3. Training Checkpoint Selection

| Epoch | Val CSI score | Val loss |
|---:|---:|---:|
| 0 | 0.4839 | 0.01243 |
| 1 | 0.5261 | 0.01221 |
| 2 | 0.4644 | 0.01245 |
| 3 | 0.4463 | **0.01168** |
| 4 | **0.5931** | 0.01222 |
| 5 | 0.4948 | 0.01260 |
| 6 | 0.5102 | 0.01182 |
| 7 | 0.5258 | 0.01206 |
| 8 | 0.5253 | 0.01215 |
| 9 | 0.5348 | 0.01214 |

Epoch 4 is used for the report because it has the best aggregate validation
CSI. Epoch 3 has the lowest validation loss but materially worse CSI and is
not used as the primary checkpoint.

## 4. Overall Forecast Metrics

| Metric | R4-d2 | R4-b baseline |
|---|---:|---:|
| MAE (mm/h) | 0.3645 | 0.3264 |
| RMSE (mm/h) | 1.9142 | 1.9336 |
| Mean error (mm/h) | +0.0093 | -0.0839 |
| Intensity ratio | **1.0204** | 0.8151 |

| Threshold | CSI | POD | FAR | Frequency bias |
|---:|---:|---:|---:|---:|
| 0.1 mm/h | 0.6253 | 0.7600 | 0.2209 | 0.9754 |
| 2.5 mm/h | 0.4351 | 0.7213 | 0.4770 | 1.3793 |
| 8 mm/h | 0.3231 | 0.4657 | 0.4865 | 0.9068 |
| 16 mm/h | 0.2140 | 0.2809 | 0.5271 | 0.5940 |
| 32 mm/h | 0.0907 | 0.0961 | 0.3810 | 0.1553 |

The mechanism corrects the baseline under-intensity, but the correction is
not selective enough across intensity regimes. At 2.5 mm/h the bias is 1.38,
while the model still misses most 16/32 mm/h pixels.

## 5. Period Metrics

| Period | Threshold | CSI | POD | FAR | Bias | Intensity ratio |
|---|---:|---:|---:|---:|---:|---:|
| 0--1 h | 16 | 0.3252 | 0.4330 | 0.4336 | 0.7646 | 0.9889 |
| 0--1 h | 32 | 0.1693 | 0.1865 | 0.3538 | 0.2887 | 0.9889 |
| 1--2 h | 16 | 0.0962 | 0.1245 | 0.7026 | 0.4186 | 1.0525 |
| 1--2 h | 32 | 0.0012 | 0.0012 | 0.9186 | 0.0153 | 1.0525 |

Compared with R4-b, the first-hour CSI16 is slightly higher (0.3252 vs
0.3221), but first-hour CSI32 is lower (0.1693 vs 0.1878). In the second
hour, both strong-echo scores are lower than baseline, especially CSI32.
The near-zero second-hour CSI32 means that intensity preservation alone did
not preserve the spatial location of extreme echoes.

## 6. FSS

Mean FSS over validation sample-lead pairs:

| Threshold | FSS 1x1 | FSS 3x3 | FSS 5x5 | R4-b 1x1 | R4-b 3x3 | R4-b 5x5 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 mm/h | **0.2275** | **0.3352** | **0.3763** | 0.2236 | 0.3233 | 0.3591 |
| 32 mm/h | 0.0780 | 0.1109 | 0.1201 | **0.0994** | **0.1465** | **0.1643** |

At 16 mm/h, d2 improves neighborhood-scale spatial agreement by about
1.7%, 3.7%, and 4.8% for the 1x1, 3x3, and 5x5 windows. At 32 mm/h, all FSS
values decline, confirming that the remaining problem is specifically the
location and persistence of strong echoes rather than only pixel noise.

## 7. Rollout Stability

| Lead range | MAE | RMSE | Intensity ratio |
|---|---:|---:|---:|
| 0--1 h | 0.2808 | 1.6346 | 0.9889 |
| 1--2 h | 0.4482 | 2.1578 | 1.0525 |

Lead intensity ratio rises from near-neutral in the first hour to 1.1199 at
120 minutes. The model therefore avoids the R4-b intensity collapse but
starts to over-preserve or amplify rain at later leads. This is a softer
failure than the previous source explosion, but it still causes poor strong-
echo spatial skill.

## 8. Conclusion

R4-d2 achieved the intended first-order effect of the new mechanism:

- intensity decay over 20 steps was substantially reduced;
- first-hour intensity stayed close to the target distribution;
- 16 mm/h neighborhood FSS improved at all tested windows;
- the edge residual flow did not produce the earlier catastrophic intensity
  explosion.

It did not yet achieve the required end-to-end improvement:

- CSI16 is approximately unchanged overall;
- CSI32 and strong-echo FSS are worse than R4-b;
- second-hour FAR16 is 0.7026;
- second-hour CSI32 is only 0.0012;
- later-lead intensity ratio rises above 1.1.

Decision: retain the edge-flow mechanism, but do not proceed to the full
birth mechanism yet. The next work should constrain long-horizon source
accumulation and improve strong-echo persistence/location before adding a
newborn branch.

## 9. Artifacts

- Training log: `lightning_logs/version_49/metrics.csv`
- Checkpoint: `work_dirs/bth_r4d2_edge_factorized_s20_20ep_seed0/checkpoints/val-csi-epoch=04-val_csi_score=0.593111.ckpt`
- Evaluation summary: `.research/r4d2_edge_factorized_s20_eval/precipitation_evaluation/summary.json`
- Per-lead metrics: `.research/r4d2_edge_factorized_s20_eval/precipitation_evaluation/per_lead_metrics.csv`
- FSS metrics: `.research/r4d2_edge_factorized_s20_eval/precipitation_evaluation/per_window_metrics.csv`
- Baseline report: `.research/baselines/r4b_0640662_val/baseline_report.md`
