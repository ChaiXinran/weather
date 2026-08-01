# R4-b pretrained motion-only 5-epoch analysis

## Scope and naming

This experiment is **R4-b**, not R4-a. R4-a is the proposed fixed optical-flow
operator check; the completed run learns 20 incremental flow fields from a
pretrained ConvLSTM history encoder and evolves the last observed Radar field
without a source/sink term.

- Experiment: `bth_r4b_motion_pre0788_5ep_seed0`
- Encoder initialization: ConvLSTM `val_csi_score=0.788316`; only
  `model.cell_list.*` was loaded
- Training: 5 epochs, seed 0; encoder frozen for epochs 0--1 and jointly tuned
  from epoch 2
- Selected checkpoint: epoch 4, both best validation loss (`0.009416`) and best
  validation score (`0.622479`)
- Evaluation: frozen 2025 validation split, 4 events / 932 windows, frozen
  Marshall--Palmer conversion, full spatial/object evaluator
- Evaluation artifacts:
  `work_dirs/bth_r4b_motion_pre0788_best_valdiag/saved/precipitation_evaluation/`

## Training behavior

Validation loss decreased monotonically from `0.009997` to `0.009416` and
training loss from `0.048212` to `0.038809`. Validation CSI score initially fell
from `0.555650` to `0.529903` while the encoder was frozen, then recovered after
unfreezing to `0.601160` and finally `0.622479`. The last validation-loss gain
was only `7.7e-6`, so the five-epoch run is sufficient for a first mechanism
decision; extending this completed OneCycle schedule is not justified.

## Overall and period skill

| Model | MAE | RMSE | Intensity ratio | CSI16 | CSI32 |
|---|---:|---:|---:|---:|---:|
| R4-b motion-only | **0.3224** | **1.9537** | 0.7712 | **0.2111** | 0.1022 |
| ConvLSTM 0.788 | 0.4096 | 2.8028 | 0.9326 | 0.2050 | 0.1191 |
| ConvLSTM 0.775 | 0.3892 | 2.6793 | 0.8467 | 0.2018 | 0.1216 |
| SimVP R3 | 0.4142 | 2.2798 | 1.1023 | 0.2145 | **0.1457** |

R4-b has the lowest continuous error and slightly exceeds both ConvLSTM
checkpoints at overall CSI16. This is not a complete heavy-rain improvement:
CSI32 is lower, and the intensity ratio shows systematic decay.

| Metric | R4-b 0--1 h | R4-b 1--2 h | ConvLSTM 0.788 0--1 h | ConvLSTM 0.788 1--2 h |
|---|---:|---:|---:|---:|
| CSI16 | 0.3132 | 0.1060 | 0.3347 | 0.1098 |
| CSI32 | 0.1783 | 0.0125 | 0.2350 | 0.0544 |
| FAR16 | 0.4511 | 0.7085 | 0.4768 | 0.8213 |
| FAR32 | 0.4862 | 0.8411 | 0.6754 | 0.9305 |
| Bias16 | 0.7683 | 0.4899 | 0.9205 | 1.2400 |
| Bias32 | 0.4174 | 0.0842 | 1.4170 | 2.8820 |

The lower FAR is partly genuine removal of displaced false objects, but cannot
be interpreted alone: the very low Bias32, especially in hour two, means the
model often predicts no 32 mm/h object at all.

## Spatial and object diagnostics

| Model | Thr. | FSS3 | FSS5 | Field centroid km | Matched IoU | Object POD | Object FAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4-b | 16 | 0.3148 | 0.3512 | **68.89** | 0.4172 | 0.1586 | **0.3072** |
| ConvLSTM 0.788 | 16 | **0.4113** | **0.4778** | 79.43 | 0.4135 | **0.2530** | 0.5457 |
| ConvLSTM 0.775 | 16 | 0.4022 | 0.4683 | 75.94 | **0.4210** | 0.2451 | 0.5274 |
| R4-b | 32 | 0.1446 | 0.1630 | **56.73** | **0.4973** | 0.0910 | **0.3377** |
| ConvLSTM 0.788 | 32 | 0.2991 | 0.3536 | 87.90 | 0.4205 | **0.2520** | 0.6814 |
| ConvLSTM 0.775 | 32 | **0.3028** | **0.3607** | 84.05 | 0.4297 | 0.2450 | 0.6705 |

The explicit transport branch provides strong evidence of better localization:
relative to ConvLSTM 0.788, field-centroid error falls by 10.55 km at 16 mm/h
and 31.18 km at 32 mm/h; matched IoU improves, especially at 32 mm/h; and
object FAR is much lower. However, FSS and object POD fall sharply. The model
matches the few surviving objects well but loses too many true objects.

This is confirmed by spatial mass diagnostics:

| Threshold | Area ratio | Energy ratio |
|---:|---:|---:|
| 16 mm/h | 0.5028 | 0.4176 |
| 32 mm/h | 0.1966 | 0.1745 |

Repeated bilinear transport plus the absence of source/intensity correction
causes severe strong-rain area and energy collapse. The prediction range remains
bounded in `[0,1]`, so clipping overflow is not responsible.

## Statistical and protocol interpretation

Against persistence, event bootstrap gives positive CSI deltas at 16 mm/h
(`+0.0842`, 95% CI `[+0.0375,+0.1304]`) and 32 mm/h (`+0.0293`, 95% CI
`[+0.0044,+0.0543]`). These intervals do **not** establish improvement over
ConvLSTM because the current evaluator bootstraps model versus persistence, not
R4-b versus ConvLSTM. Only four validation events and one seed are available.

## Dedicated motion diagnostics

The follow-up diagnostic is stored under
`work_dirs/bth_r4b_motion_pre0788_motiondiag/saved/motion_diagnostics/` and
compares four modes using the same predicted flows:

- recursive: normal 20-step rollout;
- cumulative single warp: sum incremental flows and sample the initial field
  once per lead;
- teacher forced: warp each true previous frame with the predicted flow;
- teacher-forced zero-flow: use each true previous frame unchanged, the strict
  six-minute persistence control.

The flow is numerically stable: mean magnitude is 0.341 pixels per six minutes,
mean batch maximum is 1.101 pixels, no pixels reach 90% of the +/-2-pixel
component limit, and spatial/temporal variation is small. It has a persistent
mean rightward component of about 0.28--0.32 pixels per step.

Recursive transport outperforms the cumulative-sum single-warp approximation
(MAE 0.03946 vs 0.04567; CSI16 0.2111 vs 0.1726). Therefore repeated bilinear
resampling is not the main cause of the observed forecast failure; naive flow
summation also ignores proper deformation composition.

The strict teacher-forced control changes the mechanism conclusion:

| Mode | MAE norm. | RMSE norm. | CSI16 | CSI32 | Centroid16 km | Centroid32 km |
|---|---:|---:|---:|---:|---:|---:|
| Predicted flow | **0.01356** | **0.03330** | 0.5629 | 0.3686 | 39.88 | 48.38 |
| Zero flow | 0.01541 | 0.03933 | **0.6201** | **0.5519** | **27.30** | **32.14** |

| Mode | FSS3@16 | FSS5@16 | FSS3@32 | FSS5@32 |
|---|---:|---:|---:|---:|---:|
| Predicted flow | 0.8929 | 0.9267 | 0.7257 | 0.7681 |
| Zero flow | **0.9238** | **0.9554** | **0.8751** | **0.9138** |

Predicted flow reduces background-dominated continuous error, but it is worse
than zero flow for strong-rain CSI, FSS, centroid location, and Bias. Thus the
full-rollout centroid/FAR gains cannot yet be attributed to correct motion;
they are substantially confounded by erosion and selective survival of fewer
objects.

## Verdict

**Motion gate failed / no-go for R4-c promotion.**

The normal rollout has lower centroid error and object FAR than ConvLSTM, but
the teacher-forced zero-flow control shows that this is not sufficient evidence
of a correct motion field. The learned flow worsens strong-rain localization
relative to six-minute persistence while reducing continuous error. R4-b also
fails the full promotion conditions because FSS and object POD deteriorate,
first-hour CSI drops by more than 0.005 relative to ConvLSTM 0.788, and
second-hour CSI32 and strong-rain mass collapse.

Do not add a source head yet without first establishing that the flow itself is
reasonable; otherwise a source branch could mask a defective motion field.
Before R4-c:

1. Do not add the source head yet; it could hide the motion error.
2. Rebalance the teacher-forced objective away from background-dominated
   Smooth-L1 toward event/gradient/object-weighted transport supervision.
3. Calibrate displacement scale from training-set optical flow or centroid
   statistics and compare against a fixed-flow R4-a baseline.
4. Re-run the zero-flow teacher-forced control; R4-b passes only when predicted
   flow improves strong-rain CSI/FSS/centroid metrics rather than MAE alone.
