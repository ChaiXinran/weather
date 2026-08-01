# R4-b warp-space x stop-gradient 2x2 training analysis

## Scope

Four seed-0, five-epoch runs use the same 0.788316 ConvLSTM encoder,
motion-only architecture, loss weights, data split, optimizer schedule, and
freeze/unfreeze protocol. Only evolution variable space and cross-step gradient
flow differ.

| ID | Evolution space | Stop-gradient | Experiment |
|---|---|---|---|
| A | normalized dBZ | no | `bth_r4b_motion_pre0788_5ep_seed0` |
| B | normalized dBZ | yes | `bth_r4b_norm_stopgrad_5ep_seed0` |
| C | rain rate | no | `bth_r4b_rain_nostop_5ep_seed0` |
| D | rain rate | yes | `bth_r4b_rain_stopgrad_5ep_seed0` |

These are training/standard-validation results. Full spatial/object/motion
diagnostics have not yet been run for B--D, so this record does not promote a
model by CSI alone.

## Training trends

| Run | Validation CSI score by epoch 0/1/2/3/4 | Best score (epoch) | Best val loss (epoch) |
|---|---|---:|---:|
| A norm/no-stop | .5557 / .5299 / .6012 / .5953 / .6225 | .6225 (4) | .009416 (4) |
| B norm/stop | .5058 / .4976 / .5882 / .5638 / .5846 | .5882 (2) | **.009309 (3)** |
| C rain/no-stop | .5421 / .5786 / .6058 / .6122 / **.6425** | **.6425 (4)** | .009477 (3) |
| D rain/stop | .4750 / .4985 / .5757 / .5769 / .5911 | .5911 (4) | .009533 (3) |

All runs reduce training loss. The encoder is frozen in epochs 0--1 and
unfrozen at epoch 2; every run shows its largest useful recovery around or after
unfreezing. C is the only variant whose validation CSI increases at every epoch.
Its validation loss reaches its minimum at epoch 3 and then slightly worsens,
while CSI improves substantially at epoch 4. This is the expected continuous-
error versus strong-rain-skill trade-off, not simple divergence.

Stop-gradient variants converge to lower-area, more conservative solutions.
In normalized dBZ, stop-gradient obtains the best MSE-style validation loss but
loses precipitation score. Thus blocking rollout gradients alone does not make
the learned flow more useful under the current background-dominated transport
objective.

## Factor effects at epoch 4

| Contrast | Delta validation CSI score |
|---|---:|
| Rain-rate space at no-stop: C - A | **+0.0200** |
| Stop-gradient in normalized dBZ: B - A | -0.0379 |
| Stop-gradient in rain rate: D - C | **-0.0514** |
| Rain-rate space with stop-gradient: D - B | +0.0065 |

Rain-rate evolution has a positive main effect, strongest when cross-step
gradients are retained. Stop-gradient has a negative effect in both variable
spaces and interacts negatively with rain-rate evolution. The combination does
not inherit the full benefit of physical-space warping.

## Best-candidate precipitation behavior

At epoch 4, C (rain/no-stop) versus A (norm/no-stop):

| Metric | A norm/no-stop | C rain/no-stop | Delta |
|---|---:|---:|---:|
| CSI16 0--1 h | .3132 | **.3250** | +.0118 |
| CSI16 1--2 h | .1060 | **.1112** | +.0052 |
| CSI32 0--1 h | .1783 | **.1869** | +.0086 |
| CSI32 1--2 h | **.0125** | .0097 | -.0028 |
| FAR16 0--1 h | .4511 | **.4402** | -.0110 |
| FAR16 1--2 h | .7085 | **.6896** | -.0189 |
| FAR32 0--1 h | .4862 | **.4407** | -.0455 |
| FAR32 1--2 h | .8411 | **.8359** | -.0053 |
| Intensity ratio 0--1 h | .8586 | **.8739** | +.0152 |
| Intensity ratio 1--2 h | .6824 | **.7070** | +.0246 |

C improves first-hour CSI16/32, second-hour CSI16, FAR at all reported
threshold/period combinations, and intensity retention. It does not solve
second-hour CSI32; Bias32 in hour two remains only .0619, so extreme-rain
survival is still the main failure.

## Checkpoint interpretation

- A: best CSI and loss are epoch 4.
- B: use epoch 2 `best_val_csi.ckpt` for precipitation diagnostics; epoch 3 is
  only the val-loss candidate.
- C: use epoch 4 `best_val_csi.ckpt`; it is the 2x2 winner.
- D: use epoch 4 `best_val_csi.ckpt`.

## Decision

1. Retain rain-rate/no-stop (C) as the only candidate for full
   spatial/object/motion diagnosis.
2. Reject stop-gradient as a default under the current loss. It may be revisited
   only together with strong-rain/edge-weighted per-step motion supervision;
   the present experiment does not show benefit.
3. Do not extend any completed five-epoch OneCycle run. C ends with its best CSI,
   but first run full diagnostics; if it passes the motion gate, a longer run
   must restart with a longer scheduler rather than resume at zero learning rate.
4. Do not enter R4-c yet. C must beat its own teacher-forced zero-flow control
   in strong-rain CSI/FSS/centroid measures, not merely beat A's validation
   score.

