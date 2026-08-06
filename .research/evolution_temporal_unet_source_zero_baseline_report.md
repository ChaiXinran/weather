# Evolution Temporal U-Net Source Baseline Report

Date: 2026-08-06

## 1. Executive Summary

This report resolves the source-head checkpoint ambiguity and compares the
source experiments using validation only. The earlier best score was:

```text
CSI = 0.602775
```

The checkpoint is:

```text
work_dirs/bth_temporal_unet_factorized_s20_warmup_seed0/checkpoints/
val-csi-epoch=00-val_csi_score=0.602775.ckpt
```

However, this is the first validation epoch of the warm-up run. It is not a
source-head model that has demonstrated stable learned source dynamics. The
source-head is initialized close to zero, and the score is therefore mainly a
motion-plus-near-zero-source score.

The strict zero-source validation used the motion checkpoint directly and
produced:

```text
val_csi_score = 0.5648994897
val_loss      = 0.0175145343
```

The scheduled source run produced exactly the same epoch-0 score:

```text
epoch 0: CSI = 0.564899
epoch 1: CSI = 0.411575
```

This confirms that the source-head update, rather than the motion checkpoint,
causes the sharp degradation after the first optimization epoch.

## 2. Experiments

| Experiment | Initialization | Source training | Best/first validation CSI | Interpretation |
|---|---|---|---:|---|
| Motion-only | Temporal U-Net from scratch | No source | 0.439920 | Motion architecture baseline; below ConvLSTM reference |
| Full source, original | Motion checkpoint | Source-only, LR 2e-4, free rollout | 0.564887 at epoch 0 | Initial near-zero source state |
| Full source, warm-up | Motion checkpoint | 3 epochs teacher-forced, LR 5e-5 | 0.602775 at epoch 0; 0.406984 at epoch 1 | Initial score is not stable source learning |
| Scheduled source | Motion checkpoint | Scheduled sampling, LR 1e-5 | 0.564899 at epoch 0; 0.411575 at epoch 1 | Confirms the same post-update failure |
| Strict zero-source validation | Motion checkpoint | No optimizer/update; 20-step free rollout | 0.564899 | Reference for motion plus operator with zero source |

All reported metrics are from the validation split. The Test split was not
used.

## 3. Located Checkpoints

The earlier 0.60 checkpoint is:

```text
work_dirs/bth_temporal_unet_factorized_s20_warmup_seed0/checkpoints/
val-csi-epoch=00-val_csi_score=0.602775.ckpt
```

Related checkpoints:

```text
work_dirs/bth_temporal_unet_factorized_s20_warmup_seed0/checkpoints/
val-csi-epoch=01-val_csi_score=0.406984.ckpt

work_dirs/bth_temporal_unet_factorized_s20_scheduled_lr1e5_seed0/checkpoints/
val-csi-epoch=00-val_csi_score=0.564899.ckpt

work_dirs/bth_temporal_unet_factorized_s20_scheduled_lr1e5_seed0/checkpoints/
val-csi-epoch=01-val_csi_score=0.411575.ckpt
```

The strict zero-source run does not produce a useful trained checkpoint; its
reference is the motion checkpoint:

```text
work_dirs/bth_temporal_unet_motion_10ep_seed0/checkpoints/
val-csi-epoch=06-val_csi_score=0.439920.ckpt
```

## 4. Zero-Source Validation Metrics

The strict zero-source run gave:

| Metric | First hour | Second hour |
|---|---:|---:|
| CSI at 16 mm/h | 0.273390 | 0.085447 |
| CSI at 32 mm/h | 0.170629 | 0.017717 |
| POD at 16 mm/h | 0.431980 | 0.145673 |
| POD at 32 mm/h | 0.215150 | 0.019921 |
| Area ratio at 16 mm/h | 1.012069 | 0.850520 |
| Area ratio at 32 mm/h | 0.476073 | 0.144347 |
| Intensity ratio | 1.121361 | 1.165591 |

Lead-time CSI at 32 mm/h falls from:

```text
6 min:   0.569483
30 min:  0.111853
60 min:  0.042652
120 min: 0.004312
```

The source outputs were effectively zero:

```text
birth_source_abs = 0
clear_source_abs = 0
edge_source_abs  = 0
growth_source_scale_ratio = 0
decay_source_scale_ratio  = 0
```

Therefore the zero-source model is not suffering from source explosion. Its
main limitation is long-lead motion and strong-rain disappearance.

## 5. Why the 0.602775 Checkpoint Is Misleading

The source-head is initialized with a steady-regime bias and nearly zero
growth/decay magnitudes. At epoch 0, the model is close to:

```text
motion checkpoint + EvolutionOperator + near-zero source
```

After the first source update, the predicted source is applied recursively for
20 forecast steps. A source error at step `t` changes the input at step `t+1`,
which changes the next source prediction. This feedback can amplify a small
single-step source error.

The observed sequence is:

```text
warm-up epoch 0: 0.602775
warm-up epoch 1: 0.406984

scheduled epoch 0: 0.564899
scheduled epoch 1: 0.411575
```

The repeated pattern across two protocols shows that the issue is not solved
by teacher-forcing scheduling or by reducing the source LR from `2e-4` to
`1e-5` alone.

## 6. Current Evaluation Standard

The current validation objective is ordinary MSE:

```text
val_loss = MSE(prediction, target)
```

The main model-selection metric is:

```text
val_csi_score
```

The CSI score is calculated over validation precipitation thresholds and lead
periods, with the reported score aggregating the configured BTH validation
criteria. For diagnosis, the following metrics should be reported together:

```text
CSI at 16 and 32 mm/h, first and second hour
POD and FAR at 16 and 32 mm/h
lead-wise CSI at 6, 30, 60, 90, 120 minutes
area ratio and intensity ratio
val_loss
source absolute magnitude
source growth/decay scale ratio
source saturation fraction
evolved-above-rmax fraction
```

CSI should be the primary checkpoint-selection criterion for this task. MSE
and mechanism loss are secondary diagnostics because they do not directly
measure strong-rain event skill under free rollout.

## 7. Findings and Decision

1. The earlier 0.602775 checkpoint has been located, but it should be treated
   as an initial-state snapshot, not as evidence that the learned source
   mechanism is successful.
2. The strict zero-source reference is CSI `0.564899`.
3. Training source parameters causes a repeatable collapse to approximately
   `0.41` after one epoch.
4. The motion/operator baseline is already weak at long lead times for the
   32 mm/h threshold, but source training currently makes it worse.
5. More epochs of the current source protocol should not be run or used for
   comparison.

## 8. Recommended Next Experiment

Use the zero-source model as the reference and train only a tightly bounded
source correction:

```text
source_lr = 1e-6
one epoch only
small source magnitude regularization
free-rollout validation after every epoch
early stop if CSI drops by more than 0.02 from 0.564899
```

Before changing the architecture, log source magnitude and saturation by lead
time. If a small source update still lowers CSI, the next investigation should
target the source target construction, active mask, and factorized operator
calibration rather than the Temporal U-Net backbone.
