# R4-c0 oracle-source audit and R4-c1 source-only analysis

## Physical operator correction

The evolution operator now adds a signed source in rain-rate space:

```text
advected_rain = Warp(previous_rain, flow)
evolved_rain = clamp_min(advected_rain + source_rain, 0)
```

`source_rain` is an increment in mm/h per six-minute evolution step. The
operator returns normalized-dBZ predictions plus `advected_rain`,
`source_rain`, and `evolved_rain` diagnostics. With no source, the validated
R4-b path is unchanged. A zero-initialized source head also reproduces the
motion-only prediction. Ten targeted WSL OpenSTL checks passed; pytest itself
is not installed in that environment.

## R4-c0 protocol

- Frozen motion checkpoint: `.640662`.
- Oracle definition:
  `true_rain_t - warp(true_rain_t-1, predicted_flow_t)`.
- Teacher-forced true previous frame at every lead; no model training.
- Full training split: 11,245 windows, used to select source bounds.
- Frozen validation split: 932 windows, used only as an independent mechanism
  check, not to select the bound.

Artifacts:

- `work_dirs/bth_r4c0_oracle_source_train_scale05_0640662/`
- `work_dirs/bth_r4c0_oracle_source_scale05_0640662/`

## Oracle-source distribution

| Region | Train mean abs | Train abs P95 | Train abs P99 | Val abs P99 |
|---|---:|---:|---:|---:|
| All pixels | .130 | .46 | 2.78 | 2.95 |
| Active union >=.1 mm/h | .582 | 2.56 | 8.26 | 9.69 |
| 16-mm/h union | 6.458 | 18.36 | 27.32 | 27.23 |
| 32-mm/h union | 9.607 | 24.89 | 33.35 | 32.30 |
| Existing interior | .820 | 3.60 | 9.80 | 11.72 |
| Previous edge band | .187 | .60 | 3.20 | 3.86 |
| Newborn >=.1 mm/h | .126 | .38 | 1.34 | .98 |
| True growth | 1.102 | 4.80 | 12.90 | 14.00 |
| True decay | .753 | 3.18 | 8.33 | 9.30 |

The signs are physically ordered. In true-growth pixels, 72.6% of oracle
source values are positive; in true-decay pixels, 76.3% are negative. In the
32-mm/h union, 74.5% are positive and mean signed source is +6.23 mm/h. Source
is therefore both necessary and physically interpretable.

The generic active-area P99 (8.26) is too small for the primary 32-mm/h goal.
The training 32-mm/h absolute P99 is 33.35 and positive P99 is 34.41; validation
independently gives 32.30/33.23. The controlled symmetric bound is therefore:

```text
source_rain = 35 mm/h * tanh(raw_source)
```

## R4-c1 implementation

- Encoder and motion head loaded from `.640662` and frozen for all three
  epochs.
- New shallow source head: 3x3 Conv, GroupNorm+SiLU, 3x3 Conv+SiLU, 1x1 Conv,
  20 signed maps.
- Final layer zero initialized.
- Loss: recursive precipitation R2 + weighted oracle-source Huber + light L1
  sparsity + spatial/temporal TV.
- No learned motion gate and no source/motion joint fine-tuning.

## R4-c1 results

The first weighting gave every background pixel weight one:

| Epoch | CSI score | CSI16 0--1 h | CSI32 0--1 h | CSI16 1--2 h | CSI32 1--2 h | Mean abs source |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .640814 | .322073 | .187893 | .112277 | .009285 | .00362 |
| 1 | .640852 | .322084 | .187920 | .112307 | .009270 | .00572 |
| 2 | .641011 | .322138 | .188015 | .112289 | .009285 | .00320 |

Because background dominated, a controlled rerun restricted direct source
supervision to the active union (`rain >= .1` or `|oracle source| >= .1`) while
retaining full-field sparsity:

| Epoch | CSI score | CSI16 0--1 h | CSI32 0--1 h | CSI16 1--2 h | CSI32 1--2 h | Mean abs source |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .640982 | .322139 | .187948 | .112296 | .009299 | .00385 |
| 1 | **.641226** | **.322168** | **.188128** | **.112333** | .009298 | .00562 |
| 2 | .640984 | .322142 | .188017 | .112283 | .009271 | .00338 |

The active source-supervision loss does not decrease (`.03059 -> .03066`) and
the predicted source remains two orders of magnitude below strong-rain oracle
residuals. POD32 and Bias32 remain effectively unchanged. The tiny CSI gain
(.000564 over R4-b) is not a source-mechanism pass.

## Decision

R4-c0 passes: the missing signed residual is large, sign-consistent, stable
between train and validation, and especially important at 16/32 mm/h.

R4-c1 as currently parameterized fails the mechanism gate. Do not enter R4-c2
or unfreeze motion. More identical epochs are not justified. The next isolated
test should remove the recursive forecast-gradient conflict entirely and
pretrain the source head only on active oracle residuals, with explicit
positive/negative and 16/32-mm/h diagnostics. If direct-only loss still cannot
fall, the final history feature is insufficient and the source input needs an
explicit history-change cue or temporal decoder.

Selected diagnostic checkpoint (not promoted as a new baseline):
`work_dirs/bth_r4c1_source_active_3ep_seed0/checkpoints/`
`val-csi-epoch=01-val_csi_score=0.641226.ckpt`.
