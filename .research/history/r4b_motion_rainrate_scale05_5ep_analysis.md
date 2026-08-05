# R4-b rain-rate motion-only scale-0.5 five-epoch analysis

## Protocol

- Frozen BTH 2025 validation split: 932 windows from four events.
- Input/target: 10 historical frames to 20 recursive future frames, six minutes
  per lead.
- Model: ConvLSTM history encoder initialized from the 0.788316 checkpoint;
  the original direct image head is excluded and replaced by a motion-only
  flow head plus differentiable warp.
- Evolution field: rain rate.
- Displacement bound: 1 pixel per six-minute step, equivalent to multiplying
  the original 2-pixel-bound flow by 0.5 for identical motion-head weights.
- No learned flow gate, oracle gate supervision, or source/sink head.
- Training: five epochs, seed 0, batch size 8, precipitation R2 loss.
- Experiment: `work_dirs/bth_r4b_motion_rainrate_scale05_5ep_seed0`.
- Metrics: `lightning_logs/version_40/metrics.csv`.

## Five-epoch trend

| Epoch | Val CSI score | CSI16 0--1 h | CSI32 0--1 h | CSI16 1--2 h | CSI32 1--2 h |
|---:|---:|---:|---:|---:|---:|
| 0 | .529999 | .306570 | .147331 | .074607 | .000745 |
| 1 | .529821 | .304654 | .145678 | .075318 | .002086 |
| 2 | .589512 | .315135 | .171642 | .094026 | .004354 |
| 3 | .598675 | .316953 | .172374 | .099546 | .004901 |
| 4 | **.633323** | **.321598** | **.184482** | **.109681** | **.008781** |

The score is flat for the first two epochs and rises consistently from epoch 2
through epoch 4. Epoch 4 is the CSI-best checkpoint. Validation-loss best is
epoch 3 at .009423, so loss-best and precipitation-score-best remain distinct.

## CSI-best comparison with ConvLSTM 0.788316

| Period | Threshold | Model | CSI | POD | FAR | Bias |
|---|---:|---|---:|---:|---:|---:|
| 0--1 h | 16 | ConvLSTM 0.788 | **.334680** | **.481568** | .476818 | .920459 |
| 0--1 h | 16 | R4-b scale .5 | .321598 | .430801 | **.440781** | .770362 |
| 0--1 h | 32 | ConvLSTM 0.788 | **.235034** | **.459963** | .675389 | 1.416969 |
| 0--1 h | 32 | R4-b scale .5 | .184482 | .215016 | **.434954** | .380528 |
| 1--2 h | 16 | ConvLSTM 0.788 | **.109817** | **.221651** | .821253 | 1.240028 |
| 1--2 h | 16 | R4-b scale .5 | .109681 | .145923 | **.693666** | .476353 |
| 1--2 h | 32 | ConvLSTM 0.788 | **.054393** | **.200262** | .930514 | 2.882044 |
| 1--2 h | 32 | R4-b scale .5 | .008781 | .009181 | **.832480** | .054808 |

The R4-b model reduces FAR at every period/threshold, but this is accompanied
by large reductions in POD and Bias. It is therefore conservative rather than
strictly better. The first-hour CSI16 deficit is modest (.01308), and
second-hour CSI16 is essentially tied (-.00014), but 32-mm/h retrieval remains
the decisive failure: first-hour CSI32 is 21.5% lower and second-hour CSI32 is
83.9% lower than ConvLSTM 0.788.

The weighted validation score falls from .788316 to .633323 (-.154993,
-19.7%). The 1-pixel motion limit fixes over-displacement but cannot preserve
or create strong-rain area because motion-only warping has no intensity/source
mechanism. The result should be retained as the clean, conservative R4-b
motion-only ablation, not promoted as a replacement for the ConvLSTM baseline.

## Decision

1. Retain epoch 4 (`val-csi-epoch=04-val_csi_score=0.633323.ckpt`) as the best
   model from this controlled five-epoch run.
2. Do not replace ConvLSTM 0.788316: the motion-only model has lower FAR but
   materially worse 32-mm/h POD, Bias, CSI, and total validation score.
3. The five-epoch upward trend indicates optimization was not fully flat, but
   the remaining gap is dominated by strong-rain survival, not merely
   displacement. More identical training is not yet justified as a mechanism
   fix.

## Five-epoch low-learning-rate continuation

The epoch-4 `.633323` checkpoint was continued for five fresh-scheduler epochs
with motion-head LR `1e-4` and encoder LR `2e-5`. The attempted command-line
encoder-freeze override of zero did not replace the configured value, so the
actual logged protocol retained a two-epoch encoder freeze. This matters for
interpretation: epochs 0--1 update only the motion path; the encoder becomes
trainable from epoch 2.

| Continuation epoch | Val CSI score | CSI16 0--1 h | CSI32 0--1 h | CSI16 1--2 h | CSI32 1--2 h |
|---:|---:|---:|---:|---:|---:|
| 0 | .628492 | .320785 | .183337 | .106953 | .008709 |
| 1 | **.640662** | **.322080** | **.187828** | **.112302** | **.009226** |
| 2 | .619840 | .317199 | .183301 | .104194 | .007574 |
| 3 | .617094 | .317174 | .182669 | .104119 | .006566 |
| 4 | .628359 | .318654 | .185547 | .107563 | .008298 |

The continuation improves the previous best by `.007339` (+1.16%) at epoch 1,
then drops immediately when the encoder is unfrozen. The improvement is real
but small and is attributable to conservative motion-head refinement, not to
continued backbone adaptation.

| Period | Threshold | Metric | Previous .633323 | Continued .640662 | ConvLSTM .788316 |
|---|---:|---|---:|---:|---:|
| 0--1 h | 16 | CSI / POD / FAR / Bias | .3216 / .4308 / .4408 / .7704 | .3221 / .4385 / .4519 / .8001 | .3347 / .4816 / .4768 / .9205 |
| 0--1 h | 32 | CSI / POD / FAR / Bias | .1845 / .2150 / .4350 / .3805 | .1878 / .2226 / .4540 / .4077 | .2350 / .4600 / .6754 / 1.4170 |
| 1--2 h | 16 | CSI / POD / FAR / Bias | .1097 / .1459 / .6937 / .4764 | .1123 / .1538 / .7061 / .5234 | .1098 / .2217 / .8213 / 1.2400 |
| 1--2 h | 32 | CSI / POD / FAR / Bias | .0088 / .0092 / .8325 / .0548 | .0092 / .0098 / .8587 / .0692 | .0544 / .2003 / .9305 / 2.8820 |

The continued checkpoint now slightly exceeds ConvLSTM at CSI16 in hour two
while retaining lower FAR, but it still retrieves almost none of the
second-hour 32-mm/h area. Select continuation epoch 1 and stop identical
training here; further backbone training is counterproductive under this
motion-only objective.

Continuation artifacts:

- `work_dirs/bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0/`
- `lightning_logs/version_41/metrics.csv`
- Selected checkpoint:
  `val-csi-epoch=01-val_csi_score=0.640662.ckpt`
