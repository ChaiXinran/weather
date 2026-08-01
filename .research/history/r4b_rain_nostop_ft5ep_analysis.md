# R4-b rain-rate/no-stop five-epoch continuation analysis

## Question and protocol

This run tests the hypothesis that R4-b C failed the motion gate only because
its original five-epoch validation CSI was still increasing.

- Initialization: full epoch-4 C checkpoint (`val_csi_score=0.642496`).
- Experiment: `bth_r4b_rain_nostop_ft5ep_from0642496_seed0`.
- Model/loss/data/seed: unchanged rain-rate/no-stop R4-b protocol.
- Optimizer/scheduler: fresh Adam and fresh five-epoch OneCycle; the exhausted
  optimizer and scheduler state were deliberately not restored.
- The final recorded configuration retained the original two-epoch encoder
  freeze. Although `0` was requested, the repository's config merge treats a
  numeric zero as unset and restored the config value `2`. Thus this experiment
  is an extension under the original freeze protocol, not an all-layers-from-
  epoch-zero fine-tune.

## Training trend

| Continuation epoch | Validation CSI score |
|---:|---:|
| starting C checkpoint | 0.642496 |
| 0 | **0.660787** |
| 1 | 0.639803 |
| 2 | 0.587591 |
| 3 | 0.602436 |
| 4 | 0.614035 |

The first continuation epoch improves the score by 0.018291, so the original C
was not fully exhausted. Restarting the original peak learning rates is too
aggressive for a converged model: score falls sharply around the OneCycle peak
and only partly recovers during annealing. The selected checkpoint is therefore
explicit epoch 0, not `last.ckpt`.

The result supports limited low-learning-rate fine-tuning if further score
optimization is desired. It does not support repeatedly appending five-epoch
OneCycle schedules at the original maximum learning rates.

## Best-checkpoint motion gate

Artifacts:
`work_dirs/bth_r4b_rain_nostop_ft_ep0_motiondiag/saved/motion_diagnostics/`.

| Metric | Continued predicted rain-rate flow | Zero flow | Gate |
|---|---:|---:|---|
| MAE normalized | **0.01474** | 0.01541 | pass |
| CSI16 | 0.61245 | **0.62011** | fail |
| CSI32 | 0.45686 | **0.55187** | fail |
| FSS3@16 | 0.91852 | **0.92381** | fail |
| FSS5@16 | 0.94906 | **0.95536** | fail |
| FSS3@32 | 0.80691 | **0.87515** | fail |
| FSS5@32 | 0.84821 | **0.91384** | fail |
| Centroid@16 (km) | 33.27 | **27.30** | fail |
| Centroid@32 (km) | 42.52 | **32.14** | fail |
| Area ratio@16 | 0.90463 | **1.00273** | fail |
| Area ratio@32 | 0.63833 | **1.00490** | fail |

Relative to pre-continuation C, recursive CSI improves from 0.22023/0.10522 to
0.22428/0.10987 at 16/32 mm/h. However, teacher-forced rain-rate CSI changes
from 0.61553/0.45994 to 0.61245/0.45686, a slight deterioration. The validation
score gain therefore comes from the recursive forecast/loss trade-off, not from
learning a motion field that beats six-minute persistence.

Flow is stable (mean magnitude 0.328 pixels per six minutes, mean batch maximum
1.224 pixels, zero 90%-limit saturation), so the failed gate is not numerical
instability.

## Decision

**Retain epoch 0 as the best scoring R4-b candidate, but do not enter R4-c.**

The additional experiment confirms that more optimization can raise aggregate
forecast CSI. It simultaneously falsifies the narrower claim that more epochs
alone make the motion mechanism correct. The predeclared motion criteria remain
worse than zero flow, particularly CSI/FSS/centroid at 32 mm/h.

If another optimization-only control is desired, use one short run with lower
maximum learning rates and no encoder refreeze. Scientifically, the higher-value
next change remains strong-rain/edge-weighted per-step motion supervision; only
after that change passes this same gate should R4-c add a source head.
