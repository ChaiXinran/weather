# R4-a fixed-flow and R4-b2 motion-gate analysis

## R4-a: history-only fixed optical flow

Farneback flow uses only observed history. `last` uses the final observed frame
pair; `median3` takes the pixelwise median of the final three historical
frame-pair flows. All modes use teacher-forced rain-rate warping on the frozen
932-window validation set.

| Mode | MAE normalized | CSI16 | CSI32 |
|---|---:|---:|---:|
| Zero flow | 0.01541 | 0.62011 | **0.55187** |
| Farneback last | 0.01482 | 0.65796 | 0.53404 |
| Farneback median3 | 0.01446 | **0.66446** | 0.54408 |
| Learned x0.25 | 0.01419 | 0.64816 | **0.57066** |
| Learned x0.5 | **0.01392** | 0.65879 | 0.56043 |
| Learned x1 | 0.01474 | 0.61245 | 0.45686 |

R4-a establishes partial motion identifiability at six minutes / 10 km:
history-only fixed flow clearly improves MAE and CSI16 but not aggregate CSI32.
The scaled learned flow has the best combined continuous/extreme-rain result,
so the learned direction signal is competitive with traditional flow after
calibration.

Persistent-object endpoint error confirms that both fixed and learned flows
help moving objects and harm near-static objects. For 16-mm/h objects,
Farneback-median3 reduces endpoint error from .641 to .365 pixels for .5--1
pixel motion and from 1.103 to .746 for >1-pixel motion, while increasing
near-static error from .041 to .293. The same pattern holds at 32 mm/h. A
spatial motion-necessity gate is therefore required; a lead scalar alone cannot
separate objects within the same frame.

Artifacts:
`work_dirs/bth_r4a_fixed_flow_diag/saved/fixed_flow_diagnostics/`.

## R4-b2a: gate trained only by the existing forecast objective

A spatial per-lead gate was added as `v_final = sigmoid(g) * v_raw`, initialized
at .5. The 0.660787 encoder/raw-motion model was loaded and frozen for all five
epochs, leaving only the gate trainable.

Validation CSI stayed nearly flat (.661329, .660905, .660932, .659389,
.659630), but mean gate rose from .642 to .882 and then about .936. On the
epoch-0 CSI-best checkpoint, strong-rain object gates are already .973--.988
for every motion bin. The gate therefore collapses toward full flow and does
not learn motion necessity. Existing forecast/transport losses still reward
background-dominated transport and reproduce the original failure.

## R4-b2b: strong-rain oracle-scale gate supervision

Training-only targets select the lowest-error teacher-forced scale from
`[0,.25,.5,.75,1]` at each active pixel. Supervision is restricted to the
union of previous/next >=8-mm/h rain and weighted further at 16/32 mm/h.
Inference remains history-only. Encoder and raw motion remain frozen.

A first mixed-objective run with weight .1 was stopped after two epochs: gate
mean rose .627 -> .826 while the oracle target stayed .560, proving the
forecast gradient still dominated. A clean three-epoch oracle-only pretraining
then behaved correctly:

| Epoch | Gate loss | Gate mean | Target mean | Val CSI |
|---:|---:|---:|---:|---:|
| 0 | .35869 | .54890 | .55996 | .60121 |
| 1 | .34841 | .58220 | .55990 | .60181 |
| 2 | **.34569** | .56856 | .55961 | .60156 |

Epoch 2 / `last.ckpt` is selected by mechanism loss. Its teacher-forced scale
audit is:

| Residual alpha applied to gated flow | MAE | CSI16 | CSI32 |
|---:|---:|---:|---:|
| 0 | .01541 | .62011 | .55187 |
| .25 | .01429 | .64299 | .56956 |
| .5 | .01380 | .65957 | **.57508** |
| .75 | **.01362** | **.66456** | .56914 |
| 1 | .01368 | .65881 | .54884 |

The deployed gated flow (`alpha=1`) now directly beats zero on MAE and CSI16.
CSI32 misses zero by only .00303, a large recovery from the ungated .45686.
Residual .5--.75 remains optimal, so calibration is improved but incomplete.

The gate also learns a real motion-state ordering:

| Threshold | Near-static gate | Subpixel gate | Moving gate | Fast/difficult gate |
|---:|---:|---:|---:|---:|
| 16 | .572 | .638 | .712 | .723 |
| 32 | .579 | .655 | .675 | .651 |

Moving-object endpoint error improves substantially over zero, but near-static
objects are still over-moved. The final ConvLSTM hidden state contains some
motion-necessity information but does not separate the states sharply enough.

Artifacts:

- `work_dirs/bth_r4b2_spatial_gate_only_5ep_seed0/`
- `work_dirs/bth_r4b2_oracle_gate_only_5ep_seed0/` (stopped mismatch run)
- `work_dirs/bth_r4b2_oracle_gate_pretrain_3ep_seed0/`
- `work_dirs/bth_r4b2_oracle_gate_ep2_diag/saved/flow_scale_diagnostics/`

## Detailed evaluation protocol and categorical metrics

The detailed rerun uses 932 validation windows from four independent 2025
events. Every window contains 10 historical frames and 20 future frames at
six-minute spacing on the 66 x 70 grid. Leads 1--10 are reported as 0--1 h and
leads 11--20 as 1--2 h. Radar is converted with the frozen Marshall--Palmer
relation and categorical thresholds are rain rates of 16 and 32 mm/h.

The next three tables are **teacher-forced one-step motion diagnostics**. At
each lead, the true previous frame is warped, so they isolate motion quality
and are not 20-step operational rollout scores. MAE in the source artifacts is
in normalized Radar units; CSI/POD/FAR/Bias are calculated in mm/h after the
frozen conversion.

### Overall teacher-forced metrics

| Mode | CSI16 | POD16 | FAR16 | Bias16 | CSI32 | POD32 | FAR32 | Bias32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero flow | .6201 | .7666 | .2355 | 1.0027 | .5519 | .7130 | .2905 | 1.0049 |
| Farneback median3 | **.6645** | .7721 | **.1734** | .9342 | .5441 | .6022 | **.1506** | .7089 |
| Learned x.25 | .6482 | .7698 | .1961 | .9576 | **.5707** | **.6945** | .2381 | .9115 |
| Learned x.5 | .6588 | .7669 | .1763 | .9311 | .5604 | .6466 | .1921 | .8003 |
| Learned x1 | .6125 | .7234 | .2003 | .9046 | .4569 | .5138 | .1951 | .6383 |
| Supervised gate x1 (deployed) | .6588 | .7657 | .1749 | .9280 | .5488 | .6252 | .1820 | .7643 |
| Supervised gate x.5 (residual audit) | .6596 | **.7747** | .1839 | .9492 | **.5751** | .6893 | .2237 | .8880 |
| Supervised gate x.75 (residual audit) | **.6646** | .7727 | .1740 | .9355 | .5691 | .6617 | .1973 | .8243 |

### Teacher-forced 0--1 h

| Mode | CSI16 | POD16 | FAR16 | Bias16 | CSI32 | POD32 | FAR32 | Bias32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero flow | .6206 | .7666 | .2348 | 1.0018 | .5519 | .7127 | .2901 | 1.0039 |
| Farneback median3 | **.6671** | **.7734** | **.1708** | .9328 | .5453 | .6018 | **.1468** | .7053 |
| Learned x.25 | .6467 | .7687 | .1971 | .9575 | .5674 | .6931 | .2422 | .9147 |
| Learned x.5 | .6533 | .7637 | .1812 | .9327 | .5524 | .6428 | .2029 | .8064 |
| Learned x1 | .6001 | .7156 | .2120 | .9082 | .4581 | .5264 | .2209 | .6757 |
| Supervised gate x1 (deployed) | .6569 | .7662 | .1783 | .9324 | .5560 | .6430 | .1957 | .7995 |
| Supervised gate x.5 (residual audit) | .6561 | .7732 | .1875 | .9516 | **.5731** | **.6935** | .2325 | .9036 |
| Supervised gate x.75 (residual audit) | .6613 | .7721 | .1783 | .9396 | .5696 | .6710 | .2096 | .8490 |

### Teacher-forced 1--2 h

| Mode | CSI16 | POD16 | FAR16 | Bias16 | CSI32 | POD32 | FAR32 | Bias32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero flow | .6196 | .7665 | .2362 | 1.0036 | .5519 | **.7133** | .2909 | 1.0059 |
| Farneback median3 | .6618 | .7708 | .1761 | .9356 | .5428 | .6025 | **.1545** | .7127 |
| Learned x.25 | .6497 | .7710 | .1950 | .9577 | .5741 | .6960 | .2337 | .9082 |
| Learned x.5 | .6645 | .7702 | .1713 | .9294 | .5690 | .6505 | .1806 | .7939 |
| Learned x1 | .6254 | .7315 | .1881 | .9009 | .4555 | .5005 | .1646 | .5991 |
| Supervised gate x1 (deployed) | .6607 | .7652 | .1713 | .9234 | .5410 | .6065 | .1663 | .7274 |
| Supervised gate x.5 (residual audit) | .6631 | **.7763** | .1801 | .9468 | **.5772** | .6850 | .2142 | .8717 |
| Supervised gate x.75 (residual audit) | **.6679** | .7734 | **.1696** | .9313 | .5686 | .6519 | .1835 | .7984 |

The teacher-forced time blocks remain similar because every prediction is a
six-minute step from a true frame. Their lead index describes the meteorological
state later in the two-hour target sequence, not accumulated forecast error.
The consistent result is that full learned flow has low Bias32 (strong-rain
undercoverage), while reduced/gated flow recovers POD and CSI. Farneback obtains
low FAR partly by becoming conservative: its Bias32 is only about .71.

### Recursive 20-step rollout

The operational validation metric recursively evolves the model's own previous
prediction. This is where errors accumulate and is therefore reported
separately from the motion audit.

| Model / period / threshold | CSI | POD | FAR | Bias |
|---|---:|---:|---:|---:|
| Ungated 0.660787, 0--1 h, 16 mm/h | **.3272** | **.4450** | **.4473** | .8051 |
| Supervised gate ep2, 0--1 h, 16 mm/h | .3119 | .4233 | .4577 | .7806 |
| Ungated 0.660787, 0--1 h, 32 mm/h | **.1922** | **.2279** | **.4486** | .4133 |
| Supervised gate ep2, 0--1 h, 32 mm/h | .1826 | .2205 | .4847 | .4279 |
| Ungated 0.660787, 1--2 h, 16 mm/h | **.1159** | **.1540** | **.6807** | .4822 |
| Supervised gate ep2, 1--2 h, 16 mm/h | .0988 | .1320 | .7184 | .4688 |
| Ungated 0.660787, 1--2 h, 32 mm/h | **.01275** | **.01344** | **.8016** | .0677 |
| Supervised gate ep2, 1--2 h, 32 mm/h | .00415 | .00432 | .9039 | .0449 |

Thus supervised gating improves isolated one-step motion but currently worsens
recursive rollout, especially second-hour 32-mm/h survival. This is direct
evidence that the next design must decouple one-step motion calibration from
long-horizon displacement/intensity survival. It is not yet evidence for adding
a source head, because the near-static gate and recursive transport still need
repair first.

Detailed artifacts:

- `work_dirs/bth_r4a_fixed_flow_detailed/saved/fixed_flow_diagnostics/`
- `work_dirs/bth_r4b2_oracle_gate_ep2_detailed/saved/flow_scale_diagnostics/`

## Decision

R4-b2 is a positive mechanism result but **not yet ready for R4-c**. It passes
zero-flow at 16 mm/h and identifies moving objects, but it narrowly fails CSI32
and still degrades near-static objects.

The next controlled model change should feed an explicit history-motion cue to
the gate (for example the final frame difference and/or history-only
Farneback-median3 magnitude), while keeping raw learned flow frozen. This tests
whether the remaining problem is gate input identifiability rather than gate
capacity. After that, repeat the same object-conditional gate and zero-flow
audit before any source head is added.
