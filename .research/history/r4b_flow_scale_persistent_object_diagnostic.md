# R4-b D1/D2 flow-scale and persistent-object diagnostic

## Scope

This is a no-training mechanism audit of the best continued R4-b checkpoint
(`val_csi_score=0.660787`). It uses the frozen 2025 validation split (932
windows, four events) and rain-rate warping throughout.

- D1 scales the identical predicted flow by
  `[-1, 0, .25, .5, .75, 1, 1.25]` and reports full-field, per-lead, and
  per-event scores.
- D2 matches persistent 16/32-mm/h objects between each true previous/next
  frame, excludes ambiguous split/merge candidates, requires area and energy
  ratios in `[0.5, 2]`, and compares the rain-weighted object flow with the
  observed rain-weighted centroid displacement.
- Artifacts:
  `work_dirs/bth_r4b_ft_ep0_flow_scale_diag/saved/flow_scale_diagnostics/`.

Persistent-object counts below are object-window observations. Overlapping
forecast windows repeat some meteorological objects, so they are diagnostic
observations rather than 115,707 independent storms.

## D1 full-field scale and sign audit

| Alpha | MAE normalized | CSI16 | CSI32 |
|---:|---:|---:|---:|
| -1.00 | 0.02566 | 0.45026 | 0.33168 |
| 0.00 | 0.01541 | 0.62011 | 0.55187 |
| 0.25 | 0.01419 | 0.64816 | **0.57066** |
| 0.50 | **0.01392** | **0.65879** | 0.56043 |
| 0.75 | 0.01411 | 0.64575 | 0.51411 |
| 1.00 | 0.01474 | 0.61245 | 0.45686 |
| 1.25 | 0.01584 | 0.56982 | 0.40622 |

The warp sign is not reversed: `alpha=-1` is dramatically worse. Positive
scaled flow beats zero flow simultaneously in continuous error and strong-rain
CSI. The all-field optimum is about 0.5 for MAE/CSI16 and 0.25 for CSI32. Thus
the learned flow contains useful directional information, but applying its full
magnitude everywhere causes over-transport, especially for extreme rain.

This conclusion is not driven by one event. All four validation events select
0.5 for CSI16. Three select 0.25 for CSI32 and one selects 0.5. Per-lead
selection is also stable after the first few leads: CSI16 generally favors 0.5
and CSI32 generally favors 0.25.

## D2 persistent-object direction, scale, and necessity

| Threshold / true displacement | Count | True displacement | Predicted flow | Mean cosine | Positive direction | Median magnitude ratio | Best alpha by endpoint error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 / near-static `<.2 px` | 37,065 | .041 | .405 | n/a | n/a | n/a | **0** |
| 16 / subpixel `.2-.5 px` | 22,922 | .349 | .438 | .559 | 83.0% | 1.265 | **.5** |
| 16 / moving `.5-1 px` | 11,359 | .641 | .507 | .729 | 90.6% | .800 | **1** |
| 16 / fast/difficult `>1 px` | 6,730 | 1.103 | .561 | .680 | 87.1% | .534 | **1.25** |
| 32 / near-static `<.2 px` | 21,618 | .018 | .385 | n/a | n/a | n/a | **0** |
| 32 / subpixel `.2-.5 px` | 8,215 | .402 | .416 | .483 | 78.4% | 1.028 | **.5** |
| 32 / moving `.5-1 px` | 4,661 | .639 | .439 | .621 | 85.7% | .678 | **1** |
| 32 / fast/difficult `>1 px` | 3,137 | 1.116 | .470 | .623 | 85.1% | .429 | **1.25** |

The dominant defect is conditional, not a uniform scale or sign error:

1. Near-static objects are over-moved. Their true displacement is almost zero,
   but the head still emits about 0.39--0.41 pixels; zero flow is optimal.
2. Subpixel objects benefit from reducing flow to about half scale.
3. Clearly moving objects benefit from retaining full flow. Their mean
   direction cosine is positive and high, and 86--91% point into the correct
   half-plane.
4. Fast/difficult matches prefer more than the emitted magnitude, although
   centroid motion can include morphology/source effects and these matches are
   the least identifiable group.

Therefore the global 0.25/0.5 optimum mainly reflects a mixture dominated by
near-static and subpixel objects. It would be incorrect to permanently multiply
every flow by one fixed small constant.

## Revised mechanism verdict

The previous statement "predicted flow is worse than zero flow" remains true
for the uncalibrated `alpha=1` output, but D1/D2 refine its interpretation:

- direction convention is correct;
- learned flow contains real motion signal;
- physical motion is identifiable enough to beat zero flow without retraining;
- the current head fails to predict when little or no motion should be applied;
- a single unconditional full-strength flow causes the motion-gate failure.

This is stronger evidence than the earlier aggregate gate and changes the next
model decision from a generic loss rewrite to **R4-b2 motion confidence/scale
gating**. A first implementation should predict `g in [0,1]` per sample and
lead (or a small spatial confidence map) and use `v_final = g * v_raw`, with a
conservative initialization near 0.5. Strong-rain/edge-weighted teacher-forced
supervision remains appropriate for learning that gate.

## Promotion decision

Do not enter R4-c yet. Post-hoc scaled flow passes the aggregate zero-flow test,
and persistent moving objects show genuine directional skill, so R4-b is now a
promising rather than failed mechanism. However, the deployed model still emits
uncalibrated full-strength flow and over-moves the numerous near-static objects.

Next order:

1. Run the history-only fixed optical-flow R4-a identifiability baseline.
2. Implement R4-b2 motion confidence/scale gating, informed by the D1/D2
   results rather than hard-coding one global alpha.
3. Repeat D1/D2 and the full rollout evaluation.
4. Add the R4-c source term only after the learned gated model beats zero flow
   on persistent moving objects and does not degrade near-static objects.
