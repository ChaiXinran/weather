# Research workspace run log

## 2026-08-01 - R4-a and R4-b2 spatial motion gate

- R4-a history-only Farneback median3 improved CSI16 from zero-flow .62011 to
  .66446 but did not improve CSI32 (.54408 vs .55187). Persistent moving
  objects improve while near-static objects are over-moved, establishing
  partial six-minute motion identifiability and the need for spatial gating.
- Added an optional spatial per-lead flow gate and compatible loading of the
  0.660787 encoder/raw-motion checkpoint.
- Gate-only training under the existing objective collapsed toward full flow
  (gate about .94), confirming the need for direct motion-necessity supervision.
- Added strong-rain oracle-scale gate targets using teacher-forced candidate
  scales. Oracle-only gate pretraining learned an ordered gate (.57 near-static
  vs .71 moving at 16 mm/h), recovered teacher-forced CSI16 to .65881 at its
  deployed scale, and reduced the CSI32 deficit to .00303, but still over-moves
  near-static objects.
- R4-b2 is promising but R4-c remains gated. Next change: explicit historical
  difference/fixed-flow cue into the confidence gate.
- Report: `.research/history/r4a_r4b2_motion_gate_analysis.md`.

## 2026-08-01 - R4-b D1/D2 no-training flow audit

- Audited the 0.660787 checkpoint with rain-rate teacher-forced flow scales
  `[-1, 0, .25, .5, .75, 1, 1.25]` on all 932 validation windows.
- Positive scaled flow beats zero flow: alpha .5 gives CSI16 .65879 and alpha
  .25 gives CSI32 .57066, versus zero-flow .62011/.55187. Negative flow is
  dramatically worse, excluding a global sign/convention error.
- Across four events, CSI16 always selects alpha .5; CSI32 selects .25 in three
  events and .5 in one.
- 115,707 persistent object-window observations show conditional calibration:
  near-static objects prefer zero, subpixel objects prefer .5, clearly moving
  objects prefer 1, and fast/difficult matches prefer 1.25. Moving-object flow
  points into the correct half-plane for roughly 86--91% of observations.
- Revised interpretation: flow contains real directional signal but lacks a
  motion-necessity/confidence gate. Proceed to R4-a and R4-b2 gating, not R4-c.
- Report: `.research/history/r4b_flow_scale_persistent_object_diagnostic.md`.
- Artifacts: `work_dirs/bth_r4b_ft_ep0_flow_scale_diag/`.

## 2026-08-01 - R4-b five-epoch continuation from C

- Initialized the full rain-rate/no-stop C model from its 0.642496 checkpoint
  and ran a fresh five-epoch OneCycle under the original two-epoch encoder
  freeze protocol.
- Validation CSI followed 0.660787, 0.639803, 0.587591, 0.602436, 0.614035.
  Epoch 0 proves limited additional optimization headroom, while the later
  decline shows that restarting the original peak learning rates is too
  aggressive for the converged model.
- The epoch-0 checkpoint improved recursive CSI16/32 to 0.22428/0.10987, but
  its teacher-forced rain-rate flow remained below zero flow for CSI, FSS,
  centroid error, and strong-rain area. R4-c remains no-go.
- Analysis: `.research/history/r4b_rain_nostop_ft5ep_analysis.md`.
- Artifacts: `work_dirs/bth_r4b_rain_nostop_ft5ep_from0642496_seed0/` and
  `work_dirs/bth_r4b_rain_nostop_ft_ep0_motiondiag/`.

## 2026-08-01 - R4-b rain-rate/no-stop full motion gate

- Evaluated the epoch-4 best checkpoint of the 2x2 winner on all 932 frozen
  validation windows with the full precipitation/spatial/object/event protocol.
- Rain-rate/no-stop modestly improved rollout CSI, RMSE, FSS, centroid error,
  object POD, and object FAR over normalized-dBZ/no-stop, but strong-rain area
  and energy retention deteriorated further.
- In the strict teacher-forced comparison, predicted rain-rate flow reduced
  continuous error but was worse than zero flow for CSI16/32, FSS16/32,
  centroid error, and thresholded area retention. The motion gate remains
  failed, so R4-c is not authorized.
- Full report:
  `.research/history/r4b_rain_nostop_full_motion_analysis.md`.
- Artifacts remain under `work_dirs/bth_r4b_rain_nostop_best_valdiag/` and
  `work_dirs/bth_r4b_rain_nostop_motiondiag/`.

## 2026-08-01 — R4-b warp-space x stop-gradient 2x2 training ablation

- Completed three controlled five-epoch runs that, together with the original
  R4-b run, form a normalized-dBZ/rain-rate x no-stop/stop-gradient 2x2 design.
- Rain-rate/no-stop achieved the best validation precipitation score (0.642496)
  and improved first-hour CSI16/32, FAR, and intensity retention over the
  normalized-dBZ/no-stop run.
- Stop-gradient reduced validation precipitation score in both variable spaces,
  despite slightly improving normalized-dBZ val loss, and is rejected as the
  current default.
- Training analysis: `.research/history/r4b_warp_stopgrad_2x2_training_analysis.md`.

## 2026-08-01 — R4-b pretrained motion-only diagnostic

- Completed a 5-epoch seed-0 learned motion-only run initialized from the
  ConvLSTM 0.788316 encoder checkpoint.
- Evaluated epoch 4 on the frozen 2025 validation split with the full
  precipitation, spatial, object, event, and bootstrap evaluator.
- The initial rollout showed lower centroid error/object FAR but severe
  strong-rain area, energy, FSS, and object-POD loss. A dedicated flow diagnostic
  then showed predicted flow was worse than a teacher-forced zero-flow control
  for strong-rain CSI/FSS/centroid metrics despite lower continuous error;
  classified R4-b as failing the motion gate and not ready for R4-c.
- Full analysis: `.research/history/r4b_motion_pre0788_5ep_analysis.md`.
- Large diagnostics remain under
  `work_dirs/bth_r4b_motion_pre0788_best_valdiag/`.
- Added a no-retraining operator audit with normalized-dBZ, linear-Z,
  rain-rate, and zero-flow teacher-forced modes. Physical-space warping restored
  substantial strong-rain CSI/FSS but did not yet beat zero flow.

## 2026-07-31 — Historical version archive

- Created `.research/history/` with a human-readable README and machine-readable version index.
- Archived the completed first formal 30-epoch Radar-only run as version V0.1, including frozen inputs, model/loss/training settings, checkpoint semantics, full key results, conclusions, later changes, and links to original artifacts.
- Large checkpoints, plots, and per-sample/per-object tables remain in `work_dirs/bth_simvp_gsta_formal_seed0/` and were not duplicated.

## 2026-07-31 — Context compression

- Audited `.research/detail.md`, event manifests, Z-R artifacts, configs, data-loading/evaluation/training code, work directories, checkpoints, logs, and current Git state.
- Created a machine-readable project manifest, data dictionary, experiment matrix, frozen-decision log, open-question list, and the human-readable context index.
- Historical metrics are labeled by protocol; incomplete R1 results are explicitly separated from completed formal results.
- No training was launched and no model/configuration code was changed during this context-compression run.
## 2026-08-01 - R4-a/R4-b2 detailed time-and-rain-rate metrics

- Recomputed the frozen 932-window teacher-forced motion diagnostics with
  CSI, POD, FAR, and Bias split into 0--1 h versus 1--2 h and 16 versus
  32 mm/h thresholds.
- Added the matching recursive 20-step validation breakdown for the ungated
  0.660787 checkpoint and oracle-supervised gate epoch 2.
- Teacher-forced gated/reduced flow improves motion alignment, but the deployed
  gate loses second-hour CSI32 and the oracle gate worsens recursive survival;
  R4-c therefore remains a no-go.
- Report: `.research/history/r4a_r4b2_motion_gate_analysis.md`.
- Detailed artifacts:
  `work_dirs/bth_r4a_fixed_flow_detailed/saved/fixed_flow_diagnostics/` and
  `work_dirs/bth_r4b2_oracle_gate_ep2_detailed/saved/flow_scale_diagnostics/`.
## 2026-08-01 - Formal simplified R4-b configuration

- Added `configs/bth_radar/ConvLSTM_evolution_motion_rainrate_scale05.py` as
  the canonical motion-only R4-b configuration.
- The configuration fixes the `rain_rate` motion-only path, disables the
  optional learned gate and oracle supervision, and limits displacement to
  1 pixel per 6-minute step. Relative to the original 2-pixel bound, this is
  exactly the selected global 0.5 flow calibration for an existing checkpoint.
- Stop-gradient/no-stop-gradient is not promoted as part of this inference
  configuration; the explicit override was removed.
- No source/sink or intensity-generation head is present.
## 2026-08-01 - R4-b rain-rate scale-0.5 five-epoch run

- Completed five epochs in the WSL OpenSTL environment from the ConvLSTM
  0.788316 history encoder using the formal rain-rate motion-only configuration.
- Validation CSI score progressed .529999, .529821, .589512, .598675, and
  .633323; epoch 4 is CSI-best, while epoch 3 is val-loss-best (.009423).
- Against ConvLSTM 0.788316, epoch 4 lowers FAR in every 16/32-mm/h time block
  but loses POD and Bias severely, especially CSI32 in hour two (.00878 versus
  .05439). It is a conservative motion ablation, not a baseline replacement.
- Report: `.research/history/r4b_motion_rainrate_scale05_5ep_analysis.md`.
- Artifacts: `work_dirs/bth_r4b_motion_rainrate_scale05_5ep_seed0/` and
  `lightning_logs/version_40/metrics.csv`.
## 2026-08-02 - R4-b scale-0.5 low-LR five-epoch continuation

- Continued the `.633323` checkpoint for five epochs with motion-head LR
  `1e-4` and encoder LR `2e-5`.
- Actual configured encoder freeze remained two epochs; the best score was
  `.640662` at continuation epoch 1 while only the motion path was updating.
- Scores then fell to `.619840/.617094` after encoder unfreezing and recovered
  only to `.628359`, showing that further backbone adaptation is harmful here.
- The selected continuation checkpoint slightly exceeds ConvLSTM 0.788 at
  second-hour CSI16 (.11230 vs .10982), but second-hour CSI32 remains only
  .00923 vs .05439. Stop identical continuation and retain it as the best clean
  motion-only scale-0.5 checkpoint.
- Report: `.research/history/r4b_motion_rainrate_scale05_5ep_analysis.md`.
## 2026-08-05 - R4-c0 oracle source and R4-c1 frozen-motion source head

- Refactored the evolution operator so signed source is added in rain-rate
  space and returned separately from advected/evolved rain; source-free R4-b
  compatibility and zero-source initialization passed ten targeted checks.
- Audited oracle source on 11,245 train and 932 validation windows using the
  frozen `.640662` motion checkpoint. Train/val absolute P99 is 27.32/27.23
  mm/h in 16-mm/h regions and 33.35/32.30 in 32-mm/h regions.
- Set `Smax=35 mm/h` from training extreme-rain P99, with validation used only
  as independent confirmation.
- Ran two three-epoch frozen-motion source-head experiments. Both stayed near
  zero source; the active-weight rerun peaked at `.641226` and did not reduce
  direct source loss. POD32/Bias32 did not recover.
- R4-c0 passes but R4-c1 fails its mechanism gate. Do not enter joint R4-c2;
  next isolate direct active oracle-source pretraining.
- Report: `.research/history/r4c0_r4c1_source_analysis.md`.
