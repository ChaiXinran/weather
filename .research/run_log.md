# Research workspace run log

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
