# Research workspace run log

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

## 2026-07-31 — Historical version archive

- Created `.research/history/` with a human-readable README and machine-readable version index.
- Archived the completed first formal 30-epoch Radar-only run as version V0.1, including frozen inputs, model/loss/training settings, checkpoint semantics, full key results, conclusions, later changes, and links to original artifacts.
- Large checkpoints, plots, and per-sample/per-object tables remain in `work_dirs/bth_simvp_gsta_formal_seed0/` and were not duplicated.

## 2026-07-31 — Context compression

- Audited `.research/detail.md`, event manifests, Z-R artifacts, configs, data-loading/evaluation/training code, work directories, checkpoints, logs, and current Git state.
- Created a machine-readable project manifest, data dictionary, experiment matrix, frozen-decision log, open-question list, and the human-readable context index.
- Historical metrics are labeled by protocol; incomplete R1 results are explicitly separated from completed formal results.
- No training was launched and no model/configuration code was changed during this context-compression run.
