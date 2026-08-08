# Open questions

Last updated: 2026-08-08

- Which heavy-rain-aware loss improves CSI at 16/32 mm/h without materially worsening first-hour CSI, FAR, or weak-rain area?
- Should the 20-frame horizon continue to use recursive 10→10 SimVP calls, or should the architecture predict all 20 frames directly?
- Does the validation precipitation score generalize beyond the four 2025 validation events? Complete R1 before drawing conclusions.
- How stable are the Radar-only results across seeds 0, 1, and 2?
- What is the correct PWV preprocessing, temporal alignment, missing-data mask, and normalization protocol?
- How should DEM enter the network—as a static adapter, conditioning branch, or terrain-aware correction—and what ablation isolates its contribution?
- Can motion and source/sink terms be made explicit and physically interpretable while preserving forecast skill?
- Which external-year protocol should be used for 2023, given sparse/duplicated Rain data?
- Should station/gauge observations be added for truly independent rainfall validation?
- When should object trajectories, lifetime, split/merge statistics, and full CRA decomposition be promoted from planned to formal metrics?
- Is the current validation batch-size mismatch (requested 16, observed 8) intentional or a configuration-merge bug?
- Should `torch.set_float32_matmul_precision("high" or "medium")` be enabled for RTX 4060 Tensor Cores after a reproducibility/speed check?
- What numerical IoU/score/margin settings best separate confident one-to-one motion matches from ambiguous object matches in the V3a routing cache?
- Can a dedicated decay expert materially reduce +60/+120 min FAR without losing the motion candidate's CSI gain?
- How much of the oracle-routed gain can the inference-time router recover from history, direct-prior context, and lead information alone?
- After V3a, should growth be introduced only with PWV/environmental conditioning or first tested as a Radar-only expert?
