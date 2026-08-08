# Frozen project decisions

Last updated: 2026-08-08

1. The first reproducible baseline is Radar-only. PWV and DEM are introduced only after the Radar pipeline, evaluation, and persistence comparison are stable.
2. The task is fixed as 10 observed frames (60 min) to 20 forecast frames (120 min), at six-minute intervals on a 66×70 grid.
3. Formal truth is future Radar reflectivity converted to rain rate, not direct Rain PNG. Rain PNG is used for Z-R selection and diagnostics because it is capped, quantized, and not independent station validation.
4. The operational Z-R relation is Marshall–Palmer, \(Z=200R^{1.6}\), with \(z_0=0\) dBZ. Selection used 2025 train/validation only; test was not used. Evidence: `.research/zr_protocol_decision.json`.
5. Rain/Radar comparison uses a frozen +42-minute Rain lag, zero row shift, and +1 column shift.
6. Radar PNGs are retained as the source of truth, while training reads a lossless uint8 NumPy mmap cache. Normalization remains in the data loader.
7. Persistence is a mandatory baseline under exactly the same valid mask and thresholds as the model.
8. Evaluation is frozen as five layers: data quality, pixel scores, spatial structure, object evolution, and statistical confidence.
9. Strict 1×1 categorical scores are primary; neighborhood/FSS results are diagnostic and may not replace them.
10. Undefined categorical denominators are recorded as NaN/null and excluded from macro averages, never silently forced to zero.
11. Checkpointing now keeps validation-loss best, precipitation-score best, and last checkpoints separately. Old “best means val-loss best” results must remain labeled as such.
12. Current MSE is a baseline loss, not the intended final loss. A strong-rain-aware objective is planned but not yet frozen or implemented.
13. The current engineering baseline is DirectPhysicsHybrid V2 with a reported Validation weighted CSI score of 0.937194; the report is `.research/mixed/satge1.md`.
14. V3a prioritizes observed forecast skill over requiring one branch to own the final output. ConvLSTM remains a stable prior while U-Net may make strong spatially selective corrections.
15. V3a uses routed preserve, motion, and decay candidates. Growth is deferred because current source attribution is nearly neutral and Radar-only initiation is weakly identifiable.
16. V3a routing uses 16 mm/h storm footprints, 32 mm/h core refinement, a two-grid initial search radius, multi-threshold soft labels, and ignore masks for ambiguous cases.
17. Mechanism exploration is effect-first: seed 0 and Validation are sufficient initially; Test, multi-seed runs, bootstrap gates, and broad sweeps are deferred until a mechanism is worth formalizing.
