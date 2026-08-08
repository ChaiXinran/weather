# OpenSTL BTH Research Agent Guide

Before changing data loading, model code, training configs, evaluation, or
reporting, read `.research/detail.md`. It is the authoritative source for:

- local WSL and server environment differences;
- dataset roots and Radar/PWV/RAIN cache layout;
- 10→20 task shapes, units, manifests, and frozen evaluation conventions;
- training, validation, automatic reporting, and attribution command templates;
- the current best baseline and active V3a mechanism plan.

Then read `.research/context_index.md` and only the current files linked from
the latest continuation section of `.research/detail.md`. Machine-readable
state is in `.research/project_manifest.yml`, `.research/data_dictionary.yml`,
and `.research/experiment_matrix.yml`.

Current active design documents:

- `.research/baselines/v2/summary.md` — current V2 baseline summary;
- `.research/mixed/satge1.md` — complete V2 validation/attribution report;
- `.research/mixed/v3a_routing_protocol.md` — routing-label protocol;
- `.research/mixed/v3a_implementation_plan.md` — implementation sequence.

Operational rules:

1. Preserve the user's dirty worktree and do not overwrite unrelated changes.
2. Do not use Test during mechanism development unless the user explicitly
   requests it; use the fixed Validation split and automatic report tools.
3. Keep the existing DirectPhysicsHybrid V2 implementation reproducible;
   implement major V3a changes as separate model/method/config files.
4. Local and server commands are not identical. Copy the correct template from
   `.research/detail.md`; never reuse the local Python absolute path on server.
5. Cache directories are relative to `--data_root`. Do not rebuild or overwrite
   a cache unless explicitly requested.
6. Results from different manifests, truth definitions, or evaluation
   protocols must not be compared as if they were directly equivalent.
7. During effect-first mechanism exploration, prefer a few full mechanism runs
   over broad hyperparameter sweeps or premature multi-seed significance work.
