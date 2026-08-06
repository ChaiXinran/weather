# Weight Inventory - 2026-08-06

Scan root: `D:\_Search\AIforScience\Rewritten\origin\OpenSTL`

## Summary

- Initial weight-like files found under project: 216 files.
- Initial checkpoints: 213 `.ckpt`, 6159.60 MB.
- Cleanup performed after inventory: deleted 207 non-reference `.ckpt` files, 5975.77 MB.
- Remaining checkpoints: 6 `.ckpt`, 183.83 MB.
- Diagnostic arrays: 3 `.npz`, 8.26 MB.
- No `.pth`, `.pt`, `.onnx`, `.safetensors`, `.bin`, `.h5`, or `.pkl` model files were found.
- All remaining checkpoints are under `work_dirs/`; source/config/docs directories do not contain model weights.

## Space Buckets

| Bucket | Count | Size MB | Note |
|---|---:|---:|---|
| `last.ckpt` | 31 | 958.64 | Resume-only snapshots; removable after runs are closed. |
| `val-*` epoch checkpoints | 117 | 3298.46 | Top-k/epoch snapshots; usually redundant if `best_*` aliases are kept. |
| `best*` / `selected*` checkpoints | 65 | 1902.49 | Stable named candidates; safest compact retention target. |
| Smoke checkpoints | 3 | 113.50 | Early smoke run artifacts. |

## Recommended Baseline / Reference Checkpoints

Keep these unless the corresponding report is retired:

| Role | Checkpoint |
|---|---|
| Formal SimVP baseline | `work_dirs/bth_simvp_gsta_formal_seed0/checkpoints/best.ckpt` |
| Selected SimVP R3 reference | `work_dirs/bth_simvp_gsta_r3_direct_10ep_seed0/checkpoints/selected_csi_0.714247_epoch03.ckpt` |
| ConvLSTM high-CSI baseline | `work_dirs/bth_convlstm_r2d_ft3ep_seed0/checkpoints/best_val_csi.ckpt` |
| ConvLSTM conservative/Pareto baseline | `work_dirs/bth_convlstm_r2d_ft6ep_seed0/checkpoints/best_val_csi.ckpt` |
| R4-b motion baseline used by later R4d work | `work_dirs/bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0/checkpoints/best_val_csi.ckpt` |
| Current R4-d2 evaluated mechanism checkpoint | `work_dirs/bth_r4d2_edge_factorized_s20_20ep_seed0/checkpoints/val-csi-epoch=04-val_csi_score=0.593111.ckpt` |

Keeping only the six checkpoints above would retain about 183.83 MB of `.ckpt` files and make about 5975.77 MB eligible for deletion. This is aggressive and should only be done after confirming no further resume/debug work needs the intermediate runs.

Status: this aggressive cleanup was performed on 2026-08-06. The six checkpoints listed above were retained.

## Run-Level Classification

| Category | Runs | Recommendation |
|---|---|---|
| Early smoke / sanity | `bth_simvp_gsta_smoke_10ep_s10` | Safe to delete checkpoints if its evaluation tables are enough. |
| SimVP baseline lineage | `bth_simvp_gsta_formal_seed0`, `bth_simvp_gsta_r1_5ep_seed0`, `bth_simvp_gsta_r2_5ep_seed0`, `bth_simvp_gsta_r2a_5ep_seed0`, `bth_simvp_gsta_r2b_5ep_seed0`, `bth_simvp_gsta_r2d_5ep_seed0`, `bth_simvp_gsta_r3_direct_*` | Keep formal baseline and selected R3. Compact or delete R1/R2 ablation checkpoints after reports are accepted. |
| ConvLSTM baselines | `bth_convlstm_r2d_5ep_seed0`, `bth_convlstm_r2d_ft3ep_seed0`, `bth_convlstm_r2d_ft6ep_seed0` | Keep `ft3` and `ft6` best CSI. Base 5ep run can be compacted unless needed for resume/provenance. |
| R4-b motion / rain-rate | `bth_r4b_motion_pre0788_5ep_seed0`, `bth_r4b_motion_rainrate_scale05_5ep_seed0`, `bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0`, `bth_r4b_rain_nostop_*`, `bth_r4b_rain_stopgrad_5ep_seed0`, `bth_r4b_norm_stopgrad_5ep_seed0` | Keep the continued `scale05_ft5ep` best checkpoint as the R4-b baseline; compact the rest to best aliases or logs. |
| R4-b2 gate tests | `bth_r4b2_spatial_gate_only_5ep_seed0`, `bth_r4b2_oracle_gate_only_5ep_seed0`, `bth_r4b2_oracle_gate_pretrain_3ep_seed0` | Stage tests; likely compact to best aliases or remove after report snapshots are enough. |
| R4-c source tests | `bth_r4c1_source_only_3ep_seed0`, `bth_r4c1_source_active_3ep_seed0`, `bth_r4c2a_bounded_state_tf_seed0`, `bth_r4c2b_bounded_state_tendency_tf_seed0` | Mechanism exploration with weak validation CSI for c2; keep only if needed for debugging source behavior. |
| R4-d factorized / current mechanism | `bth_r4d1a_factorized_s1_30ep_seed0`, `bth_r4d1b_factorized_capacity_s1_20ep_seed0`, `bth_r4d2_edge_factorized_s20_20ep_seed0` | Current active branch. Keep R4-d2 evaluated checkpoint; compact older R4-d1 runs if not resuming. |
| Motion diagnostic arrays | three `sample_motion_fields.npz` files under motiondiag dirs | Small, optional. Delete only if plots/CSV diagnostics are enough. |

## Practical Cleanup Tiers

1. Conservative: delete all `last.ckpt` from completed runs. Saves about 958.64 MB.
2. Compact: keep `best*` and `selected*` in each run, delete `last.ckpt` and `val-*` epoch snapshots. Saves about 4257.10 MB.
3. Aggressive: keep only the six reference checkpoints listed above. Saves about 5975.77 MB.

Deletion performed: 207 non-reference `.ckpt` files were removed. The `.npz` diagnostic files were not deleted.
