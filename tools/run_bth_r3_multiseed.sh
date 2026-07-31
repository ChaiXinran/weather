#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/ranye/miniconda3/envs/OpenSTL/bin/python
DATA_ROOT=/mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S

export PYTHONPATH=.

for seed in 0 1 2; do
  "${PYTHON_BIN}" tools/train.py \
    --dataname bth_radar \
    --method SimVP \
    --config_file configs/bth_radar/SimVP_gSTA_r3.py \
    --data_root "${DATA_ROOT}" \
    --ex_name "bth_simvp_gsta_r3_direct_10ep_seed${seed}" \
    --epoch 10 \
    --batch_size 16 \
    --val_batch_size 8 \
    --seed "${seed}" \
    --deterministic \
    --no_display_method_info \
    --skip_test_after_train
done
