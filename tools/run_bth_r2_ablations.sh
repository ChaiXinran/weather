#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/ranye/miniconda3/envs/OpenSTL/bin/python
DATA_ROOT=/mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S

export PYTHONPATH=.

for variant in r2a r2b r2d; do
  "${PYTHON_BIN}" tools/train.py \
    --dataname bth_radar \
    --method SimVP \
    --config_file "configs/bth_radar/SimVP_gSTA_${variant}.py" \
    --data_root "${DATA_ROOT}" \
    --ex_name "bth_simvp_gsta_${variant}_5ep_seed0" \
    --epoch 5 \
    --batch_size 16 \
    --val_batch_size 8 \
    --seed 0 \
    --deterministic \
    --no_display_method_info \
    --skip_test_after_train
done
