#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL
export PYTHONPATH=/mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL

/home/ranye/miniconda3/envs/OpenSTL/bin/python tools/train.py \
  --dataname bth_radar \
  --method SimVP \
  --config_file configs/bth_radar/SimVP_gSTA_smoke.py \
  --data_root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S \
  --ex_name bth_simvp_gsta_smoke_10ep_s10 \
  --epoch 10 \
  --batch_size 16 \
  --val_batch_size 16 \
  --num_workers 4 \
  --no_display_method_info \
  --deterministic
