#!/usr/bin/env bash

# End-to-end server workflow for BTH V3a after the routing cache is built.
#
# Run from the repository root (including from a PowerShell terminal):
#   bash tools/run_bth_v3a_full.sh
#
# Optional overrides:
#   RUN_TAG=my_run SEED=0 BATCH_SIZE=4 NUM_WORKERS=4 \
#     CANDIDATE_EPOCHS=3 ROUTER_EPOCHS=3 JOINT_EPOCHS=8 \
#     bash tools/run_bth_v3a_full.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CANDIDATE_EPOCHS="${CANDIDATE_EPOCHS:-3}"
ROUTER_EPOCHS="${ROUTER_EPOCHS:-3}"
JOINT_EPOCHS="${JOINT_EPOCHS:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

CANDIDATE_EXP="bth_v3a_candidate_seed${SEED}_${RUN_TAG}"
ROUTER_EXP="bth_v3a_router_seed${SEED}_${RUN_TAG}"
JOINT_EXP="bth_v3a_joint_seed${SEED}_${RUN_TAG}"
TEST_EXP="${JOINT_EXP}_test_best"

LOG_DIR="run_logs/${RUN_TAG}"
mkdir -p "${LOG_DIR}"

DIRECT_CKPT="work_dirs/bth_convlstm_r2d_ft3ep_seed0/checkpoints/best_val_csi.ckpt"
V2_CKPT="work_dirs/bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0/checkpoints/best_val_csi.ckpt"
ROUTING_CACHE="${DATA_ROOT}/V3A_ROUTING_CACHE"

require_file() {
    local path="$1"
    local description="$2"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: missing ${description}: ${path}" >&2
        exit 1
    fi
}

run_logged() {
    local log_path="$1"
    shift
    "$@" 2>&1 | tee "${log_path}"
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

require_file "${DIRECT_CKPT}" "frozen ConvLSTM checkpoint"
require_file "${V2_CKPT}" "V2 initialization checkpoint"
require_file "${ROUTING_CACHE}/manifest.json" "routing-cache manifest"
require_file "${ROUTING_CACHE}/train_labels.npy" "training routing labels"
require_file "${ROUTING_CACHE}/val_labels.npy" "validation routing labels"

echo "Run tag: ${RUN_TAG}"
echo "Repository: ${REPO_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Logs: ${LOG_DIR}"

echo "========== Stage A: motion/decay candidate pretraining =========="
run_logged "${LOG_DIR}/01_candidate.log" \
    "${PYTHON_BIN}" tools/train.py \
    --dataname bth_radar \
    --method DirectPhysicsRouted \
    --config_file configs/bth_radar/DirectPhysicsRouted_v3a.py \
    --data_root "${DATA_ROOT}" \
    --ex_name "${CANDIDATE_EXP}" \
    --epoch "${CANDIDATE_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --deterministic \
    --no_display_method_info \
    --skip_test_after_train

CANDIDATE_CKPT="work_dirs/${CANDIDATE_EXP}/checkpoints/best_val_csi.ckpt"
require_file "${CANDIDATE_CKPT}" "Stage-A best checkpoint"

echo "========== Stage B: router training =========="
run_logged "${LOG_DIR}/02_router.log" \
    "${PYTHON_BIN}" tools/train.py \
    --dataname bth_radar \
    --method DirectPhysicsRouted \
    --config_file configs/bth_radar/DirectPhysicsRouted_v3a_router.py \
    --data_root "${DATA_ROOT}" \
    --init_from_ckpt "${CANDIDATE_CKPT}" \
    --ex_name "${ROUTER_EXP}" \
    --epoch "${ROUTER_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --deterministic \
    --no_display_method_info \
    --skip_test_after_train

ROUTER_CKPT="work_dirs/${ROUTER_EXP}/checkpoints/best_val_csi.ckpt"
require_file "${ROUTER_CKPT}" "Stage-B best checkpoint"

echo "========== Stage C: joint fine-tuning =========="
run_logged "${LOG_DIR}/03_joint.log" \
    "${PYTHON_BIN}" tools/train.py \
    --dataname bth_radar \
    --method DirectPhysicsRouted \
    --config_file configs/bth_radar/DirectPhysicsRouted_v3a_joint.py \
    --data_root "${DATA_ROOT}" \
    --init_from_ckpt "${ROUTER_CKPT}" \
    --ex_name "${JOINT_EXP}" \
    --epoch "${JOINT_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --deterministic \
    --no_display_method_info \
    --skip_test_after_train

JOINT_CKPT="work_dirs/${JOINT_EXP}/checkpoints/best_val_csi.ckpt"
require_file "${JOINT_CKPT}" "Stage-C best checkpoint"

echo "========== Test: best joint checkpoint =========="
run_logged "${LOG_DIR}/04_test_best.log" \
    "${PYTHON_BIN}" tools/test.py \
    --dataname bth_radar \
    --method DirectPhysicsRouted \
    --config_file configs/bth_radar/DirectPhysicsRouted_v3a_joint.py \
    --data_root "${DATA_ROOT}" \
    --ckpt_path "${JOINT_CKPT}" \
    --ex_name "${TEST_EXP}" \
    --test \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --deterministic \
    --no_display_method_info

echo "========== Full validation report =========="
run_logged "${LOG_DIR}/05_validation_report.log" \
    "${PYTHON_BIN}" tools/report_bth_workdir.py \
    --work_dir "work_dirs/${JOINT_EXP}" \
    --data_root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"

echo "========== V3a candidate/routing attribution =========="
run_logged "${LOG_DIR}/06_v3a_attribution.log" \
    "${PYTHON_BIN}" tools/evaluate_v3a_attribution.py \
    --work_dir "work_dirs/${JOINT_EXP}" \
    --data_root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}"

SUMMARY_FILE="${LOG_DIR}/run_summary.txt"
{
    echo "V3a full run completed"
    echo "run_tag=${RUN_TAG}"
    echo "candidate_work_dir=work_dirs/${CANDIDATE_EXP}"
    echo "router_work_dir=work_dirs/${ROUTER_EXP}"
    echo "joint_work_dir=work_dirs/${JOINT_EXP}"
    echo "best_checkpoint=${JOINT_CKPT}"
    echo "test_work_dir=work_dirs/${TEST_EXP}"
    echo "validation_report=work_dirs/${JOINT_EXP}/validation_report.md"
    echo "attribution_report=work_dirs/${JOINT_EXP}/v3a_attribution.md"
    echo "attribution_json=work_dirs/${JOINT_EXP}/v3a_attribution.json"
} | tee "${SUMMARY_FILE}"

