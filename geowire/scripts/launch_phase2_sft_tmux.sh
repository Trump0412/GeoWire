#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/guojh/lq/new/GeoWire/geowire}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geothinker/bin/python}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-/mnt/guojh/lq/new/GeoWire/.deps/transformers_4_57_6:${PROJECT_ROOT}}"
SESSION="${SESSION:-geowire_phase2_sft}"
QA_MANIFEST="${QA_MANIFEST:?set QA_MANIFEST to a GeoWire Phase2 QA manifest}"
TIP_MANIFEST="${TIP_MANIFEST:-}"
CACHE_ROOT="${CACHE_ROOT:?set CACHE_ROOT to a real GeoWire cache root}"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-/mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct}"
PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT:?set PHASE1_CHECKPOINT to Phase1 geowire_adapter.pt}"
PARITY_REPORT="${PARITY_REPORT:?set PARITY_REPORT to a passing qwen bridge parity report}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/phase2_sft_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-12000}"
SAVE_EVERY="${SAVE_EVERY:-0}"
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
QA_TO_TIP="${QA_TO_TIP:-15}"
TIP_FEATURE_MODE="${TIP_FEATURE_MODE:-online_qwen}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
TRAIN_MICRO_BATCH_SIZE_PER_GPU="${TRAIN_MICRO_BATCH_SIZE_PER_GPU:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" logs/tmux

env PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/check_training_readiness.py \
  --phase phase2 \
  --manifest "${QA_MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --qwen-path "${QWEN_CHECKPOINT}" \
  --parity-report "${PARITY_REPORT}" \
  --tip-feature-mode "${TIP_FEATURE_MODE}" \
  --min-cross-frame-coverage "${MIN_CROSS_FRAME_COVERAGE:-0.01}" \
  --write "${OUTPUT_DIR}/readiness.json"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

TIP_ARGS=()
if [[ -n "${TIP_MANIFEST}" ]]; then
  TIP_ARGS+=(--tip-manifest "${TIP_MANIFEST}")
fi
DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  DEEPSPEED_ARGS+=(
    --deepspeed-config "${DEEPSPEED_CONFIG}"
    --train-micro-batch-size-per-gpu "${TRAIN_MICRO_BATCH_SIZE_PER_GPU}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  )
fi
CUDA_ENV=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_ENV="CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
fi

TRAIN_CMD="'${PYTHON_BIN}' scripts/train_sft.py --qa-manifest '${QA_MANIFEST}' ${TIP_ARGS[*]} --cache-root '${CACHE_ROOT}' --qwen-checkpoint '${QWEN_CHECKPOINT}' --phase1-checkpoint '${PHASE1_CHECKPOINT}' --output '${OUTPUT_DIR}' --steps '${STEPS}' --save-every '${SAVE_EVERY}' --device '${DEVICE}' --qa-to-tip '${QA_TO_TIP}' --tip-feature-mode '${TIP_FEATURE_MODE}' ${DEEPSPEED_ARGS[*]}"
if [[ "${NPROC_PER_NODE}" != "1" ]]; then
  TRAIN_CMD="'${PYTHON_BIN}' -m torch.distributed.run --standalone --nproc_per_node='${NPROC_PER_NODE}' scripts/train_sft.py --qa-manifest '${QA_MANIFEST}' ${TIP_ARGS[*]} --cache-root '${CACHE_ROOT}' --qwen-checkpoint '${QWEN_CHECKPOINT}' --phase1-checkpoint '${PHASE1_CHECKPOINT}' --output '${OUTPUT_DIR}' --steps '${STEPS}' --save-every '${SAVE_EVERY}' --device '${DEVICE}' --qa-to-tip '${QA_TO_TIP}' --tip-feature-mode '${TIP_FEATURE_MODE}' ${DEEPSPEED_ARGS[*]}"
fi

tmux new-session -d -s "${SESSION}" "cd '${PROJECT_ROOT}' && env PYTHONPATH='${PYTHONPATH_PREFIX}:${PYTHONPATH:-}' ${CUDA_ENV} ${TRAIN_CMD} > 'logs/tmux/${SESSION}.log' 2>&1; echo EXIT_CODE=\$? >> 'logs/tmux/${SESSION}.log'"

echo "started ${SESSION}"
echo "output: ${OUTPUT_DIR}"
echo "log: ${PROJECT_ROOT}/logs/tmux/${SESSION}.log"
