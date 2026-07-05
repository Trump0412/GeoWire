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
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" logs/tmux

env PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" "${PYTHON_BIN}" scripts/check_training_readiness.py \
  --phase phase2 \
  --manifest "${QA_MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --qwen-path "${QWEN_CHECKPOINT}" \
  --parity-report "${PARITY_REPORT}" \
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
CUDA_ENV=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_ENV="CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
fi

tmux new-session -d -s "${SESSION}" "cd '${PROJECT_ROOT}' && env PYTHONPATH='${PYTHONPATH_PREFIX}:${PYTHONPATH:-}' ${CUDA_ENV} '${PYTHON_BIN}' scripts/train_sft.py --qa-manifest '${QA_MANIFEST}' ${TIP_ARGS[*]} --cache-root '${CACHE_ROOT}' --qwen-checkpoint '${QWEN_CHECKPOINT}' --phase1-checkpoint '${PHASE1_CHECKPOINT}' --output '${OUTPUT_DIR}' --steps '${STEPS}' --device '${DEVICE}' > 'logs/tmux/${SESSION}.log' 2>&1; echo EXIT_CODE=\$? >> 'logs/tmux/${SESSION}.log'"

echo "started ${SESSION}"
echo "output: ${OUTPUT_DIR}"
echo "log: ${PROJECT_ROOT}/logs/tmux/${SESSION}.log"
