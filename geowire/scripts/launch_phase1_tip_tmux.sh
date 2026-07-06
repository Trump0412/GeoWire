#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/guojh/lq/new/GeoWire/geowire}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geothinker/bin/python}"
SESSION="${SESSION:-geowire_phase1_tip}"
MANIFEST="${MANIFEST:?set MANIFEST to a GeoWire JSONL manifest}"
CACHE_ROOT="${CACHE_ROOT:?set CACHE_ROOT to a GeoWire cache root}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/phase1_tip_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-30000}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_REAL_CACHE="${REQUIRE_REAL_CACHE:-1}"
MIN_CROSS_FRAME_COVERAGE="${MIN_CROSS_FRAME_COVERAGE:-0.01}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-${PROJECT_ROOT}}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" logs/tmux

READINESS_ARGS=()
if [[ "${REQUIRE_REAL_CACHE}" == "1" ]]; then
  READINESS_ARGS+=(--require-real-cache --min-cross-frame-coverage "${MIN_CROSS_FRAME_COVERAGE}")
fi

"${PYTHON_BIN}" scripts/check_training_readiness.py \
  --phase phase1 \
  --manifest "${MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  "${READINESS_ARGS[@]}" \
  --write "${OUTPUT_DIR}/readiness.json"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

TRAIN_CMD="'${PYTHON_BIN}' scripts/train_tip.py --manifest '${MANIFEST}' --cache-root '${CACHE_ROOT}' --output '${OUTPUT_DIR}' --steps '${STEPS}' --device '${DEVICE}'"
if [[ "${NPROC_PER_NODE}" != "1" ]]; then
  TRAIN_CMD="'${PYTHON_BIN}' -m torch.distributed.run --standalone --nproc_per_node='${NPROC_PER_NODE}' scripts/train_tip.py --manifest '${MANIFEST}' --cache-root '${CACHE_ROOT}' --output '${OUTPUT_DIR}' --steps '${STEPS}' --device '${DEVICE}'"
fi

tmux new-session -d -s "${SESSION}" "cd '${PROJECT_ROOT}' && env PYTHONPATH='${PYTHONPATH_PREFIX}:${PYTHONPATH:-}' ${TRAIN_CMD} > 'logs/tmux/${SESSION}.log' 2>&1; echo EXIT_CODE=\$? >> 'logs/tmux/${SESSION}.log'"

echo "started ${SESSION}"
echo "output: ${OUTPUT_DIR}"
echo "log: ${PROJECT_ROOT}/logs/tmux/${SESSION}.log"
