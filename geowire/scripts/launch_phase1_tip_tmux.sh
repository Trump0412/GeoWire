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

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" logs/tmux

"${PYTHON_BIN}" scripts/check_training_readiness.py \
  --phase phase1 \
  --manifest "${MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --write "${OUTPUT_DIR}/readiness.json"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" "cd '${PROJECT_ROOT}' && '${PYTHON_BIN}' scripts/train_tip.py --manifest '${MANIFEST}' --cache-root '${CACHE_ROOT}' --output '${OUTPUT_DIR}' --steps '${STEPS}' --device '${DEVICE}' > 'logs/tmux/${SESSION}.log' 2>&1; echo EXIT_CODE=\$? >> 'logs/tmux/${SESSION}.log'"

echo "started ${SESSION}"
echo "output: ${OUTPUT_DIR}"
echo "log: ${PROJECT_ROOT}/logs/tmux/${SESSION}.log"
