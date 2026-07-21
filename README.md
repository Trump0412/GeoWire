# Georoute

Georoute is a geometry-controlled cross-frame semantic transport module for
video and multi-view spatial reasoning. It builds sparse correspondence graphs
from frozen geometry features and transports visual-language token features
along those graph edges before the language model consumes the visual tokens.

## What Is Included

- Sparse correspondence graph construction and validation utilities.
- A lightweight semantic transport module with graph-control diagnostics.
- Phase 1 topology intervention pretraining.
- Phase 2 spatial instruction tuning with interleaved topology retention.
- Lightweight smoke tests and prediction-file evaluation helpers.

Large assets are not stored in this repository. Model weights, datasets,
checkpoints, cache files, generated runs, logs, and benchmark media should be
kept outside Git and passed through command-line arguments or environment
variables.

## Installation

```bash
cd geowire
python -m pip install -e ".[dev]"
python -m pytest -q
```

The source package is stored under `geowire/` for compatibility with the
training scripts.

## Quick Smoke

```bash
cd geowire
python scripts/smoke_all.py --output runs/smoke_local
```

The smoke path uses synthetic toy data. It verifies environment inspection,
coordinate contracts, sparse graph construction, cached Phase 1 training,
prediction-file evaluation, and unit tests. It does not require private data,
model weights, or benchmark media.

## Expected External Resources

Use local paths appropriate for your machine:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export QWEN_CHECKPOINT=/path/to/Qwen3-VL-2B-Instruct
export VGGT_CHECKPOINT=/path/to/VGGT-1B
export GEOROUTE_DATA_ROOT=/path/to/datasets
export GEOROUTE_CACHE_ROOT=/path/to/cache/georoute
```

The scripts also accept the corresponding command-line arguments directly.

## Phase 1: Topology Intervention Pretraining

Phase 1 trains only the graph transport module. The objective masks selected
visual tokens and asks the model to recover their semantic features using the
approved sparse correspondence graph.

```bash
cd geowire
MANIFEST=/path/to/phase1_manifest.jsonl \
CACHE_ROOT=/path/to/cache/georoute/phase1 \
QWEN_CHECKPOINT=/path/to/Qwen3-VL-2B-Instruct \
STEPS=30000 \
NPROC_PER_NODE=1 \
./scripts/launch_phase1_tip_tmux.sh
```

Each cached clip should contain `token_layout.safetensors`, `graph_coo.npz`,
and `metadata.json`. `semantic_tokens.safetensors` is optional when
`TIP_FEATURE_MODE=online_qwen`.

## Phase 2: Spatial Instruction Tuning

Phase 2 trains the language-model LoRA modules together with the graph
transport module. QA batches are interleaved with topology-retention batches to
discourage the transport module from collapsing into a generic residual adapter.

```bash
cd geowire
QA_MANIFEST=/path/to/phase2_qa_manifest.jsonl \
TIP_MANIFEST=/path/to/phase2_tip_manifest.jsonl \
CACHE_ROOT=/path/to/cache/georoute/phase2 \
QWEN_CHECKPOINT=/path/to/Qwen3-VL-2B-Instruct \
PHASE1_CHECKPOINT=/path/to/phase1/adapter.pt \
PARITY_REPORT=/path/to/passing_parity_report.json \
QA_TO_TIP=15 \
NPROC_PER_NODE=8 \
DEEPSPEED_CONFIG=configs/deepspeed_zero2.json \
./scripts/launch_phase2_sft_tmux.sh
```

## Repository Layout

```text
geowire/geowire/        core dataclasses, graph utilities, models, and training helpers
geowire/scripts/        smoke, cache, graph, training, and evaluation entrypoints
geowire/configs/        portable example configs
geowire/tests/          lightweight unit and smoke tests
```

## Reproducibility Notes

- Set all model, dataset, cache, and output paths explicitly for real runs.
- Keep generated outputs under `runs/`, `logs/`, `cache/`, or another ignored
  directory.
- Use the provided graph-control diagnostics before reporting full training
  results.
