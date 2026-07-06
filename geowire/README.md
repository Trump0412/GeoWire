# GeoWire

Geometry-wired cross-frame semantic transport for video spatial reasoning.

This package follows the top-level design contract in `docs/MASTER_DESIGN.md`,
the implementation scaffold in `docs/CODEX_IMPLEMENTATION_SCAFFOLD.md`, and the
v0.2 training protocol in `docs/TRAINING_PROTOCOL.md`.
The first milestone is a verifiable engineering loop, not full SFT:

1. inspect the environment and pinned resources;
2. verify pixel-center coordinate contracts;
3. build sparse correspondence graphs;
4. run sparse semantic transport and TIP debug objectives;
5. only then inspect the Qwen3-VL bridge.

## Remote Paths

The current shared machine layout is expected to be:

```text
/mnt/guojh/lq/new/GeoWire
/mnt/guojh/lq/new/conda/envs/geothinker
/mnt/guojh/lq/new/weights/base_models
/mnt/guojh/lq/new/datasets/benchmarks
```

Node218 (`10.99.8.18`) uses the same shared root via:

```text
10.99.3.13:/NAS/CAPFS/data/guojh -> /mnt/guojh
runtime user: leiqi
```

## Quick Start

```bash
cd /mnt/guojh/lq/new/GeoWire/geowire
/mnt/guojh/lq/new/conda/envs/geothinker/bin/python -m pip install -e .
/mnt/guojh/lq/new/conda/envs/geothinker/bin/python scripts/inspect_environment.py --write runs/env_report.json
/mnt/guojh/lq/new/conda/envs/geothinker/bin/python -m pytest -q
```

Generate the synthetic toy scene and run the first checks:

```bash
python scripts/generate_toy_scene.py --output assets/toy_scene --manifest assets/toy_scene/manifest.jsonl
python scripts/verify_coordinate_contract.py --write runs/coordinate_contract.json
python scripts/cache_vggt.py --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache --backend toy
python scripts/build_graphs.py --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache --output runs/phase0_graph
python scripts/check_training_readiness.py --phase phase1 --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache
python scripts/train_tip.py --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache --output runs/tip_debug --steps 2
```

Or run the bundled smoke:

```bash
python scripts/smoke_all.py --output runs/smoke_local
```

Download pinned resources through the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_resources.py model Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct
```

## Distributed Launch

The Phase 1 and Phase 2 launch scripts support single-node torchrun via
`NPROC_PER_NODE`. For the current node218 allocation, GPUs 0-5 are the safe
default while GPUs 6-7 are occupied by another job:

```bash
cd /mnt/guojh/lq/new/GeoWire/geowire
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
QA_MANIFEST=/path/to/phase2_qa_manifest.jsonl \
TIP_MANIFEST=/path/to/phase2_tip_manifest.jsonl \
CACHE_ROOT=/mnt/guojh/lq/new/cache/geowire/phase2 \
PHASE1_CHECKPOINT=/path/to/geowire_adapter.pt \
PARITY_REPORT=/path/to/passing_parity_report.json \
./scripts/launch_phase2_sft_tmux.sh
```

Use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8` when all eight
cards are free. The current distributed trainer uses replicated model weights
with gradient all-reduce, not ZeRO/FSDP sharding.

## Current Scope

Implemented now:

- repository skeleton and reproducibility scripts;
- immutable core dataclasses;
- pixel-center affine transforms;
- Qwen token layout approximation for multi-image mode;
- sparse graph merge, top-k, normalization and NPZ IO;
- sparse transport and GeoWire blocks;
- TIP v0.2 debug losses and graph controls;
- cached Phase 1 TIP training entrypoint;
- Phase 1 readiness and tmux launch scripts;
- prediction JSONL evaluator for benchmark reports;
- leakage audit;
- Qwen/VGGT inspection and resource download entrypoints.
- real Qwen3-VL semantic-token cache generation;
- real VGGT geometry/tracks cache generation;
- real graph construction from track and projective candidates;
- Qwen3-VL image-feature-path bridge with alpha-zero logits parity;
- Phase 2 SFT entrypoint for Qwen LoRA + GeoWire with `3 QA : 1 TIP`;
- single-node torchrun support for Phase 1 and Phase 2.

Not yet claimed:

- Phase 1 full TIP training;
- Phase 2 spatial SFT.

Those are intentionally gated behind full real manifests, graph calibration,
leakage audit, and benchmark prediction runs.
