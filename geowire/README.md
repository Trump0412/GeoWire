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
- leakage audit;
- Qwen/VGGT inspection and resource download entrypoints.

Not yet claimed:

- real VGGT cache generation;
- Qwen3-VL logits parity;
- Phase 1 full TIP training;
- Phase 2 spatial SFT.

Those are intentionally gated behind environment/resource inspection and parity
tests.
