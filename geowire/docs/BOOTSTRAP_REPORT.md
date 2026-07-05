# Bootstrap Report

Generated after the first deployment on `10.99.8.14` (`node214`).

## GPU Status

The machine has 8 x NVIDIA A100 80GB PCIe.

Observed usage during bootstrap:

| GPU | Memory | Utilization | Notes |
|---:|---:|---:|---|
| 0 | ~57.4 GB / 80 GB | 100% | FlowDA training process |
| 1 | ~57.3 GB / 80 GB | 100% | FlowDA training process |
| 2 | ~57.3 GB / 80 GB | 100% | FlowDA training process |
| 3 | ~72.3 GB / 80 GB | 0% | vLLM worker resident |
| 4 | ~57.2 GB / 80 GB | 100% | FlowDA training process |
| 5 | ~71.7 GB / 80 GB | 0% | vLLM worker resident |
| 6 | ~0 GB / 80 GB | 0% | Best candidate for smoke runs |
| 7 | ~1.0 GB / 80 GB | 0% | Light residual process |

## Conda Environments

The relevant conda root is:

```text
/mnt/guojh/lq/new/conda/miniforge3
```

Known envs:

```text
/mnt/guojh/lq/new/conda/envs/geothinker
/mnt/guojh/lq/new/conda/envs/geobridge-verl
```

The active GeoWire bootstrap env is:

```text
/mnt/guojh/lq/new/conda/envs/geothinker
```

Key versions after dependency completion:

```text
Python 3.10.20
PyTorch 2.5.1+cu124
CUDA 12.4
Transformers 4.50.0
Accelerate 1.4.0
PEFT 0.18.1
Flash-Attn 2.7.4.post1
Hydra-Core 1.3.4
OmegaConf 2.3.1
pytest 9.1.1
ruff 0.15.20
```

## Existing Shared Resources

```text
/mnt/guojh/lq/new/datasets/benchmarks/VSI-Bench       12G
/mnt/guojh/lq/new/datasets/benchmarks/MMSI_Bench      2.6G
/mnt/guojh/lq/new/datasets/benchmarks/ViewSpatial     3.7G
/mnt/guojh/lq/new/datasets/benchmarks/DSR-Bench       75M
/mnt/guojh/lq/new/weights/base_models/VGGT-1B         9.4G
/mnt/guojh/lq/new/weights/base_models/Qwen2.5-VL-7B-Instruct 16G
```

## Newly Downloaded Resource

Downloaded through `https://hf-mirror.com`:

```text
/mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct 8.3G
```

Files present:

```text
chat_template.json
config.json
generation_config.json
merges.txt
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
preprocessor_config.json
README.md
tokenizer_config.json
tokenizer.json
video_preprocessor_config.json
vocab.json
```

## Verified Commands

```bash
python -m ruff check .
python -m pytest -q
python scripts/inspect_environment.py --write runs/env_report.json
python scripts/generate_toy_scene.py --output assets/toy_scene --manifest assets/toy_scene/manifest.jsonl
python scripts/verify_coordinate_contract.py --write runs/coordinate_contract.json
python scripts/build_graphs.py --output runs/phase0_graph
python scripts/verify_graph.py --graph runs/phase0_graph/graph_coo.npz
python scripts/train_tip.py --output runs/tip_debug
python scripts/audit_leakage.py --manifest assets/toy_scene/manifest.jsonl --write runs/leakage_audit.json
```

All above commands passed.

## Current Gate

`geothinker` currently has `transformers==4.50.0`. The downloaded Qwen3-VL-4B
checkpoint is present, but the Qwen3-VL bridge is intentionally gated until
Transformers is upgraded to a Qwen3-VL-compatible release and `alpha=0` parity
passes.

No Phase 2 SFT is claimed before that parity test.
