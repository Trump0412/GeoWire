# GeoWire Progress

Last updated: 2026-07-06

## Current Status

- Local GitHub SSH auth is available for `Trump0412`.
- Local project root is `/home/chenbp/GeoWire`.
- Python package root is `/home/chenbp/GeoWire/geowire`.
- The v0.2 master design, implementation scaffold and training protocol are
  mirrored into `geowire/docs/`.

## Implemented Software Loop

- Repository-local scripts can run before `pip install -e .` via `scripts/_bootstrap.py`.
- `make smoke` and `scripts/smoke_all.py` run the current local smoke path.
- Synthetic toy scene generation is available.
- Coordinate contract roundtrip check passes for 16:9, 9:16, 4:3 and tall inputs.
- Toy cache generation writes metadata, frame transforms, qwen input metadata and token layout.
- Graph construction writes normalized sparse COO graphs from cached token layouts.
- Toy/real cache contract now includes `semantic_tokens.safetensors`; Phase 1 consumes this as frozen Qwen semantic-token cache.
- Sparse transport and GeoWire blocks are implemented with alpha initialized to zero.
- TIP v0.2 losses are implemented: recovery + substitution + optional keep.
- Non-edge isolation is retained as a diagnostic, not a training loss.
- Random/shuffled/self-loop graph diagnostics are available for Phase 1 debug.
- Cached Phase 1 entrypoint can train GeoWire from `manifest + cache_root`, saving `geowire_adapter.pt`, `metrics.jsonl`, `metrics.json` and `trainer_state.json`.
- `scripts/check_training_readiness.py` validates Phase 1 cache completeness.
- `scripts/launch_phase1_tip_tmux.sh` starts Phase 1 in tmux after readiness checks.

## Latest Local Smoke

Command:

```bash
cd /home/chenbp/GeoWire/geowire
python scripts/smoke_all.py --output runs/smoke_local
```

Result:

```text
26 passed, 1 skipped
coordinate contract passed
toy graph row sums min/max = 1.0 / 1.0
cached TIP checkpoint written
```

## Latest Server Smoke

Path:

```text
/mnt/guojh/lq/new/GeoWire
```

Command:

```bash
cd /mnt/guojh/lq/new/GeoWire/geowire
/mnt/guojh/lq/new/conda/envs/geothinker/bin/python scripts/smoke_all.py --output runs/smoke_server
```

Result:

```text
26 passed, 1 skipped
8 x A100 80GB visible
toy graph row sums min/max = 1.0 / 1.0
phase1 readiness passed
cached TIP checkpoint written:
/mnt/guojh/lq/new/GeoWire/geowire/runs/smoke_cached_phase1_server/tip_debug/geowire_adapter.pt
```

## Server Resource Check

- `Qwen3-VL-4B-Instruct`: present at `/mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct` (`8.3G`).
- `VGGT-1B`: present at `/mnt/guojh/lq/new/weights/base_models/VGGT-1B` (`9.4G`).
- Current server `transformers==4.50.0`; Qwen3-VL bridge inspection is gated until a Qwen3-VL-compatible Transformers build is installed or vendored.

## Phase 1 Launch Template

After real cached clips have `token_layout.safetensors`, `semantic_tokens.safetensors`,
`graph_coo.npz` and `metadata.json`, launch:

```bash
cd /mnt/guojh/lq/new/GeoWire/geowire
MANIFEST=/path/to/train_manifest.jsonl \
CACHE_ROOT=/mnt/guojh/lq/new/cache/geowire/phase1 \
STEPS=30000 \
SESSION=geowire_phase1_tip \
./scripts/launch_phase1_tip_tmux.sh
```

## Gated Before Real Training

- Real VGGT cache generation requires pinned `third_party/vggt`, VGGT-1B weights and visual graph audit.
- Qwen3-VL bridge still requires installed compatible Transformers/Qwen3-VL source inspection and alpha=0 parity.
- Phase 2 SFT remains blocked until Phase 1 diagnostics prove full graph beats random/shuffled controls.
