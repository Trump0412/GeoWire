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
- Real Phase 0 cache generation is implemented for Qwen3-VL semantic tokens plus VGGT geometry/tracks.
- Real graph building can use track and projective VGGT candidates with configurable precision/coverage thresholds.
- Qwen3-VL bridge is implemented as an image-feature-path wrapper and has a real alpha-zero parity script.
- Phase 2 SFT entrypoint can train Qwen LoRA + GeoWire with a `3 QA : 1 TIP` schedule.
- `scripts/launch_phase2_sft_tmux.sh` starts Phase 2 after real cache and parity readiness checks.
- `scripts/evaluate.py` scores prediction JSONL files against eval configs and writes accuracy reports.

## Latest Local Smoke

Command:

```bash
cd /home/chenbp/GeoWire/geowire
python scripts/smoke_all.py --output runs/smoke_local
```

Result:

```text
29 passed, 1 skipped
coordinate contract passed
toy graph row sums min/max = 1.0 / 1.0
cached TIP checkpoint written
eval report written with toy accuracy = 0.5
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
29 passed, 1 skipped
8 x A100 80GB visible
toy graph row sums min/max = 1.0 / 1.0
phase1 readiness passed
cached TIP checkpoint written:
/mnt/guojh/lq/new/GeoWire/geowire/runs/smoke_eval_server/tip_debug/geowire_adapter.pt
eval report written:
/mnt/guojh/lq/new/GeoWire/geowire/runs/smoke_eval_server/eval_report.json
```

## Latest Server Step50 Training Gate

Path:

```text
/mnt/guojh/lq/new/GeoWire/geowire
```

Command:

```bash
CUDA_VISIBLE_DEVICES=6 \
/mnt/guojh/lq/new/conda/envs/geothinker/bin/python scripts/train_tip.py \
  --manifest assets/toy_scene/manifest.jsonl \
  --cache-root runs/step50_toy_cache \
  --output runs/phase1_tip_step50_gpu6 \
  --steps 50 \
  --device cuda
```

Result:

```text
completed step: 50
output: runs/phase1_tip_step50_gpu6
checkpoint: runs/phase1_tip_step50_gpu6/geowire_adapter.pt
metrics: runs/phase1_tip_step50_gpu6/metrics.json
loss: 0.6762770414352417
eval_full_rec: 0.6607329845428467
eval_random_graph_gap: 0.15737617015838623
eval_shuffled_graph_gap: 0.2560940384864807
```

This validates the software training loop on an idle A100. It remains a toy
cached Phase 1 gate, not real paper training.

## Server Resource Check

- `Qwen3-VL-4B-Instruct`: present at `/mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct` (`8.3G`).
- `VGGT-1B`: present at `/mnt/guojh/lq/new/weights/base_models/VGGT-1B` (`9.4G`).
- Shared conda env still has `transformers==4.50.0`; GeoWire uses project-local overlay
  `/mnt/guojh/lq/new/GeoWire/.deps/transformers_4_57_6` for Qwen3-VL.
- VGGT source is present at `/mnt/guojh/lq/new/GeoWire/third_party/vggt`, commit `a288dd0`.

## Latest Real Backend Gate

Server path:

```text
/mnt/guojh/lq/new/GeoWire/geowire
```

Real cache smoke:

```text
manifest: assets/real_backend_smoke_scene/manifest.jsonl
cache: runs/real_backend_smoke_cache
backend: real
semantic tokens: runs/real_backend_smoke_cache/toy_scene_clip_000/semantic_tokens.safetensors
geometry: runs/real_backend_smoke_cache/toy_scene_clip_000/geometry.safetensors
graph: track_edges=32, projective_edges=138, self_edges=160
phase1 readiness: passed with --require-real-cache and min_cross_frame_coverage=0.01
```

Real Phase 1 training smoke:

```text
output: runs/real_backend_phase1_tip_smoke
steps: 2
checkpoint: runs/real_backend_phase1_tip_smoke/geowire_adapter.pt
eval_random_graph_gap: 0.02674686908721924
eval_shuffled_graph_gap: 0.04390311241149902
```

Qwen bridge parity:

```text
report: runs/qwen_bridge_parity/real_backend_smoke_report.json
passed: true
max_abs_logit_diff: 0.0
```

Real Phase 2 SFT smoke:

```text
output: runs/phase2_sft_smoke_real_backend_4step
schedule: 3 QA steps then 1 TIP step
final step: 4
final mode: tip
phase2_adapters.pt written
```

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

## Evaluation Entry

Given a prediction JSONL with at least `prediction` and `answer` fields:

```bash
python scripts/evaluate.py \
  --config configs/eval_vsi.yaml \
  --predictions /path/to/predictions.jsonl \
  --write runs/eval/report.json
```

## Gated Before Full Real Training

- Full training manifests must be built from the real SPAR/LLaVA-Hound/XVR pools with scene-level leakage audit.
- Graph thresholds must be calibrated on real clips with visual overlay audit; the smoke used loose thresholds only to verify the engineering path.
- Phase 1 should not advance to paper Phase 2 claims until full graph beats self/random/shuffled controls on a real validation subset.
- Benchmark generation wrappers still need real prediction runs for VSI/MMSI/ViewSpatial; the scorer already exists.
