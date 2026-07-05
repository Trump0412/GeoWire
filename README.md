# GeoWire

Geometry-wired cross-frame semantic transport for video spatial reasoning.

This Git repository is rooted at `/home/chenbp/GeoWire`. The Python package and
training scaffold live in `geowire/`. The three top-level design documents are
kept at the repository root and mirrored into `geowire/docs/`:

- `GeoWire_Master_Design (2).md`
- `GeoWire_Codex_Implementation_Scaffold (1).md`
- `GeoWire_Training_Protocol.md`

## Quick Smoke

```bash
cd geowire
python scripts/smoke_all.py --output runs/smoke_local
```

The smoke path intentionally uses a synthetic toy cache. It verifies software
plumbing only: environment report, coordinate contract, toy cache, sparse graph,
cached TIP training, prediction-file evaluation, and unit tests. Real VGGT/Qwen
training remains gated behind resource inspection, graph overlays, and Qwen
bridge parity.

Latest server software-training check:

```text
2026-07-06
server: /mnt/guojh/lq/new/GeoWire/geowire
gpu: CUDA_VISIBLE_DEVICES=6
output: runs/phase1_tip_step50_gpu6
result: cached Phase 1 TIP toy run completed step 50
checkpoint: runs/phase1_tip_step50_gpu6/geowire_adapter.pt
```

This is the current step50 engineering gate. It proves that the cached Phase 1
training entrypoint can run on an idle A100 and save metrics/checkpoints. It is
not claimed as real GeoWire paper training until real VGGT/Qwen caches and
bridge parity pass.

Latest real-backend server gates:

```text
real cache smoke: runs/real_backend_smoke_cache
real graph smoke: track_edges=32, projective_edges=138, self_edges=160
phase1 real smoke: runs/real_backend_phase1_tip_smoke, steps=2
qwen bridge parity: runs/qwen_bridge_parity/real_backend_smoke_report.json, max_abs_logit_diff=0.0
phase2 real smoke: runs/phase2_sft_smoke_real_backend_4step, schedule QA/QA/QA/TIP
```

Full training now depends on real manifests/cache scale-up plus graph visual
audit and threshold calibration, not on missing code stubs.

## Phase 1 Launch

Once real manifests and cache files are ready:

```bash
cd /mnt/guojh/lq/new/GeoWire/geowire
MANIFEST=/path/to/train_manifest.jsonl \
CACHE_ROOT=/mnt/guojh/lq/new/cache/geowire/phase1 \
STEPS=30000 \
SESSION=geowire_phase1_tip \
./scripts/launch_phase1_tip_tmux.sh
```

The cache for each clip must contain `token_layout.safetensors`,
`semantic_tokens.safetensors`, `graph_coo.npz`, and `metadata.json`.

## Shared Server Layout

Target clone path:

```text
/mnt/guojh/lq/new/GeoWire
```

Reusable shared resources are expected under:

```text
/mnt/guojh/lq/new/conda/envs/geothinker
/mnt/guojh/lq/new/weights/base_models
/mnt/guojh/lq/new/datasets
/mnt/guojh/lq/new/cache/geowire
```

## Sync Policy

Commit code, configs, scripts, docs, and small tests. Do not commit datasets,
model weights, checkpoints, generated `runs/`, or benchmark media.
