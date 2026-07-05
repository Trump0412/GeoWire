# Experiment Log

## M0 Bootstrap

Status: software smoke passed locally on 2026-07-06.

- Environment inspection command: `python scripts/inspect_environment.py --write runs/env_report.json`
- Coordinate contract command: `python scripts/verify_coordinate_contract.py --write runs/coordinate_contract.json`
- TIP debug command: `python scripts/train_tip.py --output runs/tip_debug`
- Full smoke command: `python scripts/smoke_all.py --output runs/smoke_local`

Latest local smoke:

```text
29 passed, 1 skipped
toy cache generated
toy graph normalized
phase1 readiness passed on toy cache
cached TIP checkpoint written
prediction JSONL evaluator wrote an eval report
coordinate roundtrip max error < 0.5 px
```

Server smoke:

```text
path: /mnt/guojh/lq/new/GeoWire/geowire
python: /mnt/guojh/lq/new/conda/envs/geothinker/bin/python
result: 29 passed, 1 skipped
gpu: 8 x A100 80GB visible
phase1 readiness: passed on toy cache
cached tip checkpoint: runs/smoke_eval_server/tip_debug/geowire_adapter.pt
eval report: runs/smoke_eval_server/eval_report.json
```

Server step50 engineering gate:

```text
date: 2026-07-06
path: /mnt/guojh/lq/new/GeoWire/geowire
python: /mnt/guojh/lq/new/conda/envs/geothinker/bin/python
gpu: CUDA_VISIBLE_DEVICES=6
manifest: assets/toy_scene/manifest.jsonl
cache: runs/step50_toy_cache
output: runs/phase1_tip_step50_gpu6
result: completed step 50
checkpoint: runs/phase1_tip_step50_gpu6/geowire_adapter.pt
metrics: runs/phase1_tip_step50_gpu6/metrics.json
loss: 0.6762770414352417
eval_full_rec: 0.6607329845428467
eval_random_graph_gap: 0.15737617015838623
eval_shuffled_graph_gap: 0.2560940384864807
```

Qwen3-VL inspection:

```text
server transformers: 4.50.0
status: gated; install or vendor a Qwen3-VL-compatible Transformers build before bridge parity
```

No Phase 1 or Phase 2 result is claimed until VGGT cache, graph overlays and Qwen bridge parity pass.
