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
shared env transformers: 4.50.0
GeoWire overlay transformers: /mnt/guojh/lq/new/GeoWire/.deps/transformers_4_57_6
overlay status: transformers 4.57.6 import ok
VGGT source: /mnt/guojh/lq/new/GeoWire/third_party/vggt @ a288dd0
```

Real backend smoke:

```text
date: 2026-07-06
manifest: assets/real_backend_smoke_scene/manifest.jsonl
cache: runs/real_backend_smoke_cache
graph: runs/real_backend_smoke_graph_loose
graph stats: track_edges=32, projective_edges=138, self_edges=160
readiness: phase1 --require-real-cache passed with min_cross_frame_coverage=0.01
note: loose graph thresholds were used for engineering smoke; real training requires calibrated thresholds and visual audit
```

Real Phase 1 smoke:

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
steps: 4
schedule: QA, QA, QA, TIP
final_loss: 0.5104734301567078
adapter: runs/phase2_sft_smoke_real_backend_4step/phase2_adapters.pt
```
