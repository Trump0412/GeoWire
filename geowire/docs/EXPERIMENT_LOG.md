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

Node218 shared mount and distributed smoke:

```text
date: 2026-07-06
host: 10.99.8.18
mount: 10.99.3.13:/NAS/CAPFS/data/guojh -> /mnt/guojh
project: /mnt/guojh/lq/new/GeoWire/geowire
runtime user: leiqi, UID/GID 1012
python: /mnt/guojh/lq/new/conda/envs/geothinker/bin/python
gpu: 8 x A100 80GB visible
distributed support: torch.distributed.run with rank-0 checkpoint/log writes
phase1 ddp smoke: runs/ddp_tip_cpu_smoke, 2 CPU processes
phase2 2-gpu smoke: runs/node218_phase2_sft_torchrun2_smoke_new
phase2 8-gpu smoke: runs/node218_phase2_sft_torchrun8_smoke
phase2 8-gpu world_size: 8
phase2 8-gpu final step: 4
phase2 8-gpu final mode: tip
phase2 8-gpu final_loss: 0.5104734301567078
note: GPUs 6-7 were occupied by an unrelated job after the smoke; use GPUs 0-5 until they are free
```

Node218 ZeRO-2 Phase 2 smoke:

```text
date: 2026-07-07
host: 10.99.8.18
command: torch.distributed.run --standalone --nproc_per_node=2 scripts/train_sft.py --deepspeed-config configs/deepspeed_zero2.json
output: runs/node218_phase2_sft_zero2_torchrun2_smoke
world_size: 2
deepspeed_enabled: true
schedule: QA, QA, QA, TIP
final_step: 4
final_mode: tip
final_loss: 0.5078125
adapter: runs/node218_phase2_sft_zero2_torchrun2_smoke/phase2_adapters.pt
```

LLaVA-Hound real-data smoke:

```text
date: 2026-07-07
host: 10.99.8.18
source_json: /mnt/guojh/lq/new/local_mirror/myproject/spatial4nips/data/train/llava_hound_64k.json
full_manifest: /mnt/guojh/lq/new/datasets/manifests/llava_hound.jsonl
full_manifest_records: 63750
sampled_file_audit: 128 / 128 records present
smoke_manifest: /mnt/guojh/lq/new/datasets/manifests/llava_hound_smoke8.jsonl
smoke_cache: /mnt/guojh/lq/new/cache/geowire/llava_hound_smoke8
smoke_real_cache_records: 8
smoke_graph_records: 8
phase1_readiness: runs/llava_smoke8_graph/phase1_readiness.json, passed
phase1_command: torch.distributed.run --standalone --nproc_per_node=2 scripts/train_tip.py --steps 20
phase1_output: runs/llava_hound_smoke8_phase1_tip_20step
phase1_final_loss: 0.3427661061286926
phase1_full_rec: 0.345902681350708
phase1_self_loop_rec: 0.736224889755249
phase1_random_graph_rec: 0.5384038686752319
phase1_shuffled_graph_rec: 0.5328946113586426
qwen_parity: runs/qwen_bridge_parity/llava_hound_smoke8_report.json, max_abs_logit_diff=0.0
phase2_readiness: runs/llava_smoke8_graph/phase2_readiness.json, passed
phase2_command: torch.distributed.run --standalone --nproc_per_node=2 scripts/train_sft.py --steps 4 --deepspeed-config configs/deepspeed_zero2.json
phase2_output: runs/llava_hound_smoke8_phase2_sft_zero2_4step
phase2_final_mode: tip
phase2_final_loss: 0.5
```

Online-Qwen TIP pilot:

```text
date: 2026-07-07
host: 10.99.8.18
commit: 1c1c697
mode: online_qwen
purpose: avoid random NFS reads of large semantic_tokens.safetensors during TIP
required_cache_files: token_layout.safetensors, graph_coo.npz, metadata.json
manifest: /mnt/guojh/lq/new/datasets/manifests/llava_hound_smoke8.jsonl
cache: /mnt/guojh/lq/new/cache/geowire/llava_hound_smoke8
phase1_command: torch.distributed.run --standalone --nproc_per_node=8 scripts/train_tip.py --tip-feature-mode online_qwen --steps 16
phase1_output: runs/online_qwen_llava_smoke8_phase1_tip_16step_8gpu
phase1_elapsed_seconds: 67
phase1_final_loss: 0.46381279826164246
phase2_command: torch.distributed.run --standalone --nproc_per_node=8 scripts/train_sft.py --tip-feature-mode online_qwen --qa-to-tip 15 --deepspeed-config configs/deepspeed_zero2.json --steps 16
phase2_output: runs/online_qwen_llava_smoke8_phase2_sft_zero2_16step_8gpu
phase2_elapsed_seconds: 41
phase2_final_mode: tip
phase2_final_loss: 0.47265625
phase2_readiness: require_real_cache=false, tip_feature_mode=online_qwen
```
