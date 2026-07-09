# GeoWire Progress

Last updated: 2026-07-07

## Current Status

- Local GitHub SSH auth is available for `Trump0412`.
- Local project root is `/home/chenbp/GeoWire`.
- Python package root is `/home/chenbp/GeoWire/geowire`.
- The v0.2 master design, implementation scaffold and training protocol are
  mirrored into `geowire/docs/`.
- `GeoWire_Method_Training_Details.md` now records the pre-experiment method,
  loss, data, memory-utilization, ETA, and go/no-go gates.

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
- Phase 2 SFT entrypoint can train Qwen LoRA + GeoWire with a `15 QA : 1 TIP` schedule.
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
- `Qwen3-VL-2B-Instruct`: present and load-checked at
  `/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct`
  (`model.safetensors` size: 4,255,140,312 bytes).
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

## Node218 Distributed Gate

Node218 (`10.99.8.18`) is connected to the shared project/data root:

```text
10.99.3.13:/NAS/CAPFS/data/guojh -> /mnt/guojh
project: /mnt/guojh/lq/new/GeoWire
python: /mnt/guojh/lq/new/conda/envs/geothinker/bin/python
runtime user: leiqi, UID/GID 1012
gpu: 8 x A100 80GB visible
```

The training entrypoints now support `torch.distributed.run` through
`NPROC_PER_NODE` in the tmux launch scripts. The current implementation keeps a
full model replica on each GPU and averages gradients with `all_reduce`; it does
not use ZeRO/FSDP sharding.

DeepSpeed is installed in the shared conda environment (`deepspeed==0.18.7` on
both node214 and node218). Phase 2 can enable ZeRO-2 by setting
`DEEPSPEED_CONFIG=./configs/deepspeed_zero2.json`; this partitions optimizer
states/gradients but still keeps frozen Qwen weights replicated on each GPU.

Local distributed Phase 1 CPU smoke:

```text
output: runs/ddp_tip_cpu_smoke
processes: 2
result: checkpoint and metrics written by rank 0
```

Node218 Phase 2 distributed smoke:

```text
2 GPU output: runs/node218_phase2_sft_torchrun2_smoke_new
8 GPU output: runs/node218_phase2_sft_torchrun8_smoke
world_size: 8
final step: 4
final mode: tip
final loss: 0.5104734301567078
adapter: runs/node218_phase2_sft_torchrun8_smoke/phase2_adapters.pt
```

Node218 Phase 2 ZeRO-2 smoke:

```text
output: runs/node218_phase2_sft_zero2_torchrun2_smoke
world_size: 2
deepspeed_config: configs/deepspeed_zero2.json
deepspeed_enabled: true
final step: 4
final mode: tip
final loss: 0.5078125
adapter: runs/node218_phase2_sft_zero2_torchrun2_smoke/phase2_adapters.pt
```

LLaVA-Hound real-data smoke:

```text
source json: /mnt/guojh/lq/new/local_mirror/myproject/spatial4nips/data/train/llava_hound_64k.json
full manifest: /mnt/guojh/lq/new/datasets/manifests/llava_hound.jsonl
full manifest records: 63750
full manifest sampled file audit: 128 / 128 records present
smoke manifest: /mnt/guojh/lq/new/datasets/manifests/llava_hound_smoke8.jsonl
smoke cache: /mnt/guojh/lq/new/cache/geowire/llava_hound_smoke8
smoke cache records: 8 real Qwen/VGGT caches, 8 graphs
phase1 readiness: runs/llava_smoke8_graph/phase1_readiness.json, passed
phase2 readiness: runs/llava_smoke8_graph/phase2_readiness.json, passed
qwen parity: runs/qwen_bridge_parity/llava_hound_smoke8_report.json, max_abs_logit_diff=0.0
phase1 smoke: runs/llava_hound_smoke8_phase1_tip_20step, world_size=2, final step=20
phase1 final loss: 0.3427661061286926
phase1 graph controls: self=0.736224889755249, random=0.5384038686752319, shuffled=0.5328946113586426, full=0.345902681350708
phase2 ZeRO-2 smoke: runs/llava_hound_smoke8_phase2_sft_zero2_4step, world_size=2, final step=4
phase2 final mode/loss: tip / 0.5
```

Online-Qwen TIP mode:

```text
commit: 1c1c697
mode: TIP clean visual tokens are computed online through frozen Qwen visual forward
large cache avoided: semantic_tokens.safetensors no longer required for formal TIP/SFT training
required cache: token_layout.safetensors, graph_coo.npz, metadata.json
phase1 launcher default: TIP_FEATURE_MODE=online_qwen
phase2 launcher default: TIP_FEATURE_MODE=online_qwen
phase2 readiness with online mode: require_real_cache=false unless explicitly requested
```

Formal Qwen-free manifest build:

```text
date: 2026-07-07
session: geowire_manifest_build
status: completed
builder: scripts/build_formal_manifests.sh
root: /mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8
report: /mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8/formal_manifest_report.json
qa manifest: phase2_all.jsonl
qa rows: 1,336,497
qa unique clips: 370,380
tip manifest: phase2_tip_multiframe.jsonl
tip rows: 944,716
tip unique clips: 179,810
phase2 estimated steps at per-GPU batch 2 on 8 GPUs: 89,104
qwen dependency: none for this build
```

Source counts in `phase2_all.jsonl`:

| Source | Rows |
|---|---:|
| LLaVA-Hound | 63,750 |
| SPAR | 234,056 |
| VSI-590K | 590,667 |
| VLM3R-VSI | 205,456 |
| VLM3R-VST | 132,568 |
| JoyAI OpenSpatial | 100,000 |
| MindCube | 10,000 |

Source counts in `phase2_tip_multiframe.jsonl`:

| Source | Rows |
|---|---:|
| LLaVA-Hound | 63,750 |
| SPAR | 158,794 |
| VSI-590K | 374,148 |
| VLM3R-VSI | 205,456 |
| VLM3R-VST | 132,568 |
| MindCube | 10,000 |

Current formal decision:

```text
Do not launch the next long cache/training pass on Qwen3-VL-4B-Instruct.
Use the already validated 4B path only as a pilot reference.
Use the verified `/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct` path,
then rerun the memory ramp and launch the formal 8-card run from the 2B path.
```

Formal 2B cache queue:

```text
date: 2026-07-07
session: geowire_cache_qwen3vl2b_px376320_f8
status: paused after early cache; resume with --skip-existing after GPU ramps
manifest: /mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8/phase2_all.jsonl
cache root: /mnt/guojh/lq/new/cache/geowire/formal_qwen3vl2b_f8_px376320_grid
qwen checkpoint: /mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
image max pixels: 376320
shards: 24
gpu use: none
early count: 1,268 cached clips at 2026-07-07 15:17:33 CST
early throughput: about 5.5 clips/sec across 24 shards
early ETA: about 18-22 hours for 370,380 unique clips; expected completion on
  2026-07-08 morning CST if the current rate holds
```

Ramps from the 2B path:

```text
phase1 session: geowire_p1_tip_qwen3vl2b_ramp512_b4_px376320_step50_8gpu
phase1 status: passed, 50 steps, EXIT_CODE=0
phase1 setting: 8 GPUs, microbatch 4/GPU, online_qwen, GEOWIRE_IMAGE_MAX_PIXELS=376320
phase1 final loss: 0.28732481598854065
phase1 final step_seconds: 10.828280536457896
phase1 peak allocated/reserved: 7.92 GB / 14.29 GB on rank 0

phase2 session b2: geowire_p2_sft_qwen3vl2b_ramp512_b2_px376320_step10_8gpu_v3
phase2 b2 status: passed, 10 QA steps, EXIT_CODE=0
phase2 b2 setting: 8 GPUs, ZeRO-2, microbatch 2/GPU
phase2 b2 step_seconds: about 1.9-2.7 seconds
phase2 b2 peak allocated/reserved: 41.27 GB / 65.95 GB on rank 0

phase2 session b3: geowire_p2_sft_qwen3vl2b_ramp512_b3_px376320_step5_8gpu
phase2 b3 status: passed, 5 QA steps, EXIT_CODE=0
phase2 b3 setting: 8 GPUs, ZeRO-2, microbatch 3/GPU
phase2 b3 step_seconds: about 3.0-3.8 seconds
phase2 b3 peak allocated/reserved: 57.43 GB / 75.72 GB on rank 0
current recommendation: use phase2 microbatch 3/GPU for the first formal run;
  b4 is likely close to the OOM boundary without tighter length bucketing.
```

Node218 8-GPU online-Qwen pilot:

```text
date: 2026-07-07
session: geowire_online_qwen_pilot8
manifest: /mnt/guojh/lq/new/datasets/manifests/llava_hound_smoke8.jsonl
cache: /mnt/guojh/lq/new/cache/geowire/llava_hound_smoke8
phase1 output: runs/online_qwen_llava_smoke8_phase1_tip_16step_8gpu
phase1 world_size: 8
phase1 steps: 16
phase1 elapsed: 67 seconds including 8-rank Qwen load
phase1 final loss: 0.46381279826164246
phase2 output: runs/online_qwen_llava_smoke8_phase2_sft_zero2_16step_8gpu
phase2 world_size: 8
phase2 steps: 16
phase2 elapsed: 41 seconds including model/ZeRO initialization
phase2 qa_to_tip: 15
phase2 final mode/loss: tip / 0.47265625
gpu memory: Phase1 about 10-11 GB/GPU, Phase2 about 17-27 GB/GPU on the smoke clips
```

Current resource note: the formal manifest build does not use GPUs. Check
`nvidia-smi` immediately before the 2B cache/ramp launch and use all eight cards
if they remain free.

## Phase 1 Launch Template

After real cached clips have `token_layout.safetensors`, `semantic_tokens.safetensors`,
`graph_coo.npz` and `metadata.json`, launch:

```bash
cd /mnt/guojh/lq/new/GeoWire/geowire
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
MANIFEST=/path/to/train_manifest.jsonl \
CACHE_ROOT=/mnt/guojh/lq/new/cache/geowire/phase1 \
STEPS=30000 \
SESSION=geowire_phase1_tip \
./scripts/launch_phase1_tip_tmux.sh
```

## Phase 2 Launch Template

After Phase 1 has produced `geowire_adapter.pt` and Qwen alpha-zero parity has
passed:

```bash
cd /mnt/guojh/lq/new/GeoWire/geowire
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
DEEPSPEED_CONFIG=./configs/deepspeed_zero2.json \
QA_TO_TIP=15 \
QA_MANIFEST=/path/to/phase2_qa_manifest.jsonl \
TIP_MANIFEST=/path/to/phase2_tip_manifest.jsonl \
CACHE_ROOT=/mnt/guojh/lq/new/cache/geowire/phase2 \
PHASE1_CHECKPOINT=/path/to/geowire_adapter.pt \
PARITY_REPORT=/path/to/passing_parity_report.json \
STEPS=12000 \
SESSION=geowire_phase2_sft \
./scripts/launch_phase2_sft_tmux.sh
```

When GPUs 6-7 are free, switch to:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8
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

- 2B `qwen_layout_grid` cache/ramp must pass on the formal manifests.
- Graph thresholds must be calibrated on real clips with visual overlay audit; the smoke used loose thresholds only to verify the engineering path.
- Phase 1 should not advance to paper Phase 2 claims until full graph beats self/random/shuffled controls on a real validation subset.
- Benchmark generation wrappers still need real prediction runs for VSI/MMSI/ViewSpatial; the scorer already exists.

## 2026-07-09 Node214 Resume

Node218 8-card formal Phase 2 run stopped before completion after node218 became
unavailable for continued use. The shared run artifacts were inspected from
node214.

Previous 8-card run:

```text
session: geowire_p2_sft_qwen3vl2b_formal_b3_ga3_px376320_8gpu
last logged step: 36300 / 60000
last logged progress: 60.5%
latest durable checkpoint: checkpoint_step_035000
latest checkpoint path:
  /mnt/guojh/lq/new/GeoWire/geowire/runs/geowire_p2_sft_qwen3vl2b_formal_b3_ga3_px376320_8gpu/checkpoint_step_035000/phase2_adapters.pt
note: steps 35001-36300 were not checkpointed and are intentionally replayed/lost.
```

Resume code update:

```text
file: geowire/geowire/training/train_sft.py
added: --phase2-checkpoint for loading Phase 2 adapters
added: --start-step for checkpoint/log numbering
added: --data-step-offset for continuing data sampling after changing world size
```

Node214 resume:

```text
session: geowire_p2_sft_qwen3vl2b_resume214_4gpu_b3_ga6_from35000_px376320
node: node214
gpus: 0,1,2,4
world size: 4
microbatch per gpu: 3
gradient accumulation: 6
effective batch: 72, matching the previous 8-card run
phase2 checkpoint: checkpoint_step_035000 from the 8-card run
start step: 35000
data step offset: 70000
local steps to run: 50000
qa_to_tip: 15
image max pixels: 376320
initial health: step 35140 reached with no OOM/Traceback/FAILED
initial speed: about 3.59 seconds per local micro-step
initial ETA: about 49.8 hours from 2026-07-09 14:50 CST
initial expected finish: about 2026-07-11 16:36 CST
```
