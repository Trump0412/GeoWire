# GeoWire Method and Training Details

Last updated: 2026-07-07

This document is the pre-experiment control record for GeoWire. It records the
method, losses, training tasks, data use, utilization target, and go/no-go gates
that must be checked before launching long runs on the 8 x A100 node.

## 0. Current Decision

The latest online-Qwen 8-GPU pilot proved that the software path works, but it
is not an acceptable formal training configuration.

Observed pilot:

| Item | Value |
|---|---:|
| Node | node218, 8 x A100 80GB |
| Phase 1 pilot | 16 steps, online frozen Qwen TIP |
| Phase 1 memory | about 10-11 GB/GPU |
| Phase 1 elapsed | 67 seconds including Qwen load |
| Phase 2 pilot | 16 steps, online Qwen + ZeRO-2 |
| Phase 2 memory | about 17-27 GB/GPU |
| Phase 2 elapsed | 41 seconds including model/ZeRO init |
| Phase 2 schedule | 15 QA : 1 TIP |

This is too small. The formal training target is to run near 65-75 GB/GPU on
A100-80G while preserving stable loss and throughput.

Status after the latest engineering pass:

- true per-device QA batching is implemented in Phase 2 SFT;
- true TIP microbatch packing is implemented in Phase 1 and Phase 2;
- checkpoint save intervals are implemented for formal Phase 2;
- the formal Qwen-free data manifests have been generated and audited;
- the 4B model path is kept only as a validated pilot path;
- the formal launch should switch to `Qwen3-VL-2B-Instruct` before the next
  cache/ramp/training pass.

Formal run is still gated until a 512- or 1024-sample 8-card tuning pilot with
the 2B model shows:

- sustained memory target: 65-75 GB/GPU;
- no OOM for the selected frame count and max text length;
- stable loss scale in bf16;
- measured samples/sec and tokens/sec;
- checkpoint save/load works after the first interval.

## 1. Method Summary

GeoWire is a geometry-controlled cross-frame semantic transport module for
video and multi-view spatial reasoning.

The method deliberately separates:

| Plane | Source | Trainable | Role |
|---|---|---:|---|
| Geometry control plane | VGGT / dataset geometry / graph builder | No | Defines which visual tokens may exchange information |
| Semantic value plane | frozen Qwen visual tokens | No in Phase 1; frozen visual encoder in Phase 2 | Provides the token values transported along geometry edges |
| GeoWire transport | sparse residual blocks | Yes | Moves semantic values only along approved geometry edges |
| Language reasoning | Qwen3-VL LoRA in Phase 2 | Yes | Turns transported visual evidence into answers |

The core update is:

```text
H_0 = H
H_{l+1} = H_l + alpha_l * W_o(P_G * W_v(LN(H_l)))
```

where `P_G` is a frozen sparse graph, `W_v/W_o/alpha` are GeoWire parameters,
and `H` is the Qwen visual-token sequence. `alpha` is initialized to zero so
the Qwen path is exactly preserved before training.

Important constraints:

- The graph topology cannot depend on text, answer labels, LLM hidden states,
  semantic similarity, or a learned router.
- GeoWire does not maintain a semantic bank, geometry bank, memory table, or
  cross-clip retrieval store.
- The only persistent per-clip structure is the sparse correspondence graph and
  token layout needed to align graph nodes with Qwen visual tokens.

## 2. Data Scope

GeoWire is separate from GeoThinker. We reuse the environment, weights,
benchmarks, and available datasets, but not the GeoThinker Stage 3/GSPO method.

Training data to include:

| Source | Available records | Use in GeoWire | Role |
|---|---:|---|---|
| LLaVA-Hound | 63,750 | Phase 1/2 | natural video semantics and temporal-spatial QA |
| SPAR / spar_234k | 234,056 | Phase 0/1/2 | multi-view spatial relations and RGB-D style evidence |
| VSI-590K | 590,667 | Phase 2 | large spatial instruction tuning |
| VLM3R-VSI | 205,456 | Phase 2 | view-spatial instruction data |
| VLM3R-VST | 132,568 | Phase 2 | temporal/view-spatial instruction data |
| JoyAI OpenSpatial | 100,000 | Phase 2 | diverse spatial QA |
| MindCube | 10,000 | Phase 2 | structured multi-view/mind-cube spatial reasoning |

Formal manifest artifacts:

```text
root: /mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8
qa manifest: phase2_all.jsonl
tip manifest: phase2_tip_multiframe.jsonl
report: formal_manifest_report.json
```

Formal manifest counts:

| Manifest | Rows | Unique clips | Notes |
|---|---:|---:|---|
| `phase2_all.jsonl` | 1,336,497 | 370,380 | all approved QA records |
| `phase2_tip_multiframe.jsonl` | 944,716 | 179,810 | single-frame records removed |

`phase2_all.jsonl` source mix:

| Source | Rows |
|---|---:|
| LLaVA-Hound | 63,750 |
| SPAR | 234,056 |
| VSI-590K | 590,667 |
| VLM3R-VSI | 205,456 |
| VLM3R-VST | 132,568 |
| JoyAI OpenSpatial | 100,000 |
| MindCube | 10,000 |

`phase2_tip_multiframe.jsonl` source mix:

| Source | Rows |
|---|---:|
| LLaVA-Hound | 63,750 |
| SPAR | 158,794 |
| VSI-590K | 374,148 |
| VLM3R-VSI | 205,456 |
| VLM3R-VST | 132,568 |
| MindCube | 10,000 |

Data not used for GeoWire training:

- SpatialLadder is not part of this GeoWire protocol.
- VSI-Bench, ReVSI, MMSI, DSR-Bench, ViewSpatial-style benchmark validation or
  test QA remain evaluation-only. Training uses the prepared VSI-590K/VLM3R
  training data by decision.

Run control:

- Every real run must write its exact manifest, source list, git commit, cache
  backend, and training launch command.
- VSI-590K and VLM3R are approved for the formal GeoWire training mixture.
- There is no third training stage in the approved plan.

## 3. Phase 0: Manifest and Graph Preparation

Goal: build reliable per-clip graph inputs before training any model.

Required artifacts per clip:

```text
metadata.json
token_layout.safetensors
graph_coo.npz
source record / original media path
```

Optional audit artifacts:

```text
geometry.safetensors
graph_overlay images
edge diagnostics
semantic_tokens.safetensors
```

The formal default is `TIP_FEATURE_MODE=online_qwen`, so full
`semantic_tokens.safetensors` caches are not required for every clip. They are
useful for small audits and debugging, but caching them for all data would
consume a large amount of disk and is not necessary for training.

Phase 0 tasks:

1. Build unified train manifests for the included datasets.
2. Normalize media paths to the shared root under `/mnt/guojh/lq/new`.
3. Build or import token layout for Qwen visual tokens.
4. Build compact sparse graphs with self edges plus high-confidence cross-frame
   or cross-view edges.
5. Write diagnostics: graph degree, edge type mix, cross-frame coverage,
   invalid/missing media count, and sample overlay audits.

Phase 0 pass conditions:

- missing media count is zero for selected training manifests;
- graph node count equals Qwen visual-token count;
- cross-frame/multi-view coverage is non-trivial on graph-rich subsets;
- visual overlay audits show reasonable correspondence edges;
- graph controls are available: self-loop, random same-degree, shuffled endpoint.

## 4. Phase 1: TIP Pretraining

Goal: train GeoWire to use the geometry graph before QA SFT can hide the
mechanism.

Trainable parameters:

| Component | Status |
|---|---:|
| VGGT / geometry source | frozen |
| Qwen visual encoder | frozen |
| Qwen language model | unused / frozen |
| GeoWire sparse transport | train |

Input:

- multi-frame or multi-view samples with valid graph nodes;
- frozen Qwen visual tokens computed online by Qwen;
- compact `P_G` sparse graph loaded from cache.

Loss:

```text
L_TIP = L_rec + lambda_sub * L_sub + lambda_keep * L_keep
```

Loss terms:

| Term | Formula / implementation | Purpose |
|---|---|---|
| `L_rec` | `1 - cosine(pred_masked, stopgrad(clean_masked))` | recover masked token semantics via graph-supported evidence |
| `L_sub` | `1 - cosine(pred_support_A, stopgrad(pred_support_B))` | make equivalent geometry supports interchangeable |
| `L_keep` | low-weight recovery on unmasked tokens | prevent transport from damaging directly visible evidence |
| non-edge isolation | diagnostic only | detect accidental leakage; not optimized as a main loss |

Recommended Phase 1 starting values:

```yaml
tip_feature_mode: online_qwen
frames: 16 first, 32 only after throughput/memory pilot
blocks: 2
mask_ratio: 0.15
lambda_sub: 0.25
lambda_keep: 0.02
lr_geowire: 2.0e-4
weight_decay: 0.05
precision: bf16
optimizer: AdamW
target_steps: 30000
```

Phase 1 data:

- graph-rich samples from LLaVA-Hound, SPAR, VLM3R, JoyAI, and MindCube;
- prioritize multi-frame/video/multi-view examples;
- downsample single-image or weak-geometry examples for TIP.

Phase 1 pass conditions before Phase 2 claims:

- full graph beats self-loop, random same-degree graph, and shuffled-endpoint
  graph on masked recovery;
- support substitution loss decreases without destroying `L_rec`;
- direct-observation keep loss remains low;
- failure cases can be explained by graph quality or visibility rather than
  token-layout bugs;
- checkpoint reload produces identical adapter outputs on a fixed smoke batch.

## 5. Phase 2: Spatial Instruction Tuning

Goal: turn geometry-supported visual evidence into real spatial answers.

Trainable parameters:

| Component | Status |
|---|---:|
| VGGT / graph builder | frozen |
| Qwen visual encoder | frozen |
| GeoWire | train / continue from Phase 1 |
| Qwen language model | LoRA |
| Qwen visual merger/projector | LoRA or small trainable subset only if enabled |

Training schedule:

```text
15 QA batches : 1 TIP batch
```

Loss:

```text
L_SFT = E[L_QA] + lambda_tip_effective * E[L_TIP]
```

where:

```text
L_QA = autoregressive cross entropy over answer tokens
lambda_tip_effective = 0.20 initially
```

Current Phase 2 defaults:

```yaml
qa_to_tip: 15
tip_feature_mode: online_qwen
base_model_for_formal_run: Qwen3-VL-2B-Instruct
base_model_for_pilot_only: Qwen3-VL-4B-Instruct
deepspeed: configs/deepspeed_zero2.json
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lr: 2.0e-5
precision: bf16
```

Formal first-run decision:

```yaml
hardware: 8 x A100 80GB
fine_tuning: LoRA first
full_parameter_tuning: ablation only after LoRA signal is known
cache_backend_first_run: qwen_layout_grid
qwen_model: Qwen3-VL-2B-Instruct
approved_large_sources:
  - VSI-590K
  - VLM3R-VSI
  - VLM3R-VST
```

LoRA vs full-parameter note:

- LoRA trains a small adapter subset, so it is faster, more stable, and leaves
  room to increase per-GPU batch size on 8 x A100.
- Full-parameter tuning trains the whole decoder stack and requires much more
  optimizer/gradient memory. It should use ZeRO-3/FSDP-style sharding and is not
  the right first run for isolating GeoWire's contribution.
- The formal first run therefore uses Qwen LoRA + trainable GeoWire; full-param
  is a later comparison if LoRA shows a useful signal.

Initial formal Phase 2 data mixture:

| Data | Suggested share | Notes |
|---|---:|---|
| SPAR / spar_234k | 25% | multi-view spatial relation and RGB-D style questions |
| VSI-590K | 25% | approved formal training source |
| VLM3R-VSI/VST | 25% | approved formal training source |
| LLaVA-Hound | 15% | natural video semantics and temporal reasoning |
| JoyAI OpenSpatial | 15% | diverse spatial QA |
| MindCube | 5% | structured multi-view reasoning |

Phase 2 pass conditions:

- QA loss decreases on a held-out non-overlap validation set;
- TIP retention loss remains meaningful rather than collapsing to zero;
- full graph still beats random/shuffled controls after SFT;
- eval improves on at least one debiased or cross-benchmark slice, not only on
  the easiest raw metric;
- memory and throughput meet the utilization target.

## 6. Follow-Up Ablations

There is no third training stage. After the two formal stages, only ablations
and comparison runs are allowed.

Allowed follow-up work:

- longer frame counts: 16 -> 32;
- Qwen2.5-VL-7B compatibility run for comparison with older ecosystems;
- GeoVR/geometry-aware base-model comparison if needed;
- LoRA vs full-parameter fine-tuning comparison if the LoRA run is promising;
- DSR-Bench and long-video evaluation.

Not allowed:

- calling these ablations a third stage;
- mixing benchmark validation/test QA into training;
- treating SpatialLadder as a GeoWire training source.

## 7. Throughput and Memory Plan

Target hardware:

```text
node218: 8 x A100 80GB
target memory: 65-75 GB/GPU
```

Why the current pilot underuses memory:

- one record per GPU per step;
- 8-frame smoke clips;
- the frozen Qwen3-VL-4B pilot was still relatively small for 80GB cards;
- ZeRO-2 partitions optimizer state/gradients but does not increase activation
  memory by itself;
- gradient accumulation increases update batch, not per-step memory.

Required implementation before formal long run:

1. Add true `--per-device-train-batch-size` to Phase 1 and Phase 2 loops.
2. Batch QA records through the processor instead of calling Qwen on one sample.
3. Batch TIP records by grouping compatible token lengths or by packing several
   independent graphs per rank.
4. Add length bucketing by frame count, visual-token count, and answer length.
5. Add an auto-ramp script:
   - try per-device batch 1, 2, 4, 6, 8;
   - try 16 frames before 32 frames;
   - record peak memory, sec/step, samples/sec, and OOM boundary.
6. Keep the final setting below the OOM boundary by at least 5 GB.

Starting tuning grid:

| Candidate | Frames | Per-GPU batch | Global microbatch on 8 GPUs | Expected use |
|---|---:|---:|---:|---|
| A | 16 | 2 | 16 | first safe real pilot |
| B | 16 | 4 | 32 | likely initial target |
| C | 16 | 6 | 48 | try if memory remains below 65 GB |
| D | 32 | 2 | 16 | long-context pilot |
| E | 32 | 4 | 32 | only if stable and still below OOM |

ETA can only be trusted after the tuning pilot. A rough planning estimate:

| Run | Assumption | Rough ETA |
|---|---|---:|
| Phase 1 30k steps | true global microbatch 32-48, 16 frames | 1-2 days |
| Phase 2 one pass over about 1.3M records, 4B pilot path | global microbatch 16, measured smoke average about 3.4 sec/step | about 3.5-4.2 days |
| Phase 2 one pass, 2B planned path | global microbatch 16, expected lower model cost | about 2.5-3.5 days |
| Phase 2 larger per-GPU batch on 2B | global microbatch 32-64 if stable | about 1.5-2.8 days |

The formal ETA must be recomputed from measured tuning logs, not from smoke
logs. The current smoke logs are initialization-heavy and too underbatched.

## 8. Evaluation Plan

Evaluation should be queued as soon as Phase 2 has the first real checkpoint.

Benchmarks:

| Benchmark | Use |
|---|---|
| VSI-Bench / VSI-Bench-Debiased | main video spatial reasoning signal |
| ReVSI | robustness / revised VSI-style signal |
| MMSI | multi-modal spatial instruction generalization |
| DSR-Bench | dynamic spatial reasoning and long-video stress |
| ViewSpatial-style slices | view-order and cross-view relation checks |

Required reports:

- accuracy by benchmark and subtype;
- debiased vs raw split;
- frame-count ablation: 4 / 8 / 16 / 32;
- graph ablation: full / self-loop / random / shuffled;
- Phase 1 checkpoint vs Phase 2 checkpoint;
- LoRA-only baseline without GeoWire;
- alpha-zero parity report for the untrained bridge.

## 9. Experiment Naming

Use names that encode the critical settings:

```text
geowire_p1_tip_qwen3vl2b_onlineqwen_f16_bpg4_8a100_YYYYMMDD
geowire_p2_sft_qwen3vl2b_zero2_qa15tip1_f16_bpg4_8a100_YYYYMMDD
geowire_eval_vsi_p2_f16_ckptSTEP_YYYYMMDD
```

Minimum files per run:

```text
trainer_state.json
metrics.jsonl
metrics.json
memory_profile.jsonl
throughput.json
leakage_audit.json
git_commit.txt
launch_command.sh
```

## 10. Immediate TODO Before Formal Training

1. Implement true per-device batching / graph packing in `train_tip.py` and
   `train_sft.py`. Status: done.
2. Build unified manifests for all selected GeoWire datasets, excluding
   SpatialLadder. Status: done.
3. Download or sync `Qwen3-VL-2B-Instruct` under the shared model root.
   Status: done; verified at
   `/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct`.
4. Build `qwen_layout_grid` cache with the 2B processor/checkpoint path.
5. Run a 512- or 1024-sample 8-card memory ramp.
6. Select the largest stable setting that keeps A100 memory around 65-75 GB.
7. Launch Phase 1 real TIP.
8. Launch Phase 2 SFT with `15 QA : 1 TIP`.
9. Start evaluation jobs from the first real Phase 2 checkpoint.
