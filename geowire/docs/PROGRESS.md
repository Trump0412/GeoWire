# GeoWire Progress

Last updated: 2026-07-06

## Current Software Gate

The repository currently passes the local software smoke path:

```bash
python scripts/smoke_all.py --output runs/smoke_local
```

Observed result:

```text
29 passed, 1 skipped
coordinate contract passed
toy cache generated
toy sparse graph normalized
cached TIP checkpoint written
prediction JSONL evaluator wrote an eval report
```

The same cached Phase 1 smoke passed on the server at
`/mnt/guojh/lq/new/GeoWire/geowire` with
`/mnt/guojh/lq/new/conda/envs/geothinker/bin/python`.

```text
29 passed, 1 skipped
checkpoint: runs/smoke_eval_server/tip_debug/geowire_adapter.pt
eval report: runs/smoke_eval_server/eval_report.json
```

The server also completed the current step50 engineering gate on 2026-07-06:

```text
gpu: CUDA_VISIBLE_DEVICES=6
command: scripts/train_tip.py --manifest assets/toy_scene/manifest.jsonl --cache-root runs/step50_toy_cache --output runs/phase1_tip_step50_gpu6 --steps 50 --device cuda
checkpoint: runs/phase1_tip_step50_gpu6/geowire_adapter.pt
metrics: runs/phase1_tip_step50_gpu6/metrics.json
final step: 50
final loss: 0.6762770414352417
eval_full_rec: 0.6607329845428467
```

This is a software-loop training proof on toy cached data. Real Phase 1 training
is still gated below.

## Done

- Script bootstrap before editable install.
- Synthetic toy scene generation.
- Coordinate contract check.
- Toy cache generation.
- Cached-layout graph construction and NPZ IO.
- Standard cached semantic-token file: `semantic_tokens.safetensors`.
- Sparse transport and GeoWire blocks.
- TIP v0.2 recovery/substitution/keep losses.
- Non-edge isolation diagnostic kept out of training loss.
- Self-loop/random/shuffled graph controls.
- Cached Phase 1 training loop with checkpoint and metrics output.
- Phase 1 readiness check and tmux launcher.
- Prediction JSONL evaluator with benchmark config/normalizer recording.
- Server cached Phase 1 toy training completed through step 50 on an idle A100.
- Real Qwen3-VL semantic-token cache generation.
- Real VGGT geometry/track cache generation from pinned `third_party/vggt`.
- Real track/projective graph construction with configurable thresholds.
- Qwen3-VL image-feature bridge with alpha-zero parity script.
- Phase 2 SFT trainer with Qwen LoRA + GeoWire and `3 QA : 1 TIP` schedule.
- Phase 2 tmux launcher with real cache and parity readiness gate.

## Latest Real Server Gates

```text
real cache: runs/real_backend_smoke_cache
real graph stats: track_edges=32, projective_edges=138, self_edges=160
phase1 real smoke: runs/real_backend_phase1_tip_smoke, steps=2
qwen parity: runs/qwen_bridge_parity/real_backend_smoke_report.json, max_abs_logit_diff=0.0
phase2 real smoke: runs/phase2_sft_smoke_real_backend_4step, schedule reached TIP on step 4
```

## Still Gated Before Full Runs

- Real training manifests and cache roots for SPAR/LLaVA-Hound/XVR are not yet selected in this repo.
- Manual visual graph audit and threshold calibration are still required before launching large Phase 1.
- Real benchmark generation is not yet run; only the prediction scorer is implemented.
