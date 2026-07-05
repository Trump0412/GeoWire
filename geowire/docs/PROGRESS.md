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

## Still Gated

- Real VGGT cache generation.
- Real Qwen visual-token caching into `semantic_tokens.safetensors`.
- Manual visual graph audit on real clips.
- Qwen3-VL bridge inspection. Current server `transformers==4.50.0` is too old
  for the Qwen3-VL inspection gate.
- Qwen bridge alpha=0 parity.
- Phase 1 real TIP training.
- Phase 2 spatial SFT.
