# Experiment Log

## M0 Bootstrap

Status: software smoke passed locally on 2026-07-06.

- Environment inspection command: `python scripts/inspect_environment.py --write runs/env_report.json`
- Coordinate contract command: `python scripts/verify_coordinate_contract.py --write runs/coordinate_contract.json`
- TIP debug command: `python scripts/train_tip.py --output runs/tip_debug`
- Full smoke command: `python scripts/smoke_all.py --output runs/smoke_local`

Latest smoke:

```text
24 passed, 1 skipped
toy cache generated
toy graph normalized
coordinate roundtrip max error < 0.5 px
```

Server smoke:

```text
path: /mnt/guojh/lq/new/GeoWire/geowire
python: /mnt/guojh/lq/new/conda/envs/geothinker/bin/python
result: 24 passed, 1 skipped
gpu: 8 x A100 80GB visible
```

Qwen3-VL inspection:

```text
server transformers: 4.50.0
status: gated; install or vendor a Qwen3-VL-compatible Transformers build before bridge parity
```

No Phase 1 or Phase 2 result is claimed until VGGT cache, graph overlays and Qwen bridge parity pass.
