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
TIP debug step, and unit tests. Real VGGT/Qwen training remains gated behind
resource inspection, graph overlays, and Qwen bridge parity.

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
