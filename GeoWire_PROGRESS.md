# GeoWire Progress

Last updated: 2026-07-06

## Current Status

- Local GitHub SSH auth is available for `Trump0412`.
- Local project root is `/home/chenbp/GeoWire`.
- Python package root is `/home/chenbp/GeoWire/geowire`.
- The v0.2 master design, implementation scaffold and training protocol are
  mirrored into `geowire/docs/`.

## Implemented Software Loop

- Repository-local scripts can run before `pip install -e .` via `scripts/_bootstrap.py`.
- `make smoke` and `scripts/smoke_all.py` run the current local smoke path.
- Synthetic toy scene generation is available.
- Coordinate contract roundtrip check passes for 16:9, 9:16, 4:3 and tall inputs.
- Toy cache generation writes metadata, frame transforms, qwen input metadata and token layout.
- Graph construction writes normalized sparse COO graphs from cached token layouts.
- Sparse transport and GeoWire blocks are implemented with alpha initialized to zero.
- TIP v0.2 losses are implemented: recovery + substitution + optional keep.
- Non-edge isolation is retained as a diagnostic, not a training loss.
- Random/shuffled/self-loop graph diagnostics are available for Phase 1 debug.

## Latest Local Smoke

Command:

```bash
cd /home/chenbp/GeoWire/geowire
python scripts/smoke_all.py --output runs/smoke_local
```

Result:

```text
24 passed, 1 skipped
coordinate contract passed
toy graph row sums min/max = 1.0 / 1.0
```

## Gated Before Real Training

- Real VGGT cache generation requires pinned `third_party/vggt`, VGGT-1B weights and visual graph audit.
- Qwen3-VL bridge still requires installed compatible Transformers/Qwen3-VL source inspection and alpha=0 parity.
- Phase 2 SFT remains blocked until Phase 1 diagnostics prove full graph beats random/shuffled controls.
