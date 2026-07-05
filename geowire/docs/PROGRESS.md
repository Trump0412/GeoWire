# GeoWire Progress

Last updated: 2026-07-06

## Current Software Gate

The repository currently passes the local software smoke path:

```bash
python scripts/smoke_all.py --output runs/smoke_local
```

Observed result:

```text
24 passed, 1 skipped
coordinate contract passed
toy cache generated
toy sparse graph normalized
TIP debug metrics written
```

## Done

- Script bootstrap before editable install.
- Synthetic toy scene generation.
- Coordinate contract check.
- Toy cache generation.
- Cached-layout graph construction and NPZ IO.
- Sparse transport and GeoWire blocks.
- TIP v0.2 recovery/substitution/keep losses.
- Non-edge isolation diagnostic kept out of training loss.
- Self-loop/random/shuffled graph controls.

## Still Gated

- Real VGGT cache generation.
- Manual visual graph audit on real clips.
- Qwen3-VL bridge inspection.
- Qwen bridge alpha=0 parity.
- Phase 1 real TIP training.
- Phase 2 spatial SFT.
