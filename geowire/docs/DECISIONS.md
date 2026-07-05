# Decisions

## Architecture

- VGGT is a frozen geometry control-plane provider.
- Qwen visual tokens are the only transported values.
- GeoWire is inserted only between Qwen visual encoder output and the original visual merger.
- Geometry features are not concatenated, added or used as cross-attention values in the main model.

## Current Resource State

- Existing VGGT-1B weights may be reused from `/mnt/guojh/lq/new/weights/base_models/VGGT-1B`.
- Existing Qwen2.5-VL weights are not the primary checkpoint.
- Qwen3-VL-4B-Instruct must be downloaded before bridge inspection.
- Existing conda env for bootstrapping: `/mnt/guojh/lq/new/conda/envs/geothinker`.

## Training Protocol v0.2

- Phase 1 optimizes `L_rec + lambda_sub L_sub + optional lambda_keep L_keep`.
- Non-edge isolation is diagnostic only and is not included in the default training loss.
- Support substitution is only valid under masked target tokens.
- Phase 2 uses `3 QA batches : 1 TIP batch` and remains gated behind Qwen bridge parity.
