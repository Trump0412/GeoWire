# Decisions

## Architecture

- VGGT is a frozen geometry control-plane provider.
- Qwen visual tokens are the only transported values.
- GeoWire is inserted only between Qwen visual encoder output and the original visual merger.
- Geometry features are not concatenated, added or used as cross-attention values in the main model.

## Current Resource State

- Existing VGGT-1B weights may be reused from `/mnt/guojh/lq/new/weights/base_models/VGGT-1B`.
- Existing Qwen2.5-VL weights are not the primary checkpoint.
- Qwen3-VL-4B-Instruct is present at `/mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct`.
- Existing conda env for bootstrapping: `/mnt/guojh/lq/new/conda/envs/geothinker`.
- Do not overwrite the shared `geothinker` Transformers install for GeoWire. Use the project-local overlay:
  `/mnt/guojh/lq/new/GeoWire/.deps/transformers_4_57_6`.
- Pinned VGGT source for server runs:
  `/mnt/guojh/lq/new/GeoWire/third_party/vggt`, commit `a288dd0`.
- Qwen3-VL bridge parity passed on the server with Transformers `4.57.6` overlay:
  `runs/qwen_bridge_parity/real_backend_smoke_report.json`, `max_abs_logit_diff=0.0`.

## Bridge Scope

- The current Qwen3-VL bridge wraps `Qwen3VLModel.get_image_features` and inserts GeoWire after Qwen image features are produced and before image-token replacement into language embeddings.
- Original Qwen placeholder scatter, RoPE, deepstack visual features, language model and generation path remain owned by Transformers.
- This bridge is acceptable for training only after alpha-zero parity passes for the target checkpoint and processor.
- Deepstack visual features are currently left unchanged; this is recorded as an implementation boundary for ablation, not a new architecture component.

## Training Protocol v0.2

- Phase 1 optimizes `L_rec + lambda_sub L_sub + optional lambda_keep L_keep`.
- Non-edge isolation is diagnostic only and is not included in the default training loss.
- Support substitution is only valid under masked target tokens.
- Phase 2 uses `3 QA batches : 1 TIP batch` and remains gated behind Qwen bridge parity.
