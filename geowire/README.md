# Georoute Package

This directory contains the runnable package and scripts for Georoute.

## Smoke Test

```bash
python -m pip install -e ".[dev]"
python scripts/smoke_all.py --output runs/smoke_local
```

The smoke test is synthetic and is intended to verify code plumbing without
requiring external models or datasets.

## Main Entry Points

- `scripts/build_graphs.py`: build sparse correspondence graphs from cached geometry.
- `scripts/train_tip.py`: Phase 1 topology intervention pretraining.
- `scripts/train_sft.py`: Phase 2 spatial instruction tuning.
- `scripts/evaluate.py`: prediction-file evaluation.
- `scripts/verify_qwen_bridge_parity.py`: parity check before Phase 2 training.

For real experiments, pass model, dataset, and cache locations through command
arguments or environment variables. The repository does not include large
assets, model weights, benchmark media, or generated checkpoints.
