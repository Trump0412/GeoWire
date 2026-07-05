.PHONY: install test lint inspect toy graph tip smoke

PYTHON ?= python
export PYTHONPATH := $(CURDIR)
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

inspect:
	$(PYTHON) scripts/inspect_environment.py --write runs/env_report.json

toy:
	$(PYTHON) scripts/generate_toy_scene.py --output assets/toy_scene --manifest assets/toy_scene/manifest.jsonl

graph: toy
	$(PYTHON) scripts/verify_coordinate_contract.py --write runs/coordinate_contract.json
	$(PYTHON) scripts/cache_vggt.py --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache --backend toy
	$(PYTHON) scripts/build_graphs.py --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache --output runs/phase0_graph
	$(PYTHON) scripts/check_training_readiness.py --phase phase1 --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache --write runs/phase1_readiness.json
	$(PYTHON) scripts/verify_graph.py --graph runs/toy_cache/toy_scene_clip_000/graph_coo.npz

tip:
	$(PYTHON) scripts/train_tip.py --manifest assets/toy_scene/manifest.jsonl --cache-root runs/toy_cache --output runs/tip_debug --steps 2

smoke: inspect graph tip test
