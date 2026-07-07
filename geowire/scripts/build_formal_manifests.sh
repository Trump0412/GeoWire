#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/guojh/lq/new/GeoWire/geowire}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geothinker/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/guojh/lq/new/local_mirror/myproject/spatial4nips/data/train}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8}"
MEDIA_ROOT="${MEDIA_ROOT:-/mnt/guojh/lq/new/local_mirror/myproject/spatial4nips/data/media}"
MAX_FRAMES="${MAX_FRAMES:-8}"

cd "${PROJECT_ROOT}"
mkdir -p "${MANIFEST_ROOT}"

make_manifest() {
  local name="$1"
  local output="$2"
  shift 2
  if [[ -s "${output}" ]]; then
    echo "manifest exists: ${output} ($(wc -l < "${output}") rows)"
    return 0
  fi
  echo "building manifest ${name} -> ${output}"
  "${PYTHON_BIN}" scripts/prepare_spatial4nips_manifest.py \
    "$@" \
    --media-root "${MEDIA_ROOT}" \
    --output "${output}" \
    --max-frames "${MAX_FRAMES}" \
    --write-report "${output%.jsonl}_report.json"
}

make_manifest llava_hound "${MANIFEST_ROOT}/llava_hound.jsonl" \
  --input "${DATA_ROOT}/llava_hound_64k.json" \
  --source-dataset llava_hound \
  --min-frames 2 \
  --trust-video-frame-count 8

make_manifest spar "${MANIFEST_ROOT}/spar.jsonl" \
  --input "${DATA_ROOT}/spar_234k.json" \
  --source-dataset spar \
  --min-frames 1

make_manifest vsi590k "${MANIFEST_ROOT}/vsi590k.jsonl" \
  --input "${DATA_ROOT}/vsi_590k.json" \
  --source-dataset vsi590k \
  --min-frames 1

make_manifest vlm3r_vsi "${MANIFEST_ROOT}/vlm3r_vsi.jsonl" \
  --input "${DATA_ROOT}/vlm3r_vsi_205k.json" \
  --source-dataset vlm3r_vsi \
  --min-frames 2

make_manifest vlm3r_vst "${MANIFEST_ROOT}/vlm3r_vst.jsonl" \
  --input "${DATA_ROOT}/vlm3r_vst_132k.json" \
  --source-dataset vlm3r_vst \
  --min-frames 2

make_manifest joyai "${MANIFEST_ROOT}/joyai.jsonl" \
  --input "${DATA_ROOT}/joyai_openspatial_100k.jsonl" \
  --source-dataset joyai \
  --min-frames 1

make_manifest mindcube "${MANIFEST_ROOT}/mindcube.jsonl" \
  --input "${DATA_ROOT}/mindcube_10k.json" \
  --source-dataset mindcube \
  --min-frames 1

if [[ ! -s "${MANIFEST_ROOT}/phase2_all_unshuffled.jsonl" ]]; then
  cat \
    "${MANIFEST_ROOT}/llava_hound.jsonl" \
    "${MANIFEST_ROOT}/spar.jsonl" \
    "${MANIFEST_ROOT}/vsi590k.jsonl" \
    "${MANIFEST_ROOT}/vlm3r_vsi.jsonl" \
    "${MANIFEST_ROOT}/vlm3r_vst.jsonl" \
    "${MANIFEST_ROOT}/joyai.jsonl" \
    "${MANIFEST_ROOT}/mindcube.jsonl" \
    > "${MANIFEST_ROOT}/phase2_all_unshuffled.jsonl"
fi

if [[ ! -s "${MANIFEST_ROOT}/phase2_all.jsonl" ]]; then
  "${PYTHON_BIN}" - "${MANIFEST_ROOT}" <<'PY'
from pathlib import Path
import random
import sys

root = Path(sys.argv[1])
lines = (root / "phase2_all_unshuffled.jsonl").read_text(encoding="utf-8").splitlines()
rng = random.Random(3407)
rng.shuffle(lines)
(root / "phase2_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
print({"phase2_all": len(lines)})
PY
fi

make_manifest llava_hound_tip "${MANIFEST_ROOT}/llava_hound_tip.jsonl" \
  --input "${DATA_ROOT}/llava_hound_64k.json" \
  --source-dataset llava_hound \
  --min-frames 2 \
  --trust-video-frame-count 8

make_manifest spar_tip "${MANIFEST_ROOT}/spar_tip.jsonl" \
  --input "${DATA_ROOT}/spar_234k.json" \
  --source-dataset spar \
  --min-frames 2

make_manifest vsi590k_tip "${MANIFEST_ROOT}/vsi590k_tip.jsonl" \
  --input "${DATA_ROOT}/vsi_590k.json" \
  --source-dataset vsi590k \
  --min-frames 2

make_manifest vlm3r_vsi_tip "${MANIFEST_ROOT}/vlm3r_vsi_tip.jsonl" \
  --input "${DATA_ROOT}/vlm3r_vsi_205k.json" \
  --source-dataset vlm3r_vsi \
  --min-frames 2

make_manifest vlm3r_vst_tip "${MANIFEST_ROOT}/vlm3r_vst_tip.jsonl" \
  --input "${DATA_ROOT}/vlm3r_vst_132k.json" \
  --source-dataset vlm3r_vst \
  --min-frames 2

make_manifest mindcube_tip "${MANIFEST_ROOT}/mindcube_tip.jsonl" \
  --input "${DATA_ROOT}/mindcube_10k.json" \
  --source-dataset mindcube \
  --min-frames 2

if [[ ! -s "${MANIFEST_ROOT}/phase2_tip_multiframe_unshuffled.jsonl" ]]; then
  cat \
    "${MANIFEST_ROOT}/llava_hound_tip.jsonl" \
    "${MANIFEST_ROOT}/spar_tip.jsonl" \
    "${MANIFEST_ROOT}/vsi590k_tip.jsonl" \
    "${MANIFEST_ROOT}/vlm3r_vsi_tip.jsonl" \
    "${MANIFEST_ROOT}/vlm3r_vst_tip.jsonl" \
    "${MANIFEST_ROOT}/mindcube_tip.jsonl" \
    > "${MANIFEST_ROOT}/phase2_tip_multiframe_unshuffled.jsonl"
fi

if [[ ! -s "${MANIFEST_ROOT}/phase2_tip_multiframe.jsonl" ]]; then
  "${PYTHON_BIN}" - "${MANIFEST_ROOT}" <<'PY'
from pathlib import Path
import random
import sys

root = Path(sys.argv[1])
lines = (root / "phase2_tip_multiframe_unshuffled.jsonl").read_text(encoding="utf-8").splitlines()
rng = random.Random(3407)
rng.shuffle(lines)
(root / "phase2_tip_multiframe.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
print({"phase2_tip_multiframe": len(lines)})
PY
fi

"${PYTHON_BIN}" - "${MANIFEST_ROOT}" <<'PY'
from pathlib import Path
import json
import sys
from geowire.utils.io import iter_jsonl, write_json

root = Path(sys.argv[1])
report = {"manifest_root": str(root), "files": {}}
for path in sorted(root.glob("*.jsonl")):
    rows = 0
    clips = set()
    frames = {}
    sources = {}
    for row in iter_jsonl(path):
        rows += 1
        clips.add(row["clip_id"])
        n = len(row["frame_paths"])
        frames[str(n)] = frames.get(str(n), 0) + 1
        src = row.get("source_dataset", "unknown")
        sources[src] = sources.get(src, 0) + 1
    report["files"][path.name] = {
        "rows": rows,
        "unique_clip_ids": len(clips),
        "frame_counts": dict(sorted(frames.items(), key=lambda item: int(item[0]))),
        "sources": sources,
    }
qa_rows = report["files"]["phase2_all.jsonl"]["rows"]
tip_rows = report["files"]["phase2_tip_multiframe.jsonl"]["rows"]
global_batch_b2 = 8 * 2
cycles_b2 = (qa_rows + global_batch_b2 * 15 - 1) // (global_batch_b2 * 15)
steps_b2 = cycles_b2 * 16
report["eta_inputs"] = {
    "qa_rows": qa_rows,
    "tip_rows": tip_rows,
    "global_batch_b2": global_batch_b2,
    "qa_to_tip": 15,
    "phase2_steps_b2": steps_b2,
}
write_json(root / "formal_manifest_report.json", report)
print(json.dumps(report["eta_inputs"], indent=2, sort_keys=True))
print(root / "formal_manifest_report.json")
PY

echo "manifest counts:"
wc -l "${MANIFEST_ROOT}"/*.jsonl | sort -n
