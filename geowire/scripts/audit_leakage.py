from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from geowire.data.leakage_audit import audit_scene_leakage
from geowire.data.manifest import load_manifest, manifest_hash
from geowire.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    report = asdict(audit_scene_leakage(records))
    report["manifest_hash"] = manifest_hash(records)
    write_json(args.write, report)
    print(args.write)
    if report["overlapping_scenes"] or report["missing_scene_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
