from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap() -> Path:
    """Make repository-local imports reliable when scripts are run before install."""

    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ.setdefault("PYTHONPATH", root_str)
    return root
