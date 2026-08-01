"""Compatibility entry point for the FLUX-native SAGE notebook builder."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runpy.run_path(
        str(ROOT / "scripts" / "build_sage_flux_transition_notebook.py"),
        run_name="__main__",
    )
