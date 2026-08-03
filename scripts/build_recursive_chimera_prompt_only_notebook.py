"""Validate or safely export the authored CHIMERA production notebook.

The checked-in notebook is the source of truth.  In particular, prompts,
settings, markdown, outputs, and ad-hoc experiment cells belong to the notebook
author and must never be regenerated from another notebook or a parallel data
file.

With no environment override this command validates the canonical notebook and
does not write anything.  ``CHIMERA_PROMPT_ONLY_NOTEBOOK_OUTPUT`` may name a new
path for a byte-for-byte export (useful to tests and one-off copies), but an
existing destination is always rejected.  There is deliberately no force-
overwrite escape hatch.  Repository changes should be applied with explicit,
marker-based notebook migrations that name the cells they are allowed to edit.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NOTEBOOK = (
    ROOT / "notebooks" / "StillLife_Recursive_CHIMERA_Prompt_Only.ipynb"
)
OUTPUT_ENVIRONMENT_VARIABLE = "CHIMERA_PROMPT_ONLY_NOTEBOOK_OUTPUT"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def validate_notebook(notebook: dict) -> None:
    if notebook.get("nbformat") != 4:
        raise RuntimeError("The CHIMERA notebook must use notebook format 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError("The CHIMERA notebook has no cells")

    cell_ids = [cell.get("id") for cell in cells]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
        raise RuntimeError("Every CHIMERA notebook cell must have a stable ID")
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError("CHIMERA notebook cell IDs must be unique")

    for cell in cells:
        if cell.get("cell_type") == "code":
            ast.parse(source(cell), filename=f"cell {cell['id']}")

    all_source = "\n".join(source(cell) for cell in cells)
    required_markers = (
        "BASE_STAGES = [",
        "from flowmorph_klein.chimera import (",
        "CHIMERA_SESSION.render_pair(",
        "RUN_DIRECTORY",
    )
    missing = [marker for marker in required_markers if marker not in all_source]
    if missing:
        raise RuntimeError(f"CHIMERA notebook is missing required markers: {missing}")


def main() -> None:
    if not CANONICAL_NOTEBOOK.is_file():
        raise FileNotFoundError(
            "The authored CHIMERA notebook is missing; restore it from version "
            "control rather than regenerating it from a different notebook."
        )

    canonical_bytes = CANONICAL_NOTEBOOK.read_bytes()
    notebook = json.loads(canonical_bytes)
    validate_notebook(notebook)

    requested_output = os.environ.get(OUTPUT_ENVIRONMENT_VARIABLE)
    if requested_output is None:
        print(f"Validated authored notebook without modifying it: {CANONICAL_NOTEBOOK}")
        return

    output = Path(requested_output).expanduser().resolve()
    if output == CANONICAL_NOTEBOOK.resolve():
        raise RuntimeError(
            "Refusing to overwrite the authored CHIMERA notebook. Edit it directly "
            "or apply a narrowly scoped migration."
        )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing notebook export: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes)
    print(f"Exported an exact copy of the authored notebook: {output}")


if __name__ == "__main__":
    main()
