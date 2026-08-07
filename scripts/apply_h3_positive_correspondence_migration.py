"""Narrowly migrate the H3 notebook to the official FL2VA prompt writer.

Only the title, the prompt-related portion of settings, the prompt explanation, and the
prompt-generation cell may change. User-authored run selection and finishing settings remain
byte-for-byte within their existing cell source.
"""

from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "StillLife_MiniMax_H3_FL2V_Interpolation.ipynb"
BUILDER_PATH = ROOT / "scripts" / "build_minimax_h3_interpolation_notebook.py"
ALLOWED_CELL_IDS = {
    "h3-00-title",
    "h3-02-settings",
    "h3-03-research",
    "h3-10-prompts-heading",
    "h3-11-prompts",
}


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def cells_by_id(notebook: dict) -> dict[str, dict]:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError("Notebook has no cell list")
    result = {cell.get("id"): cell for cell in cells if isinstance(cell, dict)}
    if len(result) != len(cells) or any(not cell_id for cell_id in result):
        raise RuntimeError("Notebook cell IDs must be present and unique")
    return result


def prompt_settings_fragment(settings_source: str) -> str:
    start = settings_source.index("# Positive correspondence language")
    end = settings_source.index("OPENAI_KEY_FILENAME =")
    return settings_source[start:end]


def replace_prompt_settings(current_source: str, reference_source: str) -> str:
    start_candidates = (
        "# Exact supplied starting instruction.",
        "# Positive correspondence language",
    )
    starts = [current_source.find(marker) for marker in start_candidates]
    starts = [index for index in starts if index >= 0]
    if len(starts) != 1:
        raise RuntimeError("Could not locate exactly one H3 prompt-settings start marker")
    end = current_source.find("OPENAI_KEY_FILENAME =")
    if end < 0 or end <= starts[0]:
        raise RuntimeError("Could not locate the H3 OpenAI settings boundary")
    migrated = (
        current_source[: starts[0]]
        + prompt_settings_fragment(reference_source)
        + current_source[end:]
    )
    if "H3_WORKFLOW_PATCH_VERSION = 3" in migrated:
        migrated = migrated.replace(
            "H3_WORKFLOW_PATCH_VERSION = 3",
            "H3_WORKFLOW_PATCH_VERSION = 4",
            1,
        )
    elif "H3_WORKFLOW_PATCH_VERSION = 4" not in migrated:
        raise RuntimeError("Unexpected H3 workflow patch version")
    if 'OPENAI_IMAGE_DETAIL = "high"' in migrated:
        migrated = migrated.replace(
            'OPENAI_IMAGE_DETAIL = "high"',
            'OPENAI_IMAGE_DETAIL = "original"',
            1,
        )
    elif 'OPENAI_IMAGE_DETAIL = "original"' not in migrated:
        raise RuntimeError("Unexpected OpenAI image-detail setting")
    return migrated


def main() -> None:
    current = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    before = deepcopy(current)
    reference = runpy.run_path(str(BUILDER_PATH))["notebook"]
    current_cells = cells_by_id(current)
    reference_cells = cells_by_id(reference)
    missing = ALLOWED_CELL_IDS - current_cells.keys()
    if missing:
        raise RuntimeError(f"H3 notebook is missing migration cells: {sorted(missing)}")

    for cell_id in ("h3-00-title", "h3-03-research", "h3-10-prompts-heading"):
        current_cells[cell_id]["source"] = deepcopy(reference_cells[cell_id]["source"])

    settings = current_cells["h3-02-settings"]
    settings["source"] = replace_prompt_settings(
        source(settings), source(reference_cells["h3-02-settings"])
    ).splitlines(keepends=True)

    prompt_cell = current_cells["h3-11-prompts"]
    prompt_cell["source"] = deepcopy(reference_cells["h3-11-prompts"]["source"])
    prompt_cell["execution_count"] = None
    prompt_cell["outputs"] = []

    before_cells = cells_by_id(before)
    for cell_id, cell in current_cells.items():
        if cell_id not in ALLOWED_CELL_IDS and cell != before_cells[cell_id]:
            raise RuntimeError(f"Migration unexpectedly changed untouched cell {cell_id}")

    migrated_settings = source(settings)
    required_preserved = (
        'SOURCE_RUN_DIRECTORY = "science_path_prompt_only_chimera_0016_20260804T173337Z"',
        "UNASSIGN_RUNTIME_WHEN_FINISHED = False",
        "RUN_FULL_H3_SEQUENCE = True",
    )
    missing_preserved = [item for item in required_preserved if item not in migrated_settings]
    if missing_preserved:
        raise RuntimeError(f"User-authored settings were not preserved: {missing_preserved}")

    NOTEBOOK_PATH.write_text(
        json.dumps(current, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Migrated H3 cells: "
        + ", ".join(sorted(ALLOWED_CELL_IDS))
        + "; all other cells and user settings preserved."
    )


if __name__ == "__main__":
    main()
