"""Narrowly add complete-output and exact-token guards to the H3 notebook.

Only prompt-related settings, prompt documentation, and the prompt-generation cell may
change. Every other cell and all unrelated/user-edited settings remain untouched.
"""

from __future__ import annotations

import json
import re
import runpy
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "StillLife_MiniMax_H3_FL2V_Interpolation.ipynb"
BUILDER_PATH = ROOT / "scripts" / "build_minimax_h3_interpolation_notebook.py"
ALLOWED_CELL_IDS = {
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


def bounded_fragment(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def replace_bounded_fragment(
    current: str,
    reference: str,
    *,
    start_marker: str,
    end_marker: str,
) -> str:
    start = current.index(start_marker)
    end = current.index(end_marker, start)
    return (
        current[:start]
        + bounded_fragment(reference, start_marker, end_marker)
        + current[end:]
    )


def replace_single_setting(current: str, reference: str, setting: str) -> str:
    pattern = re.compile(rf"^{re.escape(setting)}\s*=.*$", flags=re.MULTILINE)
    reference_match = pattern.search(reference)
    current_matches = list(pattern.finditer(current))
    if reference_match is None or len(current_matches) != 1:
        raise RuntimeError(f"Could not safely replace exactly one {setting} setting")
    return pattern.sub(reference_match.group(0), current, count=1)


def replace_settings(current: str, reference: str) -> str:
    migrated = replace_bounded_fragment(
        current,
        reference,
        start_marker="H3_WORKFLOW_PATCH_VERSION =",
        end_marker="H3_BASE_SEED =",
    )
    migrated = replace_bounded_fragment(
        migrated,
        reference,
        start_marker="# Positive correspondence language",
        end_marker="OPENAI_KEY_FILENAME =",
    )
    migrated = replace_single_setting(
        migrated, reference, "OPENAI_MAX_OUTPUT_TOKENS"
    )
    description_settings = bounded_fragment(
        reference,
        "OPENAI_H3_DESCRIPTION_MIN_CHARS =",
        "VISION_IMAGE_MAX_SIDE =",
    )
    description_pattern = re.compile(
        r"^OPENAI_H3_DESCRIPTION_MIN_CHARS\s*=.*\n"
        r"OPENAI_H3_DESCRIPTION_MAX_CHARS\s*=.*\n",
        flags=re.MULTILINE,
    )
    if description_pattern.search(migrated):
        migrated = description_pattern.sub(description_settings, migrated, count=1)
    else:
        marker = "OPENAI_MAX_ATTEMPTS = 3\n"
        if migrated.count(marker) != 1:
            raise RuntimeError("Could not locate the OpenAI attempts setting")
        migrated = migrated.replace(marker, marker + description_settings, 1)
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

    settings = current_cells["h3-02-settings"]
    settings["source"] = replace_settings(
        source(settings), source(reference_cells["h3-02-settings"])
    ).splitlines(keepends=True)

    for cell_id in ("h3-03-research", "h3-10-prompts-heading"):
        current_cells[cell_id]["source"] = deepcopy(reference_cells[cell_id]["source"])

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
        "Migrated H3 prompt-budget cells: "
        + ", ".join(sorted(ALLOWED_CELL_IDS))
        + "; all other cells and unrelated settings preserved."
    )


if __name__ == "__main__":
    main()
