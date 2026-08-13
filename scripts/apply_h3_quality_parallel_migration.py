"""Narrowly add H3 visual failure exemplars and parallel transition calls.

Only the H3 runtime settings, OpenAI quality instructions, setup validation, render
documentation/code, and final audit fields may change. Existing notebook outputs,
execution counts, manual source selection, and every unrelated cell are preserved.
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
    "h3-02-settings",
    "h3-07-drive",
    "h3-16-render-heading",
    "h3-17-render",
    "h3-22-audit-heading",
    "h3-23-audit",
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
    current_start_marker: str,
    reference_start_marker: str | None = None,
    end_marker: str,
    reference_end_marker: str | None = None,
) -> str:
    reference_start_marker = reference_start_marker or current_start_marker
    reference_end_marker = reference_end_marker or end_marker
    current_start = current.index(current_start_marker)
    current_end = current.index(end_marker, current_start)
    replacement = bounded_fragment(
        reference, reference_start_marker, reference_end_marker
    )
    return current[:current_start] + replacement + current[current_end:]


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
    settings_before = source(settings)
    settings_source = replace_bounded_fragment(
        settings_before,
        source(reference_cells["h3-02-settings"]),
        current_start_marker="H3_KEEP_NATIVE_AUDIO_IN_PAIR_CLIPS =",
        end_marker="# Positive correspondence language",
    )
    quality_settings_end = (
        "# OpenAI plans prompts"
        if "# OpenAI plans prompts" in settings_source
        else "# OpenAI is used only"
    )
    settings_source = replace_bounded_fragment(
        settings_source,
        source(reference_cells["h3-02-settings"]),
        current_start_marker='H3_OPENAI_QUALITY_GATE_INSTRUCTIONS = r"""',
        end_marker=quality_settings_end,
        reference_end_marker="# OpenAI plans prompts",
    )
    settings["source"] = settings_source.splitlines(keepends=True)

    drive = current_cells["h3-07-drive"]
    drive_source = source(drive)
    drive_source = replace_bounded_fragment(
        drive_source,
        source(reference_cells["h3-07-drive"]),
        current_start_marker='for child in ("clips",',
        end_marker="Path(HF_CACHE_DIR).mkdir",
        reference_start_marker="if (\n    not H3_REJECTED_VIDEO_SUBDIRECTORY",
    )
    drive["source"] = replace_bounded_fragment(
        drive_source,
        source(reference_cells["h3-07-drive"]),
        current_start_marker="OPENAI_CLIENT = None",
        end_marker=(
            'if H3_PROMPT_MODE == "openai_per_pair" or '
            "RUN_OPENAI_H3_QUALITY_GATE:"
        ),
    ).splitlines(keepends=True)

    current_cells["h3-16-render-heading"]["source"] = deepcopy(
        reference_cells["h3-16-render-heading"]["source"]
    )

    render = current_cells["h3-17-render"]
    render_source = source(render)
    render_source = replace_bounded_fragment(
        render_source,
        source(reference_cells["h3-17-render"]),
        current_start_marker="import glob",
        end_marker="H3_TEMPLATE =",
    )
    render_source = replace_bounded_fragment(
        render_source,
        source(reference_cells["h3-17-render"]),
        current_start_marker="def safe_name(value):",
        end_marker="def stream_command(command):",
    )
    render_source = replace_bounded_fragment(
        render_source,
        source(reference_cells["h3-17-render"]),
        current_start_marker="def h3_quality_sample_fractions():",
        end_marker="def archive_rejected_h3_clip(",
    )
    render_source = replace_bounded_fragment(
        render_source,
        source(reference_cells["h3-17-render"]),
        current_start_marker="def archive_rejected_h3_clip(",
        end_marker="def render_h3_pair(",
    )
    current_parallel_start = (
        "def render_h3_pairs_parallel(pairs):"
        if "def render_h3_pairs_parallel(pairs):" in render_source
        else "def render_h3_pairs_batched(pairs):"
    )
    render_source = replace_bounded_fragment(
        render_source,
        source(reference_cells["h3-17-render"]),
        current_start_marker=current_parallel_start,
        reference_start_marker="def render_h3_pairs_parallel(pairs):",
        end_marker="H3_CLIP_RECORDS = {}",
    )
    render_source = render_source.replace(
        'f"batched attempts in this execution; diagnostics were retained. Last error: "',
        'f"attempts in this execution; diagnostics were retained. Last error: "',
        1,
    )
    render_source = render_source.replace(
        "H3_CLIP_RECORDS.update(render_h3_pairs_batched(H3_PAIRS))",
        "H3_CLIP_RECORDS.update(render_h3_pairs_parallel(H3_PAIRS))",
        1,
    )
    render["source"] = render_source.splitlines(keepends=True)

    current_cells["h3-22-audit-heading"]["source"] = deepcopy(
        reference_cells["h3-22-audit-heading"]["source"]
    )
    audit = current_cells["h3-23-audit"]
    audit_source = source(audit)
    current_audit_start = (
        '"h3_max_parallel_transition_calls": H3_MAX_PARALLEL_TRANSITION_CALLS,'
        if '"h3_max_parallel_transition_calls": H3_MAX_PARALLEL_TRANSITION_CALLS,'
        in audit_source
        else '"h3_render_batch_size": H3_RENDER_BATCH_SIZE,'
    )
    audit["source"] = replace_bounded_fragment(
        audit_source,
        source(reference_cells["h3-23-audit"]),
        current_start_marker=current_audit_start,
        reference_start_marker=(
            '"h3_max_parallel_transition_calls": '
            "H3_MAX_PARALLEL_TRANSITION_CALLS,"
        ),
        end_marker='"quality_gate_all_pairs_passed":',
    ).splitlines(keepends=True)

    before_cells = cells_by_id(before)
    for cell_id, cell in current_cells.items():
        if cell_id not in ALLOWED_CELL_IDS and cell != before_cells[cell_id]:
            raise RuntimeError(f"Migration unexpectedly changed untouched cell {cell_id}")

    preserved_prefix_end = settings_before.index("H3_KEEP_NATIVE_AUDIO_IN_PAIR_CLIPS =")
    if settings_source[:preserved_prefix_end] != settings_before[:preserved_prefix_end]:
        raise RuntimeError("Migration changed settings before the H3 concurrency controls")
    for required in (
        'SOURCE_PROJECT_NAME = "manual_bases"',
        'SOURCE_RUN_DIRECTORY = "manual_base_oil_sciences_ideogram_1"',
    ):
        if required not in settings_source:
            raise RuntimeError(f"Migration failed to preserve manual source setting: {required}")

    NOTEBOOK_PATH.write_text(
        json.dumps(current, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Migrated H3 visual-exemplar and parallel-call cells: "
        + ", ".join(sorted(ALLOWED_CELL_IDS))
        + "; outputs, execution counts, source selection, and unrelated cells preserved."
    )


if __name__ == "__main__":
    main()
