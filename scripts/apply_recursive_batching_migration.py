"""Apply batching cells to the tracked working notebook without erasing user state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Vision.ipynb"
BUILDER = ROOT / "scripts" / "build_recursive_flowmorph_vision_notebook.py"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def find_code_cell(notebook: dict, marker: str) -> dict:
    matches = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and marker in source(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def insert_after_once(text: str, anchor: str, addition: str) -> str:
    if addition.strip() in text:
        return text
    if text.count(anchor) != 1:
        raise RuntimeError(f"Expected one notebook insertion anchor: {anchor!r}")
    return text.replace(anchor, anchor + addition, 1)


def replace_source_preserving_execution(target: dict, reference: dict) -> None:
    target["source"] = list(reference["source"])


def main() -> None:
    current = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="flowmorph-batching-") as temporary:
        temporary_root = Path(temporary)
        base_output = temporary_root / "base.ipynb"
        reference_output = temporary_root / "reference.ipynb"
        environment = dict(os.environ)
        environment["FLOWMORPH_BASE_NOTEBOOK_OUTPUT"] = str(base_output)
        environment["FLOWMORPH_NOTEBOOK_OUTPUT"] = str(reference_output)
        subprocess.check_call(
            [sys.executable, str(BUILDER)],
            cwd=ROOT,
            env=environment,
        )
        reference = json.loads(reference_output.read_text(encoding="utf-8"))

    settings_cell = current["cells"][2]
    settings = source(settings_cell)
    settings = insert_after_once(
        settings,
        "FLOWMORPH_STREAM_PAIRS_PER_CHUNK = 3\n",
        (
            "FLOWMORPH_ENDPOINT_BATCH_SIZE = 2\n"
            "FLOWMORPH_RENDER_BATCH_SIZE = 4\n"
            "FLOWMORPH_DECODE_BATCH_SIZE = 8\n"
            'FLOWMORPH_CFG_EXECUTION = "batched"\n'
            "FLOWMORPH_BATCH_OOM_BACKOFF = True\n"
            "OPENAI_CONCURRENCY = 6\n"
        ),
    )
    settings = insert_after_once(
        settings,
        "RIFE_SCALE = 1.0\n",
        "RIFE_BATCH_SIZE = 4\n",
    )
    settings_cell["source"] = settings.splitlines(keepends=True)

    for marker in (
        "if not 3 <= BASE_PROMPT_COUNT",
        "Sequence-native experimental FlowMorph contract enabled.",
        "for round_number, round_spec in enumerate(FLOWMORPH_ROUND_SPECS, start=1):",
        "RIFE_RUNNER_SOURCE = r'''",
        'command = [\n        sys.executable, "-u", str(RIFE_RUNNER_PATH),',
    ):
        replace_source_preserving_execution(
            find_code_cell(current, marker),
            find_code_cell(reference, marker),
        )

    final_video_cell = find_code_cell(current, "RIFE_FINAL_VIDEO_PATH")
    final_video = source(final_video_cell)
    final_video = insert_after_once(
        final_video,
        '        "rife_multiplier": RIFE_MULTIPLIER,\n',
        '        "rife_batch_size": RIFE_BATCH_SIZE,\n',
    )
    final_video_cell["source"] = final_video.splitlines(keepends=True)

    NOTEBOOK_PATH.write_text(
        json.dumps(current, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Applied recursive batching migration while preserving prompts, local settings, "
        "cell IDs, execution counts, and outputs."
    )


if __name__ == "__main__":
    main()
