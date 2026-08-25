from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Local_Video_Directory_Stitcher.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_source() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def test_local_stitcher_is_parseable_and_folder_driven() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert [cell["id"] for cell in notebook["cells"]] == [
        "stitch-00-title",
        "stitch-01-settings-heading",
        "stitch-02-settings",
        "stitch-03-discovery-heading",
        "stitch-04-discovery",
        "stitch-05-run-heading",
        "stitch-06-run",
        "stitch-07-preview-heading",
        "stitch-08-preview",
    ]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell["source"]))

    code = code_source()
    assert 'VIDEO_DIRECTORY = "/absolute/path/to/the/video_directory"' in code
    assert "natural_sort_key" in code
    assert '"-f", "concat"' in code
    assert '"-c", "copy"' in code
    assert '"-map", "0:a?"' in code
    assert "next_output_path" in code
    assert "RIFE" not in code
    assert "scale=" not in code
    assert "libx264" not in code
    assert "google.colab" not in code


def test_builder_refuses_to_overwrite_the_notebook() -> None:
    builder = (ROOT / "scripts" / "build_local_video_directory_stitcher_notebook.py").read_text(
        encoding="utf-8"
    )
    assert "if OUTPUT.exists():" in builder
    assert "Refusing to overwrite tracked/user notebook" in builder
