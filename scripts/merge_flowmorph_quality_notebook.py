"""Apply the quality/streaming cells without replacing user notebook state."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Vision.ipynb"


def heading_index(notebook: dict, prefix: str) -> int:
    matches = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "markdown"
        and "".join(cell.get("source", [])).strip().startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one notebook heading beginning {prefix!r}; found {matches}")
    return matches[0]


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def set_assignment(value: str, name: str, expression: str) -> str:
    pattern = rf"(?m)^{re.escape(name)}\s*=.*$"
    replacement = f"{name} = {expression}"
    updated, count = re.subn(pattern, replacement, value, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update setting {name}")
    return updated


def insert_after_assignment(value: str, name: str, addition: str) -> str:
    if addition.split("=", 1)[0].strip() in value:
        return value
    pattern = rf"(?m)^({re.escape(name)}\s*=.*\n)"
    updated, count = re.subn(pattern, rf"\1{addition}", value, count=1)
    if count != 1:
        raise RuntimeError(f"Could not insert settings after {name}")
    return updated


def main(reference_path: str) -> None:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    target = json.loads(TARGET.read_text(encoding="utf-8"))

    overview = source(target["cells"][0]).replace(
        "while reusing one image-aware interpolation prompt across those ten renders",
        "using piecewise source→shared-midpoint→target embedding conditioning",
    )
    set_source(target["cells"][0], overview)
    settings_intro = source(target["cells"][1]).replace(
        "generates one shared interpolation prompt per gap, and renders ten interior alphas from each cached endpoint pair",
        "generates one shared interpolation prompt per gap, and renders ten interior alphas while blending conditioning source→midpoint→target",
    ).replace(
        "The research defaults use 100 optimization steps per endpoint. This art-production notebook starts at 30, keeps one model loaded, probes it once, and checkpoints every 10 steps. Increase the fit steps only if a one-pair comparison shows a visible benefit.",
        "The quality-first defaults use 100 optimization steps per endpoint, keep one model loaded, probe it once, and checkpoint every 25 steps. An optional one-gap quality gate renders before the full recursion, and completed PNGs stream to Drive in small pair chunks during the full run.",
    )
    set_source(target["cells"][1], settings_intro)

    # Preserve the user's complete editable settings cell, changing only the
    # requested quality controls and adding the new test/streaming controls.
    target_settings_index = heading_index(target, "## 1.") + 1
    settings = source(target["cells"][target_settings_index])
    for name, expression in {
        "FLOWMORPH_FIT_LORA_SCALE": "1.0",
        "FLOWMORPH_RENDER_LORA_SCALE": "1.0",
        "FLOWMORPH_GUIDANCE_SCALE": "4.0",
        "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS": "100",
        "FLOWMORPH_TARGET_OPTIMIZATION_STEPS": "100",
        "FLOWMORPH_CHECKPOINT_EVERY": "25",
    }.items():
        settings = set_assignment(settings, name, expression)
    settings = insert_after_assignment(
        settings,
        "FLOWMORPH_CHECKPOINT_EVERY",
        "FLOWMORPH_STREAM_PAIRS_PER_CHUNK = 3\n"
        "FLOWMORPH_STREAM_DISPLAY_PROGRESS = True\n",
    )
    settings = insert_after_assignment(
        settings,
        "TRIAL_DISPLAY_MAX_WIDTH",
        "RUN_FLOWMORPH_ONE_GAP_TEST = True\n"
        "FLOWMORPH_ONE_GAP_TEST_INDEX = 0\n"
        "FLOWMORPH_ONE_GAP_TEST_ALPHAS = [0.25, 0.5, 0.75]\n",
    )
    set_source(target["cells"][target_settings_index], settings)

    # Replace only generated control cells. Outputs, extra recovery cells,
    # commented settings, prompts, and every unrelated cell remain untouched.
    for prefix in ("## 5.", "## 6.", "## 9."):
        target_index = heading_index(target, prefix)
        reference_index = heading_index(reference, prefix)
        if prefix != "## 5.":
            target["cells"][target_index]["source"] = reference["cells"][reference_index]["source"]
        if prefix in {"## 5.", "## 9."}:
            target_code = target["cells"][target_index + 1]
            retained_outputs = target_code.get("outputs", [])
            retained_execution_count = target_code.get("execution_count")
            target_code["source"] = reference["cells"][reference_index + 1]["source"]
            target_code["outputs"] = retained_outputs
            target_code["execution_count"] = retained_execution_count

    full_run_heading = heading_index(reference, "### Full recursive FlowMorph run")
    target_section_10 = heading_index(target, "## 10.")
    existing_full_run = [
        index
        for index, cell in enumerate(target["cells"])
        if cell.get("cell_type") == "markdown"
        and source(cell).strip().startswith("### Full recursive FlowMorph run")
    ]
    if existing_full_run:
        full_index = existing_full_run[0]
        target["cells"][full_index] = reference["cells"][full_run_heading]
        target["cells"][full_index + 1] = reference["cells"][full_run_heading + 1]
    else:
        target["cells"][target_section_10:target_section_10] = [
            reference["cells"][full_run_heading],
            reference["cells"][full_run_heading + 1],
        ]

    target_section_10 = heading_index(target, "## 10.")
    reference_section_10 = heading_index(reference, "## 10.")
    target["cells"][target_section_10]["source"] = reference["cells"][reference_section_10]["source"]

    TARGET.write_text(json.dumps(target, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {TARGET} without replacing user settings, outputs, or extra cells")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: merge_flowmorph_quality_notebook.py REFERENCE_NOTEBOOK")
    main(sys.argv[1])
