from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "StillLife_Recursive_FlowMorph_Trajectory_Background_Mask.ipynb"
)


def _load_notebook() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


def test_background_mask_notebook_is_parseable() -> None:
    notebook, _ = _load_notebook()
    assert notebook["nbformat"] == 4
    assert all(cell.get("id") for cell in notebook["cells"])
    assert len({cell["id"] for cell in notebook["cells"]}) == len(
        notebook["cells"]
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))


def test_background_mask_notebook_keeps_source_out_of_generation() -> None:
    _, code = _load_notebook()
    assert "make_background_edit_mask(" in code
    assert "prepare_flux2_klein_masked_inpaint_inputs(" in code
    assert "composite_generated_on_background(" in code
    assert "callback_on_step_end=masked_inputs.callback_on_step_end" in code
    assert '"mask_polarity": "white_editable_black_protected"' in code
    assert '"source_used_only_to_derive_mask": True' in code
    assert '"source_used_as_latent_init": False' in code
    assert '"source_used_as_image_reference": False' in code
    assert "image=source" not in code
    assert "image=trial_source" not in code


def test_background_mask_notebook_has_github_colab_link() -> None:
    notebook, _ = _load_notebook()
    first_cell = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/MNoichl/FluxFlowMorph/"
        "blob/main/notebooks/"
        "StillLife_Recursive_FlowMorph_Trajectory_Background_Mask.ipynb"
    ) in first_cell
