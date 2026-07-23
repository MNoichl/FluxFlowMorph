from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Trajectory_Init.ipynb"


def _load_notebook() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


def test_trajectory_notebook_is_parseable() -> None:
    notebook, _ = _load_notebook()
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    assert all(cell.get("id") for cell in notebook["cells"])
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in code_cells:
        ast.parse("".join(cell["source"]))


def test_trajectory_notebook_samples_zip_and_uses_selected_frame_as_init() -> None:
    _, code = _load_notebook()
    assert 'TRAJECTORY_ZIP_DIRECTORY = "' in code
    assert 'TRAJECTORY_ZIP_FILENAME = "background.zip"' in code
    assert "stage_regular_keyframes(" in code
    assert (
        '"selection_rule": '
        '"floor(keyframe_index * image_count / keyframe_count)"'
    ) in code
    assert "zip(ACTIVE_BASE_STAGES, TRAJECTORY_RECORDS, strict=True)" in code
    assert "make_strong_trajectory_reference(" in code
    assert "image=reference" in code
    assert "image=trial_reference" in code
    assert "make_soft_reference" not in code


def test_trajectory_notebook_keeps_prompt_count_editable() -> None:
    _, code = _load_notebook()
    assert "BASE_PROMPT_COUNT = None" in code
    assert "BASE_PROMPT_COUNT = len(BASE_STAGES)" not in code
    assert "BASE_PROMPT_COUNT must be between" not in code
    assert "list(BASE_STAGES)" in code


def test_trajectory_notebook_uses_true_spatial_img2img() -> None:
    _, code = _load_notebook()
    assert "TRAJECTORY_DENOISE_STRENGTH = 0.45" in code
    assert 'TRAJECTORY_GUIDE_MODE = "activity_mask"' in code
    assert "TRAJECTORY_GUIDE_ACTIVE_IS_LIGHT = True" in code
    assert "TRAJECTORY_OUTSIDE_CONTENT_OPACITY = 0.0" in code
    assert "make_trajectory_activity_guide(" in code
    assert "composite_generated_activity(" in code
    assert "prepare_flux2_klein_img2img_inputs(" in code
    assert "sigmas=list(trial_img2img.sigmas)" in code
    assert "latents=trial_img2img.latents" in code
    assert "sigmas=list(img2img.sigmas)" in code
    assert "latents=img2img.latents" in code
    assert "Treat the lighter regions" in code
    assert "Do not reproduce the guide as blobs or shading" in code
    assert "trajectory_activity_mask_path" in code
    assert "raw_generation_path" in code
    assert "TRAJECTORY_REMOVE_SYMMETRY_LANGUAGE = True" in code
    assert 'sys.modules.get("flowmorph_klein.trajectory")' in code
    assert "importlib.reload(trajectory_module)" in code


def test_trajectory_notebook_keeps_batched_flowmorph_and_rife_pipeline() -> None:
    _, code = _load_notebook()
    assert "FlowMorphSequenceSession(" in code
    assert "FLOWMORPH_ENDPOINT_BATCH_SIZE = 2" in code
    assert "FLOWMORPH_RENDER_BATCH_SIZE = 4" in code
    assert "FLOWMORPH_DECODE_BATCH_SIZE = 8" in code
    assert '"prompt_mode": "explicit_midpoint"' in code
    assert '"prompt_mode": "shared_midpoint"' in code
    assert "piecewise_source_midpoint_target_embeddings" in code
    assert "RIFE_BATCH_SIZE = 4" in code
    assert '"--batch-size", str(RIFE_BATCH_SIZE)' in code
    assert "recursive_flowmorph_trajectory_rife_ssim_loop.mp4" in code


def test_trajectory_notebook_keeps_expected_secret_filename() -> None:
    _, code = _load_notebook()
    assert 'OPENAI_KEY_FILENAME = "openaiapikey.txt"' in code
