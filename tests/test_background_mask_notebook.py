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


def test_background_mask_notebook_loads_continuous_masks() -> None:
    _, code = _load_notebook()
    assert 'MASK_ZIP_FILENAME = "mask_2.zip"' in code
    assert "MASK_INVERT = True" in code
    assert "MASK_GAMMA = 1.0" in code
    assert "MASK_EXPANSION = 0" in code
    assert "MASK_FEATHER = 0.0" in code
    assert "prepare_grayscale_edit_mask(" in code
    assert "make_background_edit_mask(" not in code
    assert "do_binarize=False" in code
    assert "pipeline.mask_processor.config.do_binarize" in code
    assert '"continuous_values_preserved": True' in code
    assert '"mask_polarity": "white_editable_black_protected"' in code


def test_background_mask_notebook_uses_official_continuous_inpaint() -> None:
    _, code = _load_notebook()
    assert "Flux2KleinInpaintPipeline.from_pretrained(" in code
    assert "mask_image=mask_result.mask" in code
    assert "image=inpaint_source" in code
    assert "prepare_flux2_klein_masked_inpaint_inputs(" not in code
    assert "callback_on_step_end=masked_inputs.callback_on_step_end" not in code
    assert "MASK_DENOISE_STRENGTH = 1.0" in code
    assert "MASK_PREVIOUS_INIT_DENOISE_STRENGTH = 0.85" in code


def test_background_mask_notebook_uses_weak_previous_inpaint_source() -> None:
    _, code = _load_notebook()
    assert "PREVIOUS_INIT_ENABLED = True" in code
    assert "PREVIOUS_INIT_BLEND = 0.12" in code
    assert "PREVIOUS_INIT_BLUR = 16.0" in code
    assert "PREVIOUS_INIT_GRAIN_STRENGTH = 0.035" in code
    assert "make_soft_reference(" in code
    assert "composite_generated_on_background(" in code
    assert "if init_image is None:" in code
    assert "else MASK_PREVIOUS_INIT_DENOISE_STRENGTH" in code
    assert "previous = image.copy()" in code
    assert '"previous_init_used": previous_init is not None' in code
    assert '"source_mask_used_as_latent_init": False' in code
    assert '"source_used_as_image_reference": False' in code
    assert "image=mask_source" not in code


def test_background_mask_notebook_hands_final_anchors_to_flowmorph() -> None:
    _, code = _load_notebook()
    assert '"flowmorph_endpoint_path": str(output_path)' in code
    assert (
        '"flowmorph_endpoint_role": "final_background_composited_anchor"'
        in code
    )
    assert "def flowmorph_endpoint_path(record):" in code
    assert "flowmorph_endpoint_path(bootstrap_left)" in code
    assert "flowmorph_endpoint_path(bootstrap_right)" in code
    assert "file_sha256(flowmorph_endpoint_path(record))" in code
    assert "flowmorph_endpoint_path(record)," in code
    assert '"raw_generation_path",' in code
    assert '"trajectory_edit_mask_path",' in code
    assert (
        'type(SEQUENCE_RUNNER.pipeline).__name__ != "Flux2KleinPipeline"'
        in code
    )
    assert code.index("release_flux_pipeline()") < code.index(
        "SEQUENCE_RUNNER = FlowMorphRunner.from_config("
    )


def test_background_mask_notebook_masks_flowmorph_interior_frames() -> None:
    _, code = _load_notebook()
    assert "FLOWMORPH_SAVE_RAW_UNMASKED_FRAMES = True" in code
    assert "def flowmorph_edit_mask_path(record):" in code
    assert "def write_interpolated_flowmorph_mask(" in code
    assert "interpolate_grayscale_edit_masks(" in code
    assert "def composite_flowmorph_decoded_frame(" in code
    assert "composite_flowmorph_decoded_frame(" in code
    assert '"flowmorph_edit_mask_path": str(frame_record["mask_path"])' in code
    assert '"flowmorph_endpoint_role": "masked_flowmorph_midpoint"' in code
    assert (
        '"interior_masking": '
        '"interpolated_continuous_masks_exact_background"'
        in code
    )
    assert '"raw_flowmorph_path": (' in code
    assert 'raw_directory = round_directory / "raw_unmasked"' in code


def test_background_mask_notebook_has_github_colab_link() -> None:
    notebook, _ = _load_notebook()
    first_cell = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/MNoichl/FluxFlowMorph/"
        "blob/main/notebooks/"
        "StillLife_Recursive_FlowMorph_Trajectory_Background_Mask.ipynb"
    ) in first_cell
