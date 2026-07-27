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


def _literal_assignment(notebook: dict, name: str):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Notebook assignment not found: {name}")


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
    assert '"continuous_values_preserved": True' in code
    assert '"mask_polarity": "white_editable_black_protected"' in code


def test_background_mask_notebook_generates_before_masking() -> None:
    _, code = _load_notebook()
    assert "Flux2KleinPipeline.from_pretrained(" in code
    assert "Flux2KleinInpaintPipeline" not in code
    assert "mask_image=mask_result.mask" not in code
    assert "image=inpaint_source" not in code
    assert "prepare_flux2_klein_masked_inpaint_inputs(" not in code
    assert "callback_on_step_end=masked_inputs.callback_on_step_end" not in code
    assert "IMAGE_INFERENCE_STEPS = 50" in code
    assert "MASK_PREVIOUS_INIT_DENOISE_STRENGTH = 0.85" in code
    assert '"mask_used_by_model": False' in code
    assert (
        '"mask_application": "post_decode_continuous_alpha_composite"'
        in code
    )
    assert code.index("raw_image = result.images[0].convert") < code.index(
        "final_image = composite_generated_on_background("
    )


def test_background_mask_notebook_uses_weak_previous_latent_img2img() -> None:
    _, code = _load_notebook()
    assert "PREVIOUS_INIT_ENABLED = True" in code
    assert "PREVIOUS_INIT_BLEND = 0.12" in code
    assert "PREVIOUS_INIT_BLUR = 16.0" in code
    assert "PREVIOUS_INIT_GRAIN_STRENGTH = 0.035" in code
    assert "make_soft_reference(" in code
    assert "composite_generated_on_background(" in code
    assert "if init_image is None:" in code
    assert "prepare_flux2_klein_img2img_inputs(" in code
    assert "strength=MASK_PREVIOUS_INIT_DENOISE_STRENGTH" in code
    assert "sigmas=list(generation_inputs.sigmas)" in code
    assert "latents=generation_inputs.latents" in code
    assert "previous = image.copy()" in code
    assert '"previous_init_used": previous_init is not None' in code
    assert '"source_mask_used_as_latent_init": False' in code
    assert '"source_used_as_image_reference": False' in code
    assert "image=mask_source" not in code


def test_background_mask_notebook_removes_sparse_prompt_regression() -> None:
    notebook, code = _load_notebook()
    assert "IMAGE_GUIDANCE_SCALE = 7.0" in code
    assert "IMAGE_LORA_SCALE = 1.2" in code
    assert "MASK_REMOVE_SPARSE_PROMPT_LANGUAGE = True" in code
    assert '"soft translucent washes": "opaque layered oil paint"' in code
    assert '"chalky faded pigments": "deep luminous oil pigments"' in code
    assert "complete full-frame image before masking" in code
    assert "large unpainted areas" in code
    stages = _literal_assignment(notebook, "BASE_STAGES")
    prompts = "\n".join(stage["prompt"] for stage in stages)
    assert "a flowing garland of night-blooming moonflowers" in prompts
    assert "a full garland of medicinal poppy, foxglove, and willow" in prompts
    assert "a dense interlacing arabesque lattice" in prompts
    assert "a loose spray of night-blooming moonflowers" not in prompts
    assert "# if FLOWMORPH_FIT_LORA_SCALE != IMAGE_LORA_SCALE:" in code
    assert "# if FLOWMORPH_RENDER_LORA_SCALE != IMAGE_LORA_SCALE:" in code
    assert "# if FLOWMORPH_GUIDANCE_SCALE != IMAGE_GUIDANCE_SCALE:" in code


def test_background_mask_notebook_does_not_reapply_masks_in_flowmorph() -> None:
    _, code = _load_notebook()
    assert "interpolate_grayscale_edit_masks" not in code
    assert "flowmorph_edit_mask_path" not in code
    assert "write_interpolated_flowmorph_mask" not in code
    assert "composite_flowmorph_decoded_frame" not in code
    assert "interior_masking" not in code
    assert 'raw_directory = round_directory / "raw_unmasked"' not in code


def test_background_mask_notebook_has_github_colab_link() -> None:
    notebook, _ = _load_notebook()
    first_cell = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/MNoichl/FluxFlowMorph/"
        "blob/main/notebooks/"
        "StillLife_Recursive_FlowMorph_Trajectory_Background_Mask.ipynb"
    ) in first_cell
