from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Prompt_Only.ipynb"
)


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_is_clean_parseable_and_has_colab_badge() -> None:
    notebook = load_notebook()
    # Users may add notes or scratch cells around the generated workflow.
    assert len(notebook["cells"]) >= 33
    assert notebook["nbformat"] == 4
    assert "StillLife_Recursive_FlowMorph_Prompt_Only.ipynb" in "".join(
        notebook["cells"][0]["source"]
    )

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        ast.parse("".join(cell.get("source", [])))


def test_notebook_is_prompt_only_without_mask_or_trajectory_inputs() -> None:
    code = code_source(load_notebook())
    assert "BASE_STAGES = [" in code
    assert (
        "BASE_PROMPT_COUNT = None  # None uses every entry in BASE_STAGES."
        in code
    )
    assert "if BASE_PROMPT_COUNT is None:" in code
    assert "BASE_PROMPT_COUNT = len(BASE_STAGES)" in code
    assert "MASK_" not in code
    assert "TRAJECTORY_" not in code
    assert "prepare_grayscale_edit_mask" not in code
    assert "stage_regular_keyframes" not in code
    assert "upload" not in code.lower()


def test_current_quality_speed_and_recursive_round_defaults_are_exposed() -> None:
    code = code_source(load_notebook())
    expected = (
        'FLOWMORPH_ROUND_SPECS = [\n'
        '    {"midpoint_count": 1, "prompt_mode": "explicit_midpoint"},\n'
        '    {"midpoint_count": 10, "prompt_mode": "shared_midpoint"},\n'
        "]"
    )
    assert expected in code
    assert "FLOWMORPH_FIT_LORA_SCALE = 1.2" in code
    assert "FLOWMORPH_RENDER_LORA_SCALE = 1.2" in code
    assert "FLOWMORPH_GUIDANCE_SCALE = 7.0" in code
    assert "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 50" in code
    assert "FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 50" in code
    assert "IMAGE_INFERENCE_STEPS = 50" in code
    assert "IMAGE_GUIDANCE_SCALE = 7.0" in code
    assert "IMAGE_LORA_SCALE = 1.2" in code
    assert "FLOWMORPH_ENDPOINT_BATCH_SIZE = 2" in code
    assert "FLOWMORPH_RENDER_BATCH_SIZE = 4" in code
    assert "FLOWMORPH_DECODE_BATCH_SIZE = 8" in code
    assert 'FLOWMORPH_CFG_EXECUTION = "batched"' in code
    assert "FLOWMORPH_RENDER_INDICES = [*range(35, 100, 5), 99]" in code
    assert "FLUX_PROMPT_MAX_SEQUENCE_LENGTH = 512" in code


def test_anchors_use_correct_weak_conventional_img2img_initialization() -> None:
    code = code_source(load_notebook())
    assert "make_soft_reference(" in code
    assert "prepare_flux2_klein_img2img_inputs(" in code
    assert "BASE_REFERENCE_STRENGTH" not in code
    assert "REFERENCE_BACKGROUND" not in code
    assert "BASE_REFERENCE_BLUR = 16.0" in code
    assert "BASE_REFERENCE_GRAIN_STRENGTH = 0.035" in code
    assert "BASE_REFERENCE_DENOISE_STRENGTH = 0.75" in code
    assert "reference_blend=1.0" in code
    assert "background_rgb=" not in code
    assert "blurred_grained_previous_without_flat_canvas" in code
    assert 'kwargs["sigmas"] = list(generation_inputs.sigmas)' in code
    assert 'kwargs["latents"] = generation_inputs.latents' in code
    assert 'kwargs["image"] = reference' not in code


def test_each_unique_endpoint_is_fitted_once_and_reconstructed_canonically() -> None:
    code = code_source(load_notebook())
    assert "ENDPOINT_CACHE = {}" in code
    assert "def fit_sequence_endpoints(records, progress_label):" in code
    assert 'if record["uid"] not in ENDPOINT_CACHE' in code
    assert "SEQUENCE_ENDPOINT_RECONSTRUCTION_ROOT" in code
    assert "ENDPOINT_RECONSTRUCTION_PATHS = {}" in code
    assert "SEQUENCE_SESSION.render_endpoint_reconstructions(" in code
    assert "def render_canonical_endpoint_reconstructions(" not in code
    assert '"canonical_endpoint_reconstruction_used": True' not in code
    assert 'final_record["canonical_endpoint_reconstruction_used"] = True' in code
    assert '"model_loads": 1' in code
    assert '"backward_probes": 1' in code


def test_multi_round_rendering_uses_explicit_then_shared_midpoint_prompts() -> None:
    code = code_source(load_notebook())
    assert (
        "for round_number, round_spec in enumerate("
        "FLOWMORPH_ROUND_SPECS, start=1):"
    ) in code
    assert "fractions = [index / (midpoint_count + 1)" in code
    assert '"one_prompt_reused_for_every_rendered_alpha": prompt_mode == "shared_midpoint"' in code
    assert "midpoint_conditionings=[shared] * midpoint_count" in code
    assert "fit_sequence_endpoints(" in code
    assert "SEQUENCE_SESSION.render_midpoints(" in code


def test_tone_stabilization_flicker_audit_and_rife_finishing_are_retained() -> None:
    code = code_source(load_notebook())
    assert "TEMPORAL_TONE_STABILIZATION_ENABLED = False" in code
    assert "from flowmorph_klein.temporal_tone import (" in code
    assert "stabilize_cyclic_tone(" in code
    assert "from flowmorph_klein.flicker_diagnostics import (" in code
    assert "def diagnose_cyclic_flicker(" not in code
    assert "RUN_RIFE_POSTPROCESS = True" in code
    assert "VIDEO_SLOWDOWN_FACTOR = 3.0" in code
    assert "RIFE_MULTIPLIER = int(round(2 * VIDEO_SLOWDOWN_FACTOR))" in code
    assert "RIFE_FINAL_FPS = 24.0" in code
    assert "recursive_flowmorph_prompt_only_rife_ssim_loop.mp4" in code
