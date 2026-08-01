from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "StillLife_SAGE_Transition_Video.ipynb"
RUNNER = ROOT / "scripts" / "sage_still_sequence_runner.py"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_source() -> str:
    notebook = load_notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_is_clean_parseable_and_linked_from_colab() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 25
    assert "StillLife_SAGE_Transition_Video.ipynb" in "".join(
        notebook["cells"][0]["source"]
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell["source"]), filename=f"cell-{index}")


def test_anchor_workflow_keeps_lora_and_weak_previous_image_continuity() -> None:
    source = code_source()
    assert 'LORA_TRIGGER = "RIJKSOIL"' in source
    assert 'BASE_PROMPT_COUNT = None' in source
    assert "make_soft_reference(" in source
    assert "prepare_flux2_klein_img2img_inputs(" in source
    assert "reference_blend=1.0" in source
    assert '"blurred_grained_previous_without_flat_canvas"' in source
    assert "REFERENCE_BACKGROUND" not in source
    assert "background_rgb=" not in source


def test_sage_is_pinned_and_isolated_from_flux_diffusers() -> None:
    source = code_source()
    assert 'SAGE_REPOSITORY_URL = "https://github.com/kan32501/SAGE.git"' in source
    assert (
        'SAGE_REPOSITORY_COMMIT = '
        '"5a30e6bfb035e2c243d90d4804ebda2addecacf4"'
    ) in source
    assert 'SAGE_FCVG_REPOSITORY = "melmass/FCVG"' in source
    assert (
        'SAGE_BASE_MODEL_ID = '
        '"stabilityai/stable-video-diffusion-img2vid-xt-1-1"'
    ) in source
    assert '"diffusers==0.27.0"' in source
    assert '"huggingface_hub==0.25.2"' in source
    assert '"transformers==4.37.2"' in source
    assert '"--system-site-packages"' in source
    assert "release_flux_pipeline()" in source
    assert 'DELETE_LOCAL_FLUX_CACHE_BEFORE_SAGE = True' in source


def test_masks_and_structural_preflight_are_visible_before_render() -> None:
    source = code_source()
    assert 'SAGE_MASK_MODE = "grabcut"' in source
    assert "cv2.grabCut(" in source
    assert 'SAGE_MASK_SOURCE_DIRECTORY = None' in source
    assert '"mask_path": mask_by_uid[record["uid"]]["mask_path"]' in source
    assert '"--phase", "prepare"' in source
    assert 'sage_preparation_manifest.json' in source
    assert 'source_matched_lines.png' in source
    assert 'sage_middle_conditions.png' in source
    prepare_position = source.index('[*SAGE_COMMAND_BASE, "--phase", "prepare"]')
    render_position = source.index('[*SAGE_COMMAND_BASE, "--phase", "render"]')
    assert prepare_position < render_position


def test_one_round_is_cyclic_resumable_and_uses_direct_sage_frames() -> None:
    source = code_source()
    assert 'SAGE_GENERATED_FRAMES_PER_GAP = 13' in source
    assert 'SAGE_REUSE_COMPLETED_GAPS = True' in source
    assert 'SAGE_OUTPUT_FPS = 12.0' in source
    assert 'SAGE_SYNTHETIC_FLOW_SCALE = 0.16' in source
    assert 'SAGE_TRAJECTORY_BEND = 0.04' in source
    assert 'sage_sequence_manifest.json' in source
    assert 'SAGE_DISPLAY_EACH_GAP_VIDEO = False' in source
    assert 'RUN_RIFE_POSTPROCESS' not in source
    assert 'FLOWMORPH_ROUND_SPECS' not in source


def test_runner_implements_sage_structural_core_and_still_disclosure() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source, filename=str(RUNNER))
    assert "TwoViewPipeline" in source
    assert "points_in_mask" in source
    assert "normalize_lines" in source
    assert "linear_sum_assignment" in source
    assert "cubic_bezier" in source
    assert "interpolate_structures" in source
    assert "rasterize_conditions" in source
    assert "StableVideoDiffusionPipelineControlNeXtReverse" in source
    assert '"still_image_adaptation": True' in source
    assert (
        '"still_motion_fallback": '
        '"deterministic synthetic cubic control vector"'
    ) in source
    assert 'anchors[(gap_index + 1) % len(anchors)]' in source
    assert 'for source_path in paths[:-1]:' in source
    assert '"sage_cyclic_one_round.mp4"' in source


def test_credentials_are_not_embedded_in_notebook_or_runner_command() -> None:
    source = code_source()
    assert 'HF_TOKEN_FILENAME = "hftoken.txt"' in source
    assert 'userdata.get("HF_TOKEN")' in source
    assert 'getpass.getpass("Hugging Face access token (hidden): ")' in source
    assert '"credential_value_recorded": False' in source
    assert 'sage_environment["HF_TOKEN"] = hf_token' in source
    assert '"--token"' not in source
