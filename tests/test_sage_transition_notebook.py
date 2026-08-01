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
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_is_clean_parseable_and_linked_from_colab() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 27
    assert "StillLife_SAGE_Transition_Video.ipynb" in "".join(
        notebook["cells"][0]["source"]
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell["source"]), filename=f"cell-{index}")


def test_all_visible_images_use_flux2_klein_and_rijksoil() -> None:
    source = code_source()
    assert 'MODEL_ID = "Runware/BFL-FLUX.2-klein-base-9B"' in source
    assert 'LORA_SOURCE = "MaxNoichl/RIJKSOIL_FLUX2_KLEIN9B_lora_01_000001650"' in source
    assert 'LORA_TRIGGER = "RIJKSOIL"' in source
    assert "Flux2KleinPipeline.from_pretrained(" in source
    assert "load_flux2_lora(" in source
    assert "pipeline.fuse_lora(" in source
    assert 'image=[left_image, right_image, init_image]' in source
    assert "prompt_embeds=prompt_package.prompt_embeds" in source
    assert "prepare_flux2_klein_img2img_inputs(" in source
    assert "interpolate_conditioning(" in source


def test_stable_video_renderer_and_credentials_are_absent() -> None:
    source = code_source()
    assert "StableVideoDiffusionPipelineControlNeXtReverse" not in source
    assert "SAGE_FCVG_REPOSITORY" not in source
    assert "SAGE_BASE_MODEL_ID" not in source
    assert "diffusers==0.27.0" not in source
    assert "controlnext.safetensors" not in source
    assert "unet.safetensors" not in source
    assert "HF_TOKEN" not in source
    assert "venv_python" not in source


def test_anchor_workflow_keeps_lora_and_weak_previous_image_continuity() -> None:
    source = code_source()
    assert "make_soft_reference(" in source
    assert "reference_blend=1.0" in source
    assert '"blurred_grained_previous_without_flat_canvas"' in source
    assert "REFERENCE_BACKGROUND" not in source
    assert "background_rgb=" not in source


def test_sage_structure_is_pinned_inspectable_and_precedes_flux_render() -> None:
    source = code_source()
    assert 'SAGE_REPOSITORY_URL = "https://github.com/kan32501/SAGE.git"' in source
    assert (
        'SAGE_REPOSITORY_COMMIT = '
        '"5a30e6bfb035e2c243d90d4804ebda2addecacf4"'
    ) in source
    assert 'SAGE_MASK_MODE = "grabcut"' in source
    assert "cv2.grabCut(" in source
    assert "subprocess.check_call(SAGE_COMMAND)" in source
    assert "sage_preparation_manifest.json" in source
    assert "source_matched_lines.png" in source
    assert "sage_middle_conditions.png" in source
    assert source.index("subprocess.check_call(SAGE_COMMAND)") < source.index(
        "Encoding each unique endpoint prompt once"
    )


def test_pytlsd_is_installed_and_verified_in_the_active_interpreter() -> None:
    source = code_source()
    assert '"pytlsd==0.0.2"' in source
    assert '"pybind11>=2.10"' in source
    assert '"import omegaconf, pytlsd; from pytlsd import lsd"' in source
    assert "sage_import_probe = subprocess.run(" in source
    assert "Repairing missing SAGE line-detector dependencies" in source
    assert "os.kill(os.getpid(), signal.SIGKILL)" in source
    assert "import numpy.char" in source
    assert "import scipy.sparse" in source
    assert "Do not reinstall again before restarting" in source


def test_flux_sage_round_is_cyclic_resumable_and_streamed() -> None:
    source = code_source()
    assert 'SAGE_GENERATED_FRAMES_PER_GAP = 13' in source
    assert 'SAGE_REUSE_COMPLETED_GAPS = True' in source
    assert 'SAGE_OUTPUT_FPS = 12.0' in source
    assert 'SAGE_FLUX_IMG2IMG_STRENGTH = 0.72' in source
    assert 'SAGE_PREVIOUS_FRAME_BLEND = 0.24' in source
    assert 'SAGE_STRUCTURE_INIT_STRENGTH = 0.24' in source
    assert "write_json_atomic(metadata_path, partial)" in source
    assert "flux_sage_transition.mp4" in source
    assert "sage_sequence_manifest.json" in source
    assert "RUN_RIFE_POSTPROCESS" not in source
    assert "FLOWMORPH_ROUND_SPECS" not in source


def test_runner_only_prepares_sage_structures_for_external_flux() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source, filename=str(RUNNER))
    assert "TwoViewPipeline" in source
    assert "points_in_mask" in source
    assert "normalize_lines" in source
    assert "linear_sum_assignment" in source
    assert "cubic_bezier" in source
    assert "interpolate_structures" in source
    assert "rasterize_conditions" in source
    assert '"renderer": "external_flux2_klein_with_project_lora"' in source
    assert 'anchors[(gap_index + 1) % len(anchors)]' in source
    assert '"condition_alphas"' in source
    assert '"generative_backend_loaded": False' in source
