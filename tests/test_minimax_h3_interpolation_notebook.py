from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "StillLife_MiniMax_H3_FL2V_Interpolation.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_source() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def markdown_source() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell.get("cell_type") == "markdown"
    )


def test_notebook_is_new_clean_parseable_and_colab_ready() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 24
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    assert "StillLife_MiniMax_H3_FL2V_Interpolation.ipynb" in "".join(notebook["cells"][0]["source"])
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell.get("source", [])), filename=f"h3-cell-{index}")


def test_h3_is_open_weight_local_fl2va_not_minimax_api() -> None:
    code = code_source()
    markdown = markdown_source()
    assert 'H3_MODEL_REPOSITORY = "Comfy-Org/MiniMax-H3"' in code
    assert 'H3_MODEL_REVISION = "eb8a16107c595128b3a578f82d2ce2f75920c355"' in code
    assert 'H3_DIFFUSION_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"' in code
    assert 'H3_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"' in code
    assert '"--disable-api-nodes"' in code
    assert '"--listen", "127.0.0.1"' in code
    assert '"h3_inference": "local_open_weights"' in code
    assert '"h3_api_used": False' in code
    assert "api.minimax" not in code.lower()
    assert "H3-Base-FL2VA" in markdown


def test_source_run_images_and_authored_prompts_are_loaded_read_only() -> None:
    code = code_source()
    assert 'SOURCE_RUN_DIRECTORY = None' in code
    assert 'SOURCE_PROJECT_NAME = "science_path_prompt_only_chimera"' in code
    assert "BASE_RECORDS = load_h3_anchor_records(SOURCE_RUN)" in code
    assert "H3_PAIRS = cyclic_h3_pairs(BASE_RECORDS)" in code
    assert 'record["authored_prompt"]' in code
    assert 'pair["left"]["authored_prompt"]' in code
    assert 'pair["right"]["authored_prompt"]' in code
    assert '"source_run_modified": False' in code
    assert 'H3_PROJECT_NAME = "minimax_h3_interpolations"' in code


def test_supplied_prompt_is_default_and_openai_option_sees_both_images_and_prompts() -> None:
    code = code_source()
    assert (
        '"The objects in #Image1 morphing into #Image2 . No camera movement, no panning, "'
        in code
    )
    assert 'H3_PROMPT_MODE = "template"' in code
    assert 'H3_LORA_TRIGGER' not in code
    assert 'H3_DURATION_SECONDS = 6.0' in code
    assert 'H3_JOB_TIMEOUT_SECONDS = 1800' in code
    assert 'strip_h3_source_only_tokens(pair["left"]["authored_prompt"])' in code
    assert 'strip_h3_source_only_tokens(pair["right"]["authored_prompt"])' in code
    assert 'if "RIJKSOIL" in payload["h3_prompt"]' in code
    assert 'OPENAI_MODEL = "gpt-5.6"' in code
    assert "OPENAI_CLIENT.responses.parse(" in code
    assert '"image_url": image_data_url(pair["left"]["resolved_path"])' in code
    assert '"image_url": image_data_url(pair["right"]["resolved_path"])' in code
    assert 'f"Picture 1 authored prompt:' in code
    assert 'f"Picture 2 authored prompt:' in code
    assert 'print("FINAL LOCAL H3 PROMPT:\\n" + plan["h3_prompt"])' in code
    assert '"openai_used_only_for_prompt_planning"' in code


def test_official_workflow_is_pinned_patched_and_pair_clips_resume() -> None:
    code = code_source()
    assert 'COMFYUI_REVISION = "2eb609766a749e3104485979615e062e401bab97"' in code
    assert 'H3_TEMPLATE_REVISION = "5097de61ef09fe75466716ac0b200515f5ea078f"' in code
    assert 'COMFY_CLI_VERSION = "1.15.0"' in code
    assert "patch_h3_ui_workflow(" in code
    assert "h3_ui_workflow_controls(workflow)" in code
    assert '"workflow_patch_version": H3_WORKFLOW_PATCH_VERSION' in code
    assert 'forbidden_demo_fragments = ("Vaporwave", "LATENT CONTROLNET", "DIRECTED BY COMFYUI")' in code
    assert "first_image=first_name" in code
    assert "last_image=last_name" in code
    assert '"--workflow", str(workflow_path), "--wait"' in code
    assert "H3_REUSE_EXISTING_CLIPS = True" in code
    assert 'prior.get("fingerprint") == fingerprint' in code
    assert 'for pair in H3_PAIRS:' in code
    assert 'H3_CLIP_RECORDS[pair["index"]] = render_h3_pair(pair)' in code


def test_square_canvas_is_verified_against_source_aspect() -> None:
    code = code_source()
    assert "H3_WIDTH = 768" in code
    assert "H3_HEIGHT = 768" in code
    assert "H3_ENFORCE_SOURCE_ASPECT = True" in code
    assert "source_aspects = [width / height for width, height in source_sizes]" in code
    assert "refusing to crop or stretch silently" in code


def test_loop_deduplicates_exact_endpoints_and_optional_rife_closes_wrap() -> None:
    code = code_source()
    assert "ImageOps.fit(" in code
    assert "for frame in frames[:-1]:" in code
    assert '"terminal_duplicate_in_video": False' in code
    assert '"generated_pair_audio_in_final_loop": False' in code
    assert "RUN_RIFE_POSTPROCESS = True" in code
    assert "RIFE_MULTIPLIER = 2" in code
    assert 'RIFE_FINAL_FPS = H3_FPS * RIFE_MULTIPLIER' in code
    assert 'shutil.copy2(H3_NATIVE_FRAME_PATHS[0], rife_input / f"{len(H3_NATIVE_FRAME_PATHS):07d}.png")' in code
    assert "if not np.array_equal(first_array, last_array):" in code
    assert "RIFE_DENSE_PATHS = dense_with_duplicate[:-1]" in code
