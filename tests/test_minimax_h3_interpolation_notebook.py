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


def test_notebook_core_cells_are_parseable_and_colab_ready() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    expected_core_ids = {
        "h3-00-title",
        "h3-01-settings-heading",
        "h3-02-settings",
        "h3-03-research",
        "h3-04-setup-heading",
        "h3-05-setup",
        "h3-06-drive-heading",
        "h3-07-drive",
        "h3-08-anchors-heading",
        "h3-09-anchors",
        "h3-10-prompts-heading",
        "h3-11-prompts",
        "h3-12-models-heading",
        "h3-13-models",
        "h3-14-server-heading",
        "h3-15-server",
        "h3-16-render-heading",
        "h3-17-render",
        "h3-18-assembly-heading",
        "h3-19-assembly",
        "h3-20-rife-heading",
        "h3-21-rife",
        "h3-22b-border-heading",
        "h3-23b-border",
        "h3-24-flashvsr-heading",
        "h3-25-flashvsr",
        "h3-22-audit-heading",
        "h3-23-audit",
    }
    assert expected_core_ids <= {cell["id"] for cell in notebook["cells"]}
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    assert "StillLife_MiniMax_H3_FL2V_Interpolation.ipynb" in "".join(notebook["cells"][0]["source"])
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
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
    assert "No object dissolves into particles" in code
    assert "introduce no new intermediate textures" in code
    assert "No crumbling, shattering, shedding, scattering" in code
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


def test_border_flicker_correction_is_post_rife_anchor_safe_and_center_safe() -> None:
    code = code_source()
    markdown = markdown_source()
    assert "RUN_BORDER_FLICKER_CORRECTION = True" in code
    assert "BORDER_WIDTH_FRACTION = 0.025" in code
    assert "BORDER_FEATHER_FRACTION = 0.040" in code
    assert "BORDER_CORRECTION_STRENGTH = 0.65" in code
    assert "BORDER_MAX_RGB_SHIFT = 0.025" in code
    assert "H3_NATIVE_ANCHOR_INDICES.append(native_index)" in code
    assert "border_input_paths = RIFE_DENSE_PATHS" in code
    assert "stabilize_cyclic_borders(" in code
    assert "index * border_anchor_multiplier for index in H3_NATIVE_ANCHOR_INDICES" in code
    assert 'border_result.report["anchor_pixels_unchanged"]' in code
    assert 'border_result.report["center_pixels_unchanged"]' in code
    assert '"minimax_h3_border_stabilized_cyclic_loop.mp4"' in code
    assert "Correct low-frequency flicker only at the image margins" in markdown


def test_flashvsr_v11_is_final_streamed_cyclic_four_x_stage() -> None:
    code = code_source()
    markdown = markdown_source()
    assert "RUN_FLASHVSR_UPSCALE = True" in code
    assert "FLASHVSR_SCALE = 4.0" in code
    assert 'FLASHVSR_MODEL_REPOSITORY = "JunhaoZhuang/FlashVSR-v1.1"' in code
    assert 'FLASHVSR_MODEL_REVISION = "ad1aceeac60dbd288e51acea9096b821a8703bee"' in code
    assert 'FLASHVSR_REPOSITORY_REVISION = "b527c6f285fb30df530f5febc8b45764a789c961"' in code
    assert 'FLASHVSR_SPARSE_REPOSITORY_REVISION = "49d6c39e4dc0303442cda3bb758b3925d4399c49"' in code
    assert "release_local_h3_server(force_stop=True)" in code
    assert "FLASHVSR_DELETE_LOCAL_H3_CHECKPOINTS_IF_DISK_LOW = False" in code
    assert '"flashvsr_v11_streaming_runner.py"' in code
    assert '"minimax_h3_flashvsr_v1_1_x4_cyclic_loop.mp4"' in code
    assert "FLASHVSR_FINAL_VIDEO_PATH\n    if FLASHVSR_FINAL_VIDEO_PATH is not None" in code
    assert '"flashvsr_frame_count_preserved"' in code
    assert "lazy-loads temporal slices" in markdown
    assert "locality-constrained sparse attention" in markdown
