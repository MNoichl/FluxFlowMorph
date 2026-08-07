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
    assert "SOURCE_RUN_DIRECTORY =" in code
    assert 'SOURCE_PROJECT_NAME = "science_path_prompt_only_chimera"' in code
    assert "source_project_directory = drive_base / SOURCE_PROJECT_NAME" in code
    assert "source_project_directory / configured_source" in code
    assert '"explicit_basename"' in code
    assert "BASE_RECORDS = load_h3_anchor_records(SOURCE_RUN)" in code
    assert "H3_PAIRS = cyclic_h3_pairs(BASE_RECORDS)" in code
    assert 'record["authored_prompt"]' in code
    assert 'pair["left"]["authored_prompt"]' in code
    assert 'pair["right"]["authored_prompt"]' in code
    assert '"source_run_modified": False' in code
    assert 'H3_PROJECT_NAME = "minimax_h3_interpolations"' in code


def test_official_fl2va_openai_writer_sees_both_images_and_prompts_and_is_visible() -> None:
    code = code_source()
    assert 'H3_PROMPT_MODE = "openai_per_pair"' in code
    assert 'H3_LORA_TRIGGER' not in code
    assert 'H3_DURATION_SECONDS = 6.0' in code
    assert 'H3_JOB_TIMEOUT_SECONDS = 1800' in code
    assert 'H3_OPENAI_PROMPT_GUIDE_VERSION = "minimax-h3-fl2va-positive-correspondence-v1"' in code
    assert "H3_OPENAI_PROMPT_WRITER_INSTRUCTIONS = r\"\"\"" in code
    assert "first-frame state -> observable intermediate changes -> progressively" in code
    assert "Match forms primarily by screen region, silhouette, scale, visual role" in code
    assert "Do not merely describe two static images" in code
    assert 'print("OPENAI H3 PROMPT-WRITER INSTRUCTIONS' in code
    assert 'strip_h3_source_only_tokens(pair["left"]["authored_prompt"])' in code
    assert 'strip_h3_source_only_tokens(pair["right"]["authored_prompt"])' in code
    assert 'if "RIJKSOIL" in payload["h3_prompt"]' in code
    assert 'OPENAI_MODEL = "gpt-5.6"' in code
    assert 'OPENAI_IMAGE_DETAIL = "original"' in code
    assert "OPENAI_CLIENT.responses.parse(" in code
    assert '"image_url": image_data_url(pair["left"]["resolved_path"])' in code
    assert '"image_url": image_data_url(pair["right"]["resolved_path"])' in code
    assert '"Picture 1 authored image prompt (semantic context only):\\n"' in code
    assert '"Picture 2 authored image prompt (semantic context only):\\n"' in code
    assert "class H3ObjectCorrespondence(BaseModel):" in code
    assert "object_correspondences: list[H3ObjectCorrespondence]" in code
    assert "H3_DISALLOWED_GENERATED_TRANSITION_TERMS = (" in code
    assert "def validate_openai_motion_proposal(proposal):" in code
    assert "validate_openai_motion_proposal(proposal)" in code
    assert '"prompt_writer_instructions": H3_OPENAI_PROMPT_WRITER_INSTRUCTIONS' in code
    assert '"disallowed_generated_terms": H3_DISALLOWED_GENERATED_TRANSITION_TERMS' in code
    assert 'print("POSITIVE OBJECT CORRESPONDENCE MAP:")' in code
    assert 'print("GENERATED TRANSITION PROMPT SENT TO LOCAL H3:\\n" + plan["h3_prompt"])' in code
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


def test_flashvsr_v11_is_final_streamed_cyclic_net_two_x_stage() -> None:
    code = code_source()
    markdown = markdown_source()
    assert "RUN_FLASHVSR_UPSCALE = True" in code
    assert "FLASHVSR_SCALE = 4.0" in code
    assert "FLASHVSR_INPUT_RESIZE_FACTOR = 0.5" in code
    assert 'FLASHVSR_MODEL_REPOSITORY = "JunhaoZhuang/FlashVSR-v1.1"' in code
    assert 'FLASHVSR_MODEL_REVISION = "ad1aceeac60dbd288e51acea9096b821a8703bee"' in code
    assert 'FLASHVSR_REPOSITORY_URL = "https://github.com/naxci1/ComfyUI-FlashVSR_Stable.git"' in code
    assert 'FLASHVSR_REPOSITORY_REVISION = "f7f55bae4c0e82b18b190d4b62a977995507c51c"' in code
    assert 'FLASHVSR_ATTENTION_BACKEND = "sparse_sage_attention"' in code
    assert 'release_h3 = globals().get("release_local_h3_server")' in code
    assert '"force_stop" in inspect.signature(release_h3).parameters' in code
    assert "release_h3(force_stop=True)" in code
    assert "release_h3()" in code
    assert "FLASHVSR_DELETE_LOCAL_H3_CHECKPOINTS_IF_DISK_LOW" in code
    assert '"flashvsr_v11_streaming_runner.py"' in code
    assert 'f"minimax_h3_flashvsr_v1_1_net_x{flashvsr_net_scale_token}_cyclic_loop.mp4"' in code
    assert '"--input-resize-factor", str(FLASHVSR_INPUT_RESIZE_FACTOR)' in code
    assert '"input_resize_factor": FLASHVSR_INPUT_RESIZE_FACTOR' in code
    assert '"net_scale": flashvsr_net_scale' in code
    assert "FLASHVSR_FINAL_VIDEO_PATH\n    if FLASHVSR_FINAL_VIDEO_PATH is not None" in code
    assert '"flashvsr_frame_count_preserved"' in code
    assert 'globals().get("BORDER_STABILIZED_PATHS")' in code
    assert "discover_h3_finishing_source(" in code
    assert "h3_source_run_root = Path(RUN_DIRECTORY).parent" in code
    assert "preferred_run=RUN_DIRECTORY" in code
    assert '"flashvsr_recovery_selection": recovered_source["selection"]' in code
    assert 'RUN_DIRECTORY = recovered_source["run_directory"]' in code
    assert 'RUN_DIRECTORY / "metadata" / "border_stabilization.json"' in code
    assert 'RUN_DIRECTORY / "metadata" / "rife_report.json"' in code
    assert "Local finishing PNGs were already cleaned" in code
    assert "stdlib venv creation failed; falling back to virtualenv" in code
    assert '"virtualenv>=20.26,<21"' in code
    assert '"setup_version": 3' in code
    assert "Resuming the compatible partial FlashVSR Stable environment." in code
    assert '"-e", str(flashvsr_root)' not in code
    assert '"flash-attn", "sageattention", "torch", "torchaudio", "torchvision"' in code
    assert "Refusing to install filtered requirements containing flash-attn" in code
    assert "BLOCK_SPARSE_ATTN_CUDA_ARCHS" not in code
    assert "FLASHVSR_SPARSE_REPOSITORY_URL" not in code
    assert '"custom_cuda_extension_compiled": False' in code
    assert "lazy-loads temporal slices" in markdown
    assert "bundled Triton Sparse Sage backend" in markdown
