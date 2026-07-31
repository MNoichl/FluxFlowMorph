from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT
    / "notebooks"
    / "StillLife_FlowMorph_LTX13B_Conditioned_Video.ipynb"
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
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 34
    assert "StillLife_FlowMorph_LTX13B_Conditioned_Video.ipynb" in "".join(
        notebook["cells"][0]["source"]
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")


def test_flowmorph_is_one_circular_round_with_twelve_interiors() -> None:
    code = code_source(load_notebook())
    assert (
        'FLOWMORPH_ROUND_SPECS = [\n'
        '    {"midpoint_count": 12, "prompt_mode": "shared_midpoint"},\n'
        "]"
    ) in code
    assert 'if len(FLOWMORPH_ROUND_SPECS) != 1:' in code
    assert "midpoint_conditionings=[shared] * midpoint_count" in code
    assert "right = incoming[(gap_index + 1) % gap_count]" in code
    assert "SEQUENCE_SESSION.render_midpoints(" in code
    assert "SEQUENCE_SESSION.render_endpoint_reconstructions(" in code
    assert 'if record["uid"] not in ENDPOINT_CACHE' in code
    assert '"flowmorph_interiors_per_gap": 12' in code
    assert '"conditions_per_gap": 14' in code
    assert "RUN_RIFE_POSTPROCESS" not in code


def test_ltx_uses_every_flowmorph_series_at_valid_temporal_indices() -> None:
    code = code_source(load_notebook())
    assert (
        'LTX_MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"'
        in code
    )
    assert (
        'LTX_MODEL_REVISION = '
        '"7c64400e1861cc0d7b98d570a1926d5408ec60cd"'
        in code
    )
    assert (
        'LTX_UPSAMPLER_REVISION = '
        '"e0c981533db26531c47dec16a124586cea53f11f"'
        in code
    )
    assert "LTX_FRAMES_PER_CONDITION_INTERVAL = 8" in code
    assert 'LTX_CONDITIONING_MODE = "dense_video"' in code
    assert 'LTX_DENSE_GUIDE_INTERPOLATION = "linear"' in code
    assert "LTX_CONDITIONING_STRENGTH = 0.85" in code
    assert "LTX_NUM_FRAMES = LTX_FRAME_INDICES[-1] + 1" in code
    assert "if LTX_NUM_FRAMES % 8 != 1:" in code
    assert "for index in range(14)" in code
    assert "LTXVideoCondition(" in code
    assert "def build_dense_ltx_guide(condition_images):" in code
    assert "video=dense_guide_frames" in code
    assert "frame_index=0" in code
    assert "frame_index=frame_index" in code
    assert "step / LTX_FRAMES_PER_CONDITION_INTERVAL" in code
    assert '"conditioning_mode": LTX_CONDITIONING_MODE' in code
    assert 'condition_paths = [' in code
    assert 'str(ENDPOINT_RECONSTRUCTION_PATHS[job["left"]["uid"]])' in code
    assert 'str(ENDPOINT_RECONSTRUCTION_PATHS[job["right"]["uid"]])' in code


def test_distilled_two_stage_quality_contract_is_exposed() -> None:
    code = code_source(load_notebook())
    assert "LTX_GUIDANCE_SCALE = 1.0" in code
    assert (
        "LTX_FIRST_PASS_TIMESTEPS = "
        "[1000, 993, 987, 981, 975, 909, 725, 0.03]"
    ) in code
    assert "LTX_SECOND_PASS_TIMESTEPS = [1000, 909, 725, 421, 0]" in code
    assert "LTX_DECODE_TIMESTEP = 0.05" in code
    assert "LTX_DECODE_NOISE_SCALE = 0.025" in code
    assert "LTX_TONE_MAP_COMPRESSION_RATIO = 0.6" in code
    assert "LTXLatentUpsamplerModel.from_pretrained(" in code
    assert "LTXLatentUpsamplePipeline(" in code
    assert "tone_map_compression_ratio=(" in code
    assert "denoise_strength=LTX_UPSCALE_DENOISE_STRENGTH" in code


def test_flux_is_released_before_one_offloaded_ltx_model_load() -> None:
    code = code_source(load_notebook())
    release_position = code.index(
        'for variable_name in (\n'
        '    "ENDPOINT_CACHE",'
    )
    ltx_load_position = code.index(
        "ltx_transformer = AutoModel.from_pretrained("
    )
    assert release_position < ltx_load_position
    assert "component.to(\"cpu\")" in code
    assert "torch.cuda.synchronize()" in code
    assert "torch.cuda.empty_cache()" in code
    assert "torch.cuda.ipc_collect()" in code
    assert "LTX_ENABLE_FP8_LAYERWISE_STORAGE = True" in code
    assert "enable_layerwise_casting(" in code
    assert "LTX_ENABLE_GROUP_OFFLOAD = True" in code
    assert "enable_group_offload(" in code
    assert "apply_group_offloading(" in code
    assert '"model_loads": 1' in code


def test_ltx_download_has_disk_cleanup_preflight_and_rerun_safety() -> None:
    code = code_source(load_notebook())
    assert "DELETE_LOCAL_FLUX_CACHE_BEFORE_LTX = True" in code
    assert "DELETE_LOCAL_FLOWMORPH_WORK_BEFORE_LTX = True" in code
    assert "CLEAN_PIP_CACHE_BEFORE_LTX = True" in code
    assert "CLEAN_INTERRUPTED_LTX_DOWNLOADS_IF_NEEDED = True" in code
    assert "LTX_DISABLE_XET_FOR_DISK_SAFETY = True" in code
    assert '("models--" + MODEL_ID.replace("/", "--"))' in code
    assert "shutil.rmtree(flux_cache_directory)" in code
    assert "shutil.rmtree(local_flowmorph_work)" in code
    assert '[sys.executable, "-m", "pip", "cache", "purge"]' in code
    assert "LTX_CACHE_DIR = HF_CACHE_DIR" in code
    assert "from huggingface_hub import HfApi" in code
    assert "def repo_download_state(" in code
    assert "hf_api.list_repo_tree(" in code
    assert 'and "/" not in item_path[' in code
    assert 'blob_root = repo_cache / "blobs"' in code
    assert "blob_path.stat().st_size == item_size" in code
    assert '"ltx_model_cached_gib"' in code
    assert '"ltx_download_remaining_gib"' in code
    assert '"required_including_headroom_gib"' in code
    assert 'repo_cache.rglob("*.incomplete")' in code
    assert "from huggingface_hub.constants import HF_XET_CACHE" in code
    assert 'os.environ["HF_HUB_DISABLE_XET"] = "1"' in code
    assert "hf_hub_constants.HF_HUB_DISABLE_XET = True" in code
    assert "shutil.rmtree(xet_cache_path)" in code
    assert '"incomplete_ltx_gib_reclaimed"' in code
    assert '"xet_temporary_gib_reclaimed"' in code
    assert "Not enough local disk for the remaining LTX download." in code
    assert '"ltx_transformer",' in code
    assert 'globals().pop(stale_name, None)' in code
    assert 'globals().get("SEQUENCE_RUNNER")' in code


def test_disk_investigation_is_read_only_drive_safe_and_persisted() -> None:
    notebook = load_notebook()
    code = code_source(notebook)
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "## 11a. Read-only local-disk investigation" in markdown
    assert "def local_tree_size_bytes(path):" in code
    assert 'if child.name == "drive":' in code
    assert '"google_drive_excluded": True' in code
    assert '"huggingface_repository_caches"' in code
    assert '"incomplete_downloads"' in code
    assert '"largest_local_cache_files"' in code
    assert 'rglob("*.incomplete")' in code
    assert '"disk_investigation.json"' in code
    diagnostic_start = code.index("def local_tree_size_bytes(path):")
    diagnostic_end = code.index(
        "from pathlib import Path\n\nLTX_ROOT",
        diagnostic_start,
    )
    diagnostic_code = code[diagnostic_start:diagnostic_end]
    assert "unlink(" not in diagnostic_code
    assert "rmtree(" not in diagnostic_code
    assert "remove(" not in diagnostic_code


def test_clips_are_resumable_and_assembled_without_duplicate_endpoints() -> None:
    code = code_source(load_notebook())
    assert "REUSE_EXISTING_LTX_CLIPS = True" in code
    assert 'saved.get("fingerprint") == fingerprint' in code
    assert 'for frame in frames[:-1]:' in code
    assert '"omitted_terminal_condition_frame": True' in code
    assert '"frames_saved_per_gap": LTX_NUM_FRAMES - 1' in code
    assert '"cyclic": True' in code
    assert '"-f",\n    "concat"' in code
    assert '"-c",\n    "copy"' in code
    assert "flowmorph_conditioned_ltx13b_cyclic.mp4" in code
    assert 'Video(\n            str(job["clip_path"]),' in code
    assert 'Video(\n    str(LTX_FINAL_VIDEO_PATH),' in code
    assert 'filename=str(job["clip_path"])' not in code
    assert "filename=str(LTX_FINAL_VIDEO_PATH)" not in code


def test_anchor_generation_contract_remains_prompt_only_and_no_beige_canvas() -> None:
    code = code_source(load_notebook())
    assert "BASE_STAGES = [" in code
    assert "BASE_PROMPT_COUNT = None" in code
    assert "make_soft_reference(" in code
    assert "prepare_flux2_klein_img2img_inputs(" in code
    assert "reference_blend=1.0" in code
    assert "REFERENCE_BACKGROUND" not in code
    assert "background_rgb=" not in code
    assert "MASK_" not in code
    assert "TRAJECTORY_" not in code
