from __future__ import annotations

import ast
import hashlib
import json
import shutil
import types
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


def cell_source(cell_id: str) -> str:
    return "".join(
        next(cell for cell in load_notebook()["cells"] if cell.get("id") == cell_id)["source"]
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


def test_manifest_or_plain_directory_images_are_loaded_read_only() -> None:
    code = code_source()
    markdown = markdown_source()
    assert "SOURCE_RUN_DIRECTORY =" in code
    assert "SOURCE_PROJECT_NAME =" in code
    assert "source_project_directory = drive_base / SOURCE_PROJECT_NAME" in code
    assert "source_project_directory / configured_source" in code
    assert '"explicit_basename"' in code
    assert "def load_h3_anchor_records_portable(source_directory):" in code
    assert 'manifest_path = source_directory / "metadata" / "base_manifest.json"' in code
    assert "if manifest_path.is_file():" in code
    assert "return load_h3_anchor_records(source_directory)" in code
    assert 'source_directory.rglob("*")' in code
    assert "NOTEBOOK_ANCHOR_IMAGE_SUFFIXES" in code
    assert "BASE_RECORDS = load_h3_anchor_records_portable(SOURCE_RUN)" in code
    assert "H3_PAIRS = cyclic_h3_pairs(BASE_RECORDS)" in code
    assert 'record.get("authored_prompt", "").strip()' in code
    assert 'pair["left"]["authored_prompt"]' in code
    assert 'pair["right"]["authored_prompt"]' in code
    assert 'BASE_RECORDS[0].get("source_kind", "base_manifest")' in code
    assert "plain image directory" in markdown
    assert "natural filename order" in markdown
    assert '"source_run_modified": False' in code
    assert 'H3_PROJECT_NAME = "minimax_h3_interpolations"' in code


def test_notebook_plain_directory_scanner_bypasses_old_manifest_only_package(
    tmp_path: Path,
) -> None:
    tree = ast.parse(cell_source("h3-09-anchors"))
    portable_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "re" for alias in node.names)
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "NOTEBOOK_ANCHOR_IMAGE_SUFFIXES"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in {
                "notebook_natural_image_key",
                "load_h3_anchor_records_portable",
            }
        )
    ]
    namespace = {
        "Path": Path,
        "load_h3_anchor_records": lambda _source: (_ for _ in ()).throw(
            FileNotFoundError("old package still requires metadata/base_manifest.json")
        ),
    }
    exec(compile(ast.Module(body=portable_nodes, type_ignores=[]), "h3-portable", "exec"), namespace)

    source = tmp_path / "plain"
    source.mkdir()
    (source / "frame_10.PNG").write_bytes(b"image")
    (source / "frame_2.jpg").write_bytes(b"image")
    (source / "notes.txt").write_text("ignore", encoding="utf-8")
    hidden = source / ".thumbnails"
    hidden.mkdir()
    (hidden / "ignored.png").write_bytes(b"image")

    records = namespace["load_h3_anchor_records_portable"](source)

    assert [record["path"] for record in records] == ["frame_2.jpg", "frame_10.PNG"]
    assert all(record["authored_prompt"] == "" for record in records)
    assert all(record["source_kind"] == "directory_scan" for record in records)


def test_official_fl2va_openai_writer_uses_both_images_and_optional_prompts() -> None:
    code = code_source()
    assert 'H3_PROMPT_MODE = "openai_per_pair"' in code
    assert 'H3_LORA_TRIGGER' not in code
    assert 'H3_DURATION_SECONDS = 6.0' in code
    assert 'H3_JOB_TIMEOUT_SECONDS = 1800' in code
    assert 'H3_OPENAI_PROMPT_GUIDE_VERSION = "minimax-h3-fl2va-temporal-coherence-v10"' in code
    assert "H3_OPENAI_PROMPT_WRITER_INSTRUCTIONS = r\"\"\"" in code
    assert "<Picture 1> state -> observable intermediate changes -> progressively" in code
    assert "Match forms primarily by screen region, silhouette, scale, visual role" in code
    assert "Do not merely describe two static images" in code
    assert "Temporal smoothness is the highest-priority motion quality" in code
    assert "minute frame-to-frame increments" in code
    assert "each adjacent-frame step is nearly imperceptible in isolation" in code
    assert "Let its speed and direction vary" in code
    assert "progressively smaller adjustments" in code
    assert "Backgrounds and large color, light, shadow, and texture areas move as stable spatial fields" in code
    assert "none of the prohibited abrupt-motion words appears" in code
    assert 'print("OPENAI H3 PROMPT-WRITER INSTRUCTIONS' in code
    assert 'strip_h3_source_only_tokens(record.get("authored_prompt", ""))' in code
    assert "authored prompt is supplied for either endpoint" in code
    assert "def append_optional_prompt_context" in code
    assert "if clean_prompt:" in code
    assert 'if "RIJKSOIL" in payload["h3_prompt"]' in code
    assert 'OPENAI_MODEL = "gpt-5.6"' in code
    assert 'OPENAI_IMAGE_DETAIL = "original"' in code
    assert "OPENAI_CLIENT.responses.parse(" in code
    assert '"image_url": image_data_url(pair["left"]["resolved_path"])' in code
    assert '"image_url": image_data_url(pair["right"]["resolved_path"])' in code
    assert '"text": "<Picture 1> — exact first frame:"' in code
    assert '"text": "<Picture 2> — exact last frame:"' in code
    assert 'append_optional_prompt_context(user_content, "<Picture 1>", pair["left"])' in code
    assert 'append_optional_prompt_context(user_content, "<Picture 2>", pair["right"])' in code
    assert 'f"{picture_label} optional authored image prompt (semantic context only):\\n"' in code
    assert "class H3ObjectCorrespondence(BaseModel):" in code
    assert "object_correspondences: list[H3ObjectCorrespondence]" in code
    assert "min_length=4, max_length=10" in code
    assert 'OPENAI_REASONING_EFFORT = "high"' in code
    assert "OPENAI_MAX_OUTPUT_TOKENS = 32768" in code
    assert "OPENAI_H3_DESCRIPTION_MIN_CHARS = 1200" in code
    assert "OPENAI_H3_DESCRIPTION_MAX_CHARS = 2400" in code
    assert "integrated_multimodal_description: str" in code
    assert "min_length=OPENAI_H3_DESCRIPTION_MIN_CHARS" not in code
    assert "max_length=OPENAI_H3_DESCRIPTION_MAX_CHARS" not in code
    assert "if len(description) < OPENAI_H3_DESCRIPTION_MIN_CHARS:" in code
    assert "(?:[.!?]|\\u2026)[\\)\\]" in code
    assert "include the exact phrase Static Shot" in code
    assert "Start visible changes concurrently in at least three spatially separated regions" in code
    assert "no large region remains an intact endpoint while another large" in code
    assert "frame-spanning dividing line, moving sheet, moving band, single propagation front" in code
    assert "Treat the background and any broad color, shadow, light, or texture area as an active" in code
    assert "fields blend, flow, drift, spread, fold, or reshape across overlapping regions" in code
    assert "Broad background and color-field mappings may instead mix and move continuously" in code
    assert 'required = ("[Shot 1]", "<Picture 1>", "<Picture 2>", "Static Shot")' in code
    assert '"output_tokens_including_reasoning": getattr(usage, "output_tokens", None)' in code
    assert '"reasoning_tokens": getattr(output_details, "reasoning_tokens", None)' in code
    assert 'f"The previous draft was rejected for this exact reason: {last_error}. "' in code
    assert '"description_characters": len(description)' in code
    assert '"description_tail": description[-180:]' in code
    assert '"openai_reasoning_effort": OPENAI_REASONING_EFFORT' in code
    assert '"structured_output_schema": "compact-correspondences+distributed-blending-description-v8"' in code
    assert 'if response.status != "completed":' in code
    assert "OpenAI H3 description ended mid-sentence" in code
    assert "H3_DISALLOWED_GENERATED_TRANSITION_TERMS = (" in code
    assert '"jitter",' in code
    assert '"stutter",' in code
    assert "def validate_openai_motion_proposal(proposal):" in code
    assert "validate_openai_motion_proposal(proposal)" in code
    assert '"prompt_writer_instructions": H3_OPENAI_PROMPT_WRITER_INSTRUCTIONS' in code
    assert '"disallowed_generated_terms": H3_DISALLOWED_GENERATED_TRANSITION_TERMS' in code
    assert 'print("POSITIVE OBJECT CORRESPONDENCE MAP:")' in code
    assert "H3_TEXT_ENCODER_CONTEXT_TOKENS = 262144" in code
    assert "H3_WORKFLOW_PATCH_VERSION = 10" in code
    assert "H3_IMAGE_CONDITIONING_TOKEN_RESERVE = 8192" in code
    assert "H3_PROMPT_MAX_UTF8_BYTES = 8192" in code
    assert "from comfy.text_encoders.minimax import MiniMaxH3Tokenizer" not in code
    assert "validate_h3_prompt_byte_budget(" in code
    assert 'plan["h3_prompt_utf8_bytes"] = prompt_byte_count' in code
    assert "H3 PROMPT UTF-8 BYTES:" in code
    assert 'prompt_plan.get("h3_prompt_utf8_bytes") != prompt_byte_count' in code
    assert '"h3_prompt_max_utf8_bytes": H3_PROMPT_MAX_UTF8_BYTES' in code
    assert "h3_prompt_text_tokens" not in code
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
    assert "def ensure_h3_server_running():" in code
    assert "H3_SERVER_STATE = ensure_h3_server_running()" in code
    assert "def render_h3_attempt(" in code
    assert "ensure_h3_server_running()" in code
    assert 'server_source = "restarted_automatically"' in code
    assert "H3_REUSE_EXISTING_CLIPS = True" in code
    assert 'prior.get("fingerprint") == fingerprint' in code
    assert "def resolve_h3_base_seed_for_render():" in code
    assert "base_seed = resolve_h3_base_seed_for_render()" in code
    assert '"seed": (base_seed + pair["index"] + retry_offset)' in code
    assert "ThreadPoolExecutor(" in code
    assert "as_completed(future_to_pair)" in code
    assert "executor.submit(render_h3_pair, pair)" in code
    assert "H3_CLIP_RECORDS.update(render_h3_pairs_parallel(H3_PAIRS))" in code


def test_live_h3_server_preflight_reuses_a_ready_server() -> None:
    tree = ast.parse(cell_source("h3-15-server"))
    ensure = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "ensure_h3_server_running"
    )

    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    namespace = {
        "H3_SERVER_START_LOCK": Lock(),
        "H3_COMFY_PROCESS": None,
        "comfy_server_ready": lambda: True,
        "H3_SERVER_URL": "http://127.0.0.1:8188",
        "COMFY_LOG_PATH": Path("/tmp/not-read.log"),
        "gpu_vram_gib": 40,
    }
    exec(
        compile(ast.Module(body=[ensure], type_ignores=[]), "server-preflight", "exec"),
        namespace,
    )
    state = namespace["ensure_h3_server_running"]()
    assert state["source"] == "already_running"
    assert state["server"] == "http://127.0.0.1:8188"


def test_render_recovers_seed_after_out_of_order_settings_rerun(tmp_path: Path) -> None:
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    persisted_seed = 1_234_567_890
    (metadata_directory / "run_seed.json").write_text(
        json.dumps({"h3_base_seed": persisted_seed}), encoding="utf-8"
    )
    tree = ast.parse(cell_source("h3-17-render"))
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_h3_base_seed_for_render"
    )
    namespace = {
        "H3_BASE_SEED": None,
        "RUN_DIRECTORY": tmp_path,
        "Path": Path,
        "json": json,
    }
    exec(
        compile(ast.Module(body=[resolver], type_ignores=[]), "seed-recovery", "exec"),
        namespace,
    )
    assert namespace["resolve_h3_base_seed_for_render"]() == persisted_seed
    assert namespace["H3_BASE_SEED"] == persisted_seed


def test_post_render_quality_gate_has_slider_hard_veto_and_remains_resumable() -> None:
    code = code_source()
    markdown = markdown_source()
    assert "RUN_OPENAI_H3_QUALITY_GATE = True" in code
    assert "H3_QUALITY_SAMPLE_FRAME_COUNT = 7" in code
    assert "H3_QUALITY_RETRY_MIN_CONFIDENCE = 0.78" in code
    assert "H3_SLIDER_RETRY_MIN_CONFIDENCE = 0.50" in code
    assert 'H3_QUALITY_GATE_VERSION = "h3-interior-slider-hard-veto-v5-seven-frames"' in code
    assert "Inspect every interior frame separately" in code
    assert "A slider, wipe, or split is a mandatory RETRY" in code
    assert "affected interior frame is sufficient" in code
    assert "For non-slider defects, remain deliberately tolerant" in code
    assert "If non-slider evidence is" in code
    assert 'OPENAI_QUALITY_REASONING_EFFORT = "high"' in code
    assert "OPENAI_QUALITY_MAX_OUTPUT_TOKENS = 8192" in code
    assert 'OPENAI_QUALITY_IMAGE_DETAIL = "original"' in code
    assert "dominant_blank_or_flat_field" in code
    assert "disconnected_collage_or_cutouts" in code
    assert "frame_wide_wipe_or_split" in code
    assert "class H3FrameQualityJudgment(BaseModel):" in code
    assert "class H3QualityJudgment(BaseModel):" in code
    assert "frame_checks: list[H3FrameQualityJudgment]" in code
    assert 'verdict: Literal["pass", "retry"]' in code
    assert "def h3_quality_sample_fractions():" in code
    assert "def h3_slider_gate_decision(judgment, expected_frame_count):" in code
    assert "(index + 1) / (count + 1)" in code
    assert '"text": "<Picture 1> — exact first endpoint:"' in code
    assert '"text": "<Picture 2> — exact last endpoint:"' in code
    assert 'f"Interior frame {frame_number}/{len(frame_paths)} "' in code
    assert "text_format=H3QualityJudgment" in code
    assert "judgment.confidence >= H3_QUALITY_RETRY_MIN_CONFIDENCE" in code
    assert 'slider_decision["slider_hard_veto"]' in code
    assert 'slider_decision["gate_integrity_retry"]' in code
    assert '"effective_reason": effective_reason' in code
    assert '"low_confidence_retry_overridden"' in code
    assert "archive_rejected_h3_clip(" in code
    assert 'H3_REJECTED_VIDEO_SUBDIRECTORY = "rejected_videos"' in code
    assert "def h3_rejected_video_directory():" in code
    assert '"H3_REJECTED_VIDEO_SUBDIRECTORY", "rejected_videos"' in code
    assert "rejected_directory = h3_rejected_video_directory()" in code
    assert 'f"_attempt_{render_attempt:02d}_rejected.mp4"' in code
    assert "Saved rejected H3 video" in code
    assert "H3_QUALITY_NEGATIVE_EXAMPLES = (" in code
    assert "retry_example_transition_front.jpg" in code
    assert "retry_example_detached_collage.jpg" in code
    assert "def resolve_h3_quality_negative_examples():" in code
    assert '"download_url": (' in code
    assert '"github_download_cache"' in code
    assert "urllib.request.urlretrieve(" in code
    assert "failed its SHA-256 check" in code
    assert '"image_url": example["image_url"]' in code
    assert '"negative_examples": [{' in code
    assert "Known RETRY example" in code
    assert "Learn only each labeled failure pattern" in code
    assert "quality_rejected_on_resume" in code
    assert '"awaiting_quality_gate": True' in code
    assert "Retrying with a new seed" in code
    assert "H3_MAX_PARALLEL_TRANSITION_CALLS = 3" in code
    assert "def render_h3_pairs_parallel(pairs):" in code
    assert "ThreadPoolExecutor(" in code
    assert "as_completed(future_to_pair)" in code
    assert "H3_RETRY_SEED_STRIDE = 1_000_003" in code
    assert "bounded parallel pool" in markdown
    assert "seven ordered interior frames" in markdown
    assert "mandatory" in markdown
    assert "global verdict says pass" in markdown
    assert "Rejected MP4s are retained in `rejected_videos`" in markdown


def test_rejected_h3_clips_are_archived_in_dedicated_folder(tmp_path: Path) -> None:
    tree = ast.parse(cell_source("h3-17-render"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "h3_rejected_video_directory",
            "archive_rejected_h3_clip",
        }
    ]
    namespace = {
        "RUN_DIRECTORY": tmp_path,
        # Deliberately omit the new setting to simulate a refreshed render cell
        # running in an older live kernel.
        "shutil": shutil,
        "Path": Path,
        "safe_name": lambda value: str(value).replace("/", "_"),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "rejected-archive", "exec"),
        namespace,
    )
    source_clip = tmp_path / "candidate.mp4"
    source_clip.write_bytes(b"candidate-video")
    rejected_path = namespace["archive_rejected_h3_clip"](
        {"index": 2, "pair_id": "left/right"}, source_clip, 3
    )
    assert rejected_path == (
        tmp_path / "rejected_videos" / "pair_0002_left_right_attempt_03_rejected.mp4"
    )
    assert rejected_path.read_bytes() == b"candidate-video"


def test_quality_gate_negative_example_assets_are_packaged() -> None:
    example_directory = ROOT / "notebooks" / "assets" / "h3_quality_examples"
    for filename in (
        "retry_example_transition_front.jpg",
        "retry_example_detached_collage.jpg",
    ):
        payload = (example_directory / filename).read_bytes()
        assert len(payload) > 10_000
        assert payload.startswith(b"\xff\xd8")


def test_quality_gate_examples_download_from_github_and_cache(tmp_path: Path) -> None:
    settings_tree = ast.parse(cell_source("h3-02-settings"))
    quality_assignments = [
        node
        for node in settings_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {
                "H3_QUALITY_EXAMPLE_RAW_BASE_URL",
                "H3_QUALITY_NEGATIVE_EXAMPLES",
            }
            for target in node.targets
        )
    ]
    settings_namespace = {
        "REPOSITORY_REF": "agent/chimera-flux-flat-morph",
    }
    exec(
        compile(
            ast.Module(body=quality_assignments, type_ignores=[]),
            "quality-example-settings",
            "exec",
        ),
        settings_namespace,
    )
    examples = settings_namespace["H3_QUALITY_NEGATIVE_EXAMPLES"]
    assert len(examples) == 2
    for example in examples:
        asset_path = ROOT / example["relative_path"]
        assert example["download_url"].startswith(
            "https://raw.githubusercontent.com/MNoichl/FluxFlowMorph/"
        )
        assert hashlib.sha256(asset_path.read_bytes()).hexdigest() == example["sha256"]

    render_tree = ast.parse(cell_source("h3-17-render"))
    resolver = next(
        node
        for node in render_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_h3_quality_negative_examples"
    )
    downloads = []

    def fake_urlretrieve(url, destination):
        configured = next(item for item in examples if item["download_url"] == url)
        Path(destination).write_bytes((ROOT / configured["relative_path"]).read_bytes())
        downloads.append(url)

    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    namespace = {
        "Path": Path,
        "PROJECT_ROOT": str(tmp_path / "empty_checkout"),
        "LOCAL_ASSET_ROOT": str(tmp_path / "cache"),
        "H3_QUALITY_NEGATIVE_EXAMPLES": examples,
        "urllib": types.SimpleNamespace(
            request=types.SimpleNamespace(urlretrieve=fake_urlretrieve)
        ),
        "sha256_file": sha256_file,
        "image_data_url": lambda path: f"data:image/jpeg;base64,{Path(path).name}",
    }
    exec(
        compile(ast.Module(body=[resolver], type_ignores=[]), "quality-examples", "exec"),
        namespace,
    )
    resolved = namespace["resolve_h3_quality_negative_examples"]()
    assert len(downloads) == 2
    assert all(item["source"] == "github_download" for item in resolved)
    assert all(item["image_url"].startswith("data:image/jpeg;base64,") for item in resolved)
    cached = namespace["resolve_h3_quality_negative_examples"]()
    assert len(downloads) == 2
    assert all(item["source"] == "github_download_cache" for item in cached)


def test_openai_calls_use_bounded_backoff_and_prompt_micro_batches() -> None:
    code = code_source()
    assert "OPENAI_REQUEST_MAX_ATTEMPTS = 5" in code
    assert "OPENAI_RETRY_INITIAL_SECONDS = 2.0" in code
    assert "OPENAI_RETRY_MAX_SECONDS = 60.0" in code
    assert "OPENAI_PROMPT_BATCH_SIZE = 4" in code
    assert "def bounded_backoff_seconds(" in code
    assert "def openai_call_with_backoff(label, operation):" in code
    assert 'headers.get("retry-after", 0.0)' in code
    assert '"insufficient_quota"' in code
    assert "except OPENAI_TRANSIENT_ERRORS as error:" in code
    assert "OpenAI(api_key=api_key, max_retries=0)" in code
    assert "OpenAI prompt pair" in code
    assert "OpenAI quality gate pair" in code
    assert "openai_prompts_since_pause >= OPENAI_PROMPT_BATCH_SIZE" in code
    assert "OPENAI_PROMPT_BATCH_PAUSE_SECONDS" in code


def test_quality_sample_fractions_cover_three_to_nine_interior_frames() -> None:
    tree = ast.parse(cell_source("h3-17-render"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "h3_quality_sample_fractions"
    )
    namespace: dict = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "quality-fractions", "exec"),
        namespace,
    )
    expected = {
        3: (0.25, 0.5, 0.75),
        5: tuple(index / 6 for index in range(1, 6)),
        7: tuple(index / 8 for index in range(1, 8)),
        9: tuple(index / 10 for index in range(1, 10)),
    }
    for count, fractions in expected.items():
        namespace["H3_QUALITY_SAMPLE_FRAME_COUNT"] = count
        assert namespace["h3_quality_sample_fractions"]() == fractions


def test_slider_frame_flag_overrides_global_pass() -> None:
    tree = ast.parse(cell_source("h3-17-render"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "h3_slider_gate_decision"
    )
    namespace = {"H3_SLIDER_RETRY_MIN_CONFIDENCE": 0.50}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "slider-veto", "exec"),
        namespace,
    )

    def check(number, flagged=False, confidence=0.99):
        return types.SimpleNamespace(
            frame_number=number,
            frame_wide_wipe_or_split=flagged,
            confidence=confidence,
        )

    judgment = types.SimpleNamespace(
        frame_checks=[check(number, flagged=number == 4) for number in range(1, 8)],
        verdict="pass",
        catastrophic_failure_types=["none"],
        confidence=0.98,
    )
    decision = namespace["h3_slider_gate_decision"](judgment, 7)
    assert decision == {
        "frame_check_complete": True,
        "slider_frame_numbers": [4],
        "slider_hard_veto": True,
        "gate_integrity_retry": False,
    }

    clean = types.SimpleNamespace(
        frame_checks=[check(number) for number in range(1, 8)],
        verdict="pass",
        catastrophic_failure_types=["none"],
        confidence=0.98,
    )
    assert namespace["h3_slider_gate_decision"](clean, 7)["slider_hard_veto"] is False

    incomplete = types.SimpleNamespace(
        frame_checks=[check(number) for number in range(1, 7)],
        verdict="pass",
        catastrophic_failure_types=["none"],
        confidence=0.98,
    )
    assert namespace["h3_slider_gate_decision"](incomplete, 7)["gate_integrity_retry"] is True


def test_final_audit_records_quality_gate_and_retry_outcomes() -> None:
    code = code_source()
    assert '"openai_used_for_prompt_planning"' in code
    assert '"openai_used_for_post_render_quality_gate"' in code
    assert '"quality_gate_all_pairs_passed"' in code
    assert '"quality_gate_rejected_attempt_count"' in code
    assert '"h3_max_parallel_transition_calls": H3_MAX_PARALLEL_TRANSITION_CALLS' in code
    assert '"h3_rejected_video_directory"' in code
    assert '"quality_gate_slider_retry_min_confidence"' in code
    assert '"quality_gate_image_detail"' in code
    assert '"quality_gate_reasoning_effort"' in code
    assert '"quality_gate_negative_examples"' in code


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
