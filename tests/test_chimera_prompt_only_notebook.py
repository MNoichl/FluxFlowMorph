from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "StillLife_Recursive_CHIMERA_Prompt_Only.ipynb"
BUILDER_PATH = ROOT / "scripts" / "build_recursive_chimera_prompt_only_notebook.py"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")


def all_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_chimera_notebook_is_parseable_and_has_colab_badge() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 33
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    assert "StillLife_Recursive_CHIMERA_Prompt_Only.ipynb" in "".join(notebook["cells"][0]["source"])
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_chimera_notebook_remains_prompt_only_and_drive_native() -> None:
    code = code_source(load_notebook())
    assert "BASE_STAGES = [" in code
    assert "BASE_PROMPT_COUNT = None" in code
    assert "drive.mount(" in code
    assert "DRIVE_PROJECT_BASE" in code
    assert "RESUME_RUN_DIRECTORY" in code
    assert "MASK_" not in code
    assert "TRAJECTORY_" not in code
    assert "prepare_grayscale_edit_mask" not in code
    assert "upload" not in code.lower()


def test_colab_setup_imports_chimera_from_the_updated_repository() -> None:
    notebook = load_notebook()
    settings = "".join(notebook["cells"][2]["source"])
    setup = "".join(notebook["cells"][6]["source"])
    code = code_source(notebook)
    assert "CHIMERA_BOOTSTRAP_B85" not in code
    assert "b85decode" not in code
    assert "spec_from_file_location" not in code
    assert 'REPOSITORY_REF = "agent/chimera-flux-flat-morph"' in settings
    assert '"origin", REPOSITORY_REF' in setup
    assert '"checkout", "--detach", "FETCH_HEAD"' in setup
    assert '"pull", "--ff-only"' not in setup
    assert "UPDATE_REPOSITORY" not in settings
    assert "UPDATE_REPOSITORY" not in setup
    assert 'importlib.import_module("flowmorph_klein.chimera")' in setup
    assert "import flowmorph_klein" in setup
    assert "from flowmorph_klein.chimera import (" in code


def test_flat_round_paper_defaults_and_flux_memory_adaptations_are_explicit() -> None:
    text = all_source(load_notebook())
    assert "CHIMERA_ROUND_SPECS = [" in text
    assert '"midpoint_count":' in text
    assert "CHIMERA_INVERSION_STEPS = 50" in text
    assert "CHIMERA_DENOISING_STEPS = 50" in text
    assert "CHIMERA_ACI_WEIGHT = 0.4" in text
    assert "CHIMERA_SAP_ACTIVE_RATIO = 0.2" in text
    assert "CHIMERA_ANCHOR_RELIABILITY_THRESHOLD = 0.45" in text
    assert "CHIMERA_ENFORCE_SAP_RELIABILITY = False" in text
    assert 'CHIMERA_LTM_MODE = "fft"' in text
    assert "CHIMERA_LTM_BANDS = 16" in text
    assert "CHIMERA_LTM_CALIBRATION_ANCHORS = 8" in text
    assert "REUSE_CHIMERA_LTM_CALIBRATION = True" in text
    assert "CHIMERA_AUTO_RENDER_BATCH_SIZE = True" in text
    assert "CHIMERA_RENDER_BATCH_MAX = 10" in text
    assert "CHIMERA_BATCH_MEMORY_RESERVE_FRACTION = 0.10" in text
    assert "CHIMERA_BATCH_MEMORY_RESERVE_GIB = 2.0" in text
    assert "CHIMERA_DECODE_BATCH_SIZE = 10" in text
    assert "CHIMERA_CFG_START_RATIO = 0.0" in text
    assert "CHIMERA_CFG_STOP_RATIO = 1.0" in text
    assert 'CHIMERA_CONDITIONING_INTERPOLATION = "slerp"' in text
    assert "CHIMERA_ALPHA_WARP_STRENGTH = 0.0" in text
    assert "center_weighted_alpha_schedule(" in text
    assert '"alpha_warp_strength": CHIMERA_ALPHA_WARP_STRENGTH' in text
    assert "conditioning_interpolation=CHIMERA_CONDITIONING_INTERPOLATION" in text
    assert 'CHIMERA_CACHE_STORAGE = "float16"' in text
    assert "CHIMERA_CACHE_STRIDE = 1" in text
    assert "CHIMERA_VELOCITY_SMOOTHING_STRENGTH = 0.10" in text
    assert "velocity_smoothing_strength=CHIMERA_VELOCITY_SMOOTHING_STRENGTH" in text
    assert "measures their timestep correspondence" in text


def test_base_pipeline_defaults_to_gpu_resident_and_releases_before_chimera() -> None:
    notebook = load_notebook()
    code = code_source(notebook)
    model_source = "".join(
        next(
            cell for cell in notebook["cells"]
            if cell.get("id") == "prompt-only-chimera-12"
        )["source"]
    )
    chimera_source = "".join(
        next(
            cell for cell in notebook["cells"]
            if cell.get("id") == "prompt-only-chimera-19"
        )["source"]
    )

    assert "BASE_PIPELINE_CPU_OFFLOAD = False" in code
    assert "BASE_PIPELINE_RESIDENT_RESERVE_GIB = 12.0" in code
    assert "CHIMERA_HANDOFF_MAX_CUDA_GIB = 1.0" in code
    assert "effective_cpu_offload = bool(BASE_PIPELINE_CPU_OFFLOAD)" in model_source
    assert "if effective_cpu_offload:" in model_source
    assert "pipeline.enable_model_cpu_offload()" in model_source
    assert 'pipeline.to("cuda")' in model_source
    assert 'globals().pop("FLUX_PIPE_CPU_OFFLOAD", None)' in model_source
    assert 'globals().pop("FLUX_PIPE_CPU_OFFLOAD_REQUESTED", None)' in model_source
    assert 'globals().get("FLUX_PIPE_CPU_OFFLOAD_REQUESTED")' in model_source
    assert "Base pipeline LoRA/residency setting changed" in model_source
    assert "def pipeline_storage_bytes(pipeline):" in model_source
    assert "def resident_memory_preflight(pipeline):" in model_source
    assert "torch.cuda.mem_get_info()" in model_source
    assert '"resident_fit": model_bytes + reserve_bytes <= free_bytes' in model_source
    assert "automatic CPU-offload fallback" in model_source
    assert "except (torch.cuda.OutOfMemoryError, RuntimeError) as error:" in model_source
    assert 'pipeline.to("cpu")' in model_source
    assert 'move_to_cpu("cpu")' in model_source
    assert "automatic low-free-VRAM fallback" in model_source
    runner_load = chimera_source.index("CHIMERA_RUNNER = FlowMorphRunner.from_config")
    final_release = chimera_source.rfind("release_flux_pipeline()", 0, runner_load)
    assert final_release >= 0
    assert 'if "FLUX_PIPE" in globals()' in chimera_source[final_release:runner_load]
    assert "allocated_gib > CHIMERA_HANDOFF_MAX_CUDA_GIB" in chimera_source
    assert "cuda_process_report()" in chimera_source
    assert "CHIMERA preparation ran out of VRAM" in chimera_source
    assert "torch.cuda.empty_cache()" in model_source


def test_cost_preview_resolves_automatic_prompt_count_without_policy_gates() -> None:
    notebook = load_notebook()
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "prompt-only-chimera-10")
    source = "".join(cell["source"])
    namespace = {
        "BASE_PROMPT_COUNT": None,
        "BASE_STAGES": [{"id": f"anchor_{index}"} for index in range(3)],
        "CHIMERA_ROUND_SPECS": [{"midpoint_count": 16}],
        "IMAGE_WIDTH": 1024,
        "IMAGE_HEIGHT": 1024,
        "CHIMERA_RENDER_BATCH_SIZE": 2,
        "CHIMERA_RENDER_BATCH_MAX": 10,
        "CHIMERA_LTM_CALIBRATION_ANCHORS": 8,
        "CHIMERA_INVERSION_STEPS": 50,
        "CHIMERA_DENOISING_STEPS": 50,
        "CHIMERA_CFG_START_RATIO": 0.0,
        "CHIMERA_CFG_STOP_RATIO": 1.0,
        "CHIMERA_LTM_MODE": "fft",
        "CHIMERA_LTM_BANDS": 16,
        "CHIMERA_BATCH_MEMORY_RESERVE_FRACTION": 0.1,
        "CHIMERA_BATCH_MEMORY_RESERVE_GIB": 2.0,
        "CHIMERA_CACHE_STORAGE": "float16",
        "CHIMERA_CACHE_STRIDE": 1,
        "CHIMERA_VELOCITY_SMOOTHING_STRENGTH": 0.10,
        "CHIMERA_CONDITIONING_INTERPOLATION": "slerp",
        "RIFE_MULTIPLIER": 2,
    }

    exec(source, namespace)

    assert namespace["BASE_PROMPT_COUNT"] == 3
    assert namespace["round_counts"] == [3, 51]
    assert namespace["cfg_active_steps"] == 50
    assert namespace["conditional_only_steps"] == 0
    assert "production contract" not in source
    assert "CHIMERA_DENOISING_STEPS !=" not in source
    assert "CHIMERA_LORA_SCALE !=" not in source


def test_fft_ltm_is_calibrated_persisted_and_part_of_pair_identity() -> None:
    code = code_source(load_notebook())
    assert "LTM_CALIBRATION_VERSION," in code
    assert "LTM_TIMESTEP_SMOOTHING_RADIUS," in code
    assert "LTMCalibration," in code
    assert 'RUN_DIRECTORY / "metadata" / "chimera_ltm_calibration.json"' in code
    assert "CHIMERA_SESSION.calibrate_ltm(" in code
    assert "CHIMERA_SESSION.set_ltm_calibration(candidate)" in code
    assert '"descriptor_normalized": True' in code
    assert '"timestep_descriptor": "conditional_velocity"' in code
    assert '"timestep_smoothing_radius": LTM_TIMESTEP_SMOOTHING_RADIUS' in code
    assert "calibration_count = min(CHIMERA_LTM_CALIBRATION_ANCHORS, len(BASE_RECORDS))" in code
    assert "len(set(calibration_indices)) != calibration_count" in code
    assert '"calibration_version": LTM_CALIBRATION_VERSION' in code
    assert '"mapping_strategy": CHIMERA_LTM_CALIBRATION.mapping_strategy' in code
    assert '"mapping_report": CHIMERA_LTM_CALIBRATION.mapping_report' in code
    assert '"independent_mapping_report"' in code
    assert '"ltm_report": CHIMERA_LTM_REPORT' in code
    assert '"calibration_fingerprint": CHIMERA_LTM_CALIBRATION.fingerprint' in code
    assert '"ltm_fingerprint": CHIMERA_LTM_FINGERPRINT' in code


def test_authored_prompts_are_valid_and_original_anchor_init_is_restored() -> None:
    notebook = load_notebook()
    stages_source = "".join(notebook["cells"][4]["source"])
    stages_tree = ast.parse(stages_source)
    assignment = next(
        node
        for node in stages_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BASE_STAGES" for target in node.targets)
    )
    stages = ast.literal_eval(assignment.value)
    code = code_source(notebook)
    assert len(stages) >= 3
    assert all(set(stage) == {"id", "science", "prompt"} for stage in stages)
    assert len({stage["id"] for stage in stages}) == len(stages)
    assert all(stage["science"].strip() and stage["prompt"].strip() for stage in stages)
    assert "BASE_REFERENCE_STRENGTH =" in code
    assert "REFERENCE_BACKGROUND = (116, 105, 91)" in code
    assert "reference_blend=BASE_REFERENCE_STRENGTH" in code
    assert "background_rgb=REFERENCE_BACKGROUND" in code
    assert 'kwargs["image"] = reference' in code
    assert "prepare_flux2_klein_img2img_inputs" not in code
    assert "BASE_REFERENCE_DENOISE_STRENGTH" not in code


def test_builder_validates_without_modifying_the_authored_notebook() -> None:
    before = NOTEBOOK_PATH.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(BUILDER_PATH)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "without modifying it" in completed.stdout
    assert NOTEBOOK_PATH.read_bytes() == before


def test_builder_exports_only_to_a_new_path_and_never_overwrites(tmp_path: Path) -> None:
    exported = tmp_path / "exported.ipynb"
    environment = dict(os.environ)
    environment["CHIMERA_PROMPT_ONLY_NOTEBOOK_OUTPUT"] = str(exported)

    subprocess.run(
        [sys.executable, str(BUILDER_PATH)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert exported.read_bytes() == NOTEBOOK_PATH.read_bytes()

    sentinel = b"user-authored notebook edit\n"
    exported.write_bytes(sentinel)
    environment["FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE"] = "1"
    refused = subprocess.run(
        [sys.executable, str(BUILDER_PATH)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert exported.read_bytes() == sentinel


def test_seed_is_random_per_new_run_and_persisted_for_resume() -> None:
    code = code_source(load_notebook())
    assert "BASE_SEED = None" in code
    assert 'SEED_MANIFEST_PATH = RUN_DIRECTORY / "metadata" / "run_seed.json"' in code
    assert "BASE_SEED = secrets.randbelow(2**63 - len(BASE_STAGES))" in code
    assert 'seed_source = "os_entropy"' in code
    assert 'existing_records[0]["seed"]' in code
    assert '"base_seed": BASE_SEED' in code


def test_render_batch_is_measured_grown_and_bounded_with_memory_reserve() -> None:
    code = code_source(load_notebook())
    assert "render_batch_max=CHIMERA_RENDER_BATCH_MAX" in code
    assert "auto_render_batch_size=CHIMERA_AUTO_RENDER_BATCH_SIZE" in code
    assert "batch_memory_reserve_fraction=CHIMERA_BATCH_MEMORY_RESERVE_FRACTION" in code
    assert "cfg_start_ratio=CHIMERA_CFG_START_RATIO" in code
    assert "cfg_stop_ratio=CHIMERA_CFG_STOP_RATIO" in code
    assert '"cfg_start_ratio": CHIMERA_CFG_START_RATIO' in code
    assert '"cfg_stop_ratio": CHIMERA_CFG_STOP_RATIO' in code
    assert '"render_batch_report": CHIMERA_SESSION.render_batch_report' in code
    assert "binary backoff" in all_source(load_notebook())


def test_midpoint_conditioning_diagnostics_are_dense_and_persisted() -> None:
    code = code_source(load_notebook())
    assert "CHIMERA_ONE_GAP_TEST_ALPHAS" not in code
    assert 'quality_midpoint_count = int(CHIMERA_ROUND_SPECS[0]["midpoint_count"])' in code
    assert "quality_alphas = list(center_weighted_alpha_schedule(" in code
    assert '"production_alphas": quality_alphas' in code
    assert '"conditioning_interpolation": CHIMERA_CONDITIONING_INTERPOLATION' in code
    assert "CHIMERA_SESSION.conditioning_diagnostics_report" in code
    assert '"conditioning_diagnostics"' in code


def test_one_gap_quality_gate_matches_production_schedule_and_correction() -> None:
    code = code_source(load_notebook())
    assert "quality_midpoint_count, strength=CHIMERA_ALPHA_WARP_STRENGTH" in code
    assert '"quality_sheet_raw.png"' in code
    assert '"quality_sheet.png"' in code
    assert "quality_tone_result = stabilize_cyclic_tone(" in code
    assert "[0, len(quality_raw_paths) - 1]" in code
    assert '"chroma_trajectory_before_after.png"' in code
    assert '"chroma_metrics": (' in code
    assert '"output_target_mae"' in code
    assert '"repository_commit": project_commit' in code
    assert '"alpha_warp_strength": CHIMERA_ALPHA_WARP_STRENGTH' in code


def test_sap_uses_one_image_aware_intermediate_prompt_per_pair() -> None:
    notebook = load_notebook()
    code = code_source(notebook)
    sap_cell = next(
        cell for cell in notebook["cells"] if cell.get("id") == "prompt-only-chimera-17"
    )
    sap_source = "".join(sap_cell["source"])
    assert 'CHIMERA_INTERMEDIATE_PROMPT_MODE = "openai_per_pair"' in code
    assert "class ChimeraIntermediateProposal(BaseModel):" in code
    assert "intermediate_prompt:" in code
    assert sap_source.count('{"type": "input_image"') == 2
    assert "Painting A authored FLUX prompt (verbatim):" in sap_source
    assert "Painting B authored FLUX prompt (verbatim):" in sap_source
    assert "{left['prompt']}" in sap_source
    assert "{right['prompt']}" in sap_source
    assert 'image_data_url(left["path"])' in sap_source
    assert 'image_data_url(right["path"])' in sap_source
    assert "propose_chimera_intermediate(" in code
    assert "OPENAI_INTERMEDIATE_PROMPT_COUNT += len(pair_jobs)" in code
    assert "proposal.prompt_a" not in code
    assert "proposal.prompt_b" not in code
    assert 'PROMPT_CONDITIONING_CACHE[job["left"]["prompt"]]' in code
    assert 'PROMPT_CONDITIONING_CACHE[job["right"]["prompt"]]' in code
    assert "anchor_conditioning=PROMPT_CONDITIONING_CACHE[proposal.intermediate_prompt]" in code
    assert 'clean.startswith(f"{LORA_TRIGGER},")' in code
    assert "contain the LoRA trigger exactly once" in code
    assert "seventeenth-century Dutch Baroque" not in sap_source
    assert "Keep sparse scenes sparse" in sap_source
    assert "The authored endpoint prompts are immutable" in sap_source
    assert "def print_pair_prompt_plan(job, label):" in code
    assert "SOURCE AUTHORED PROMPT:" in code
    assert "OPENAI INTERMEDIATE / SAP PROMPT:" in code
    assert "TARGET AUTHORED PROMPT:" in code
    assert 'print_pair_prompt_plan(test_job, "QUALITY-GATE PROMPT PLAN")' in code
    assert "ROUND {round_number} GAP {job['gap_index']} PROMPT PLAN" in code
    assert "prompt_anchor_reliability(" in code
    assert "CHIMERA_SAP_MAX_REQUERIES" in code
    assert "reliability_passed = reliability >= CHIMERA_ANCHOR_RELIABILITY_THRESHOLD" in code
    assert "reliability_passed or not CHIMERA_ENFORCE_SAP_RELIABILITY" in code
    assert '"below diagnostic threshold; accepted without requery"' in code
    assert "validate_flux_prompt_length" in code


def test_one_model_zero_shot_chimera_pair_pipeline_is_wired() -> None:
    code = code_source(load_notebook())
    assert "ChimeraConfig," in code
    assert "ChimeraFlux2Session," in code
    assert "CHIMERA_RUNNER.prepare(" in code
    assert "CHIMERA_SESSION.invert_pair(" in code
    assert "CHIMERA_SESSION.render_pair(" in code
    assert "CHIMERA_SESSION.decode_frames_to_paths(" in code
    assert '"model_loads": 1' in code
    assert '"backward_probes": 0' in code
    assert '"training_or_endpoint_optimization": False' in code
    assert "optimize_endpoint(" not in code
    assert "FlowMorphSequenceSession" not in code


def test_recursive_pairs_are_resumable_cyclic_and_release_large_caches() -> None:
    code = code_source(load_notebook())
    assert "right = incoming[(gap_index + 1) % gap_count]" in code
    assert 'completion_path = image_directory / f"{pair_uid}.chimera.json"' in code
    assert 'completion.get("pair_fingerprint") == pair_fingerprint' in code
    assert "del source_cache, target_cache, frames" in code
    assert '"cyclic": True' in code
    assert '"duplicate_terminal_frame": False' in code


def test_glcs_and_existing_rife_finishing_stack_are_available() -> None:
    code = code_source(load_notebook())
    assert "RUN_CHIMERA_DINO_GLCS = False" in code
    assert 'CHIMERA_DINO_MODEL_ID = "facebook/dinov2-base"' in code
    assert "compute_glcs_from_similarities" in code
    assert "RUN_RIFE_POSTPROCESS = True" in code
    assert "VIDEO_SLOWDOWN_FACTOR = 4.0" in code
    assert "SOURCE_SEQUENCE_FPS = 12.0 / VIDEO_SLOWDOWN_FACTOR" in code
    assert "RIFE_MULTIPLIER = 24" in code
    assert "RIFE_PERCEPTUAL_ALLOCATION = True" in code
    assert "allocate_perceptual_subdivisions(" in code
    assert '"--multipliers-json", str(RIFE_ALLOCATION_PATH)' in code
    assert "sum(RIFE_PAIR_MULTIPLIERS) != expected_rife_budget" in code
    assert "mean pixelwise CIE76 at reduced resolution" in code
    assert "RIFE_FINAL_FPS = 24.0" in code
    assert "recursive_chimera_prompt_only_rife_ssim_loop.mp4" in code
    assert "diagnose_cyclic_flicker(" in code


def test_main_frame_hold_video_reuses_dense_rife_frames_with_two_thirds_holds() -> None:
    code = code_source(load_notebook())
    assert "RUN_MAIN_FRAME_HOLD_VIDEO = True" in code
    assert "MAIN_FRAME_HOLD_FRACTION = 2.0 / 3.0" in code
    assert "from flowmorph_klein.video_timing import plan_anchor_hold_timeline" in code
    assert "main_frame_hold_plan = plan_anchor_hold_timeline(" in code
    assert "final_frame_count=target_frame_count" in code
    assert "motion_weights=motion_weights" in code
    assert "RIFE_DENSE_PATHS[dense_index]" in code
    assert "recursive_chimera_prompt_only_main_frame_holds_loop.mp4" in code
    assert '"requested_transition_fraction": 1.0 - MAIN_FRAME_HOLD_FRACTION' in code
    assert '"segments": list(main_frame_hold_plan.segments)' in code
    assert 'RIFE_RESULTS_DIRECTORY / "main_frame_hold_report.json"' in code


def test_throwaway_reconnect_cell_restores_the_interrupted_drive_run() -> None:
    notebook = load_notebook()
    recovery = next(
        cell
        for cell in notebook["cells"]
        if cell.get("id") == "prompt-only-chimera-recover-last-run"
    )
    source = "".join(recovery["source"])
    assert "THROWAWAY RECONNECT HELPER" in source
    assert "science_path_prompt_only_chimera_0024_20260805T171421Z" in source
    assert "final_recursive_chimera_sequence_tone_stabilized.json" in source
    assert 'FINAL_RECORDS = recovered_payload["records"]' in source
    assert "Rerun sections 10-13" in source


def test_smooth_target_chroma_correction_is_endpoint_anchored_plotted_and_feeds_rife() -> None:
    code = code_source(load_notebook())
    assert "TEMPORAL_TONE_STABILIZATION_ENABLED = False" in code
    assert "TEMPORAL_CHROMA_STABILIZATION_ENABLED = True" in code
    assert "TEMPORAL_CHROMA_STRENGTH = 0.70" in code
    assert "TEMPORAL_CHROMA_THRESHOLD = 0.0" in code
    assert "TEMPORAL_CHROMA_MAX_GAIN = None" in code
    assert "TEMPORAL_CHROMA_MAX_DECREASE = None" in code
    assert "TEMPORAL_CHROMA_SMOOTHNESS = 6.0" in code
    assert "TEMPORAL_CHROMA_SMOOTHING_PASSES" not in code
    assert "luminance_enabled=TEMPORAL_TONE_STABILIZATION_ENABLED" in code
    assert "max_chroma_decrease=TEMPORAL_CHROMA_MAX_DECREASE" in code
    assert "chroma_smoothness=TEMPORAL_CHROMA_SMOOTHNESS" in code
    assert "chroma_anchor_indices = [" in code
    assert 'item.get("round") != INTERPOLATION_ROUNDS' in code
    assert "chroma_anchor_indices=(" in code
    assert 'tone_result.report["chroma_trajectory"]' in code
    assert 'chroma_trajectory["desired"]' in code
    assert "if TEMPORAL_CHROMA_MAX_DECREASE is not None" in code
    assert '"desired_output_mae"' in code
    assert '"minimum_gain"' in code
    assert '"output_curvature_rms"' in code
    assert '"chroma_trajectory_before_after.png"' in code
    assert "EXPORT_FRAME_PATHS = canonical_paths[" in code


def test_embedded_variable_density_rife_runner_is_parseable() -> None:
    notebook = load_notebook()
    setup_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "prompt-only-chimera-26")
    parsed = ast.parse("".join(setup_cell["source"]))
    runner_assignment = next(
        node
        for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RIFE_RUNNER_SOURCE" for target in node.targets)
    )
    assert isinstance(runner_assignment.value, ast.Constant)
    runner_source = runner_assignment.value.value
    ast.parse(runner_source)
    assert 'parser.add_argument("--multipliers-json", required=True)' in runner_source
    assert "output_count = sum(pair_multipliers) + 1" in runner_source
    assert "step / pair_multipliers[index]" in runner_source
