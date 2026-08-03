from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "StillLife_Recursive_CHIMERA_Prompt_Only.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def all_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_chimera_notebook_is_parseable_and_has_colab_badge() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 33
    assert "StillLife_Recursive_CHIMERA_Prompt_Only.ipynb" in "".join(
        notebook["cells"][0]["source"]
    )
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
    setup = "".join(notebook["cells"][6]["source"])
    code = code_source(notebook)
    assert "CHIMERA_BOOTSTRAP_B85" not in code
    assert "b85decode" not in code
    assert "spec_from_file_location" not in code
    assert "import flowmorph_klein" in setup
    assert "from flowmorph_klein.chimera import (" in code


def test_flat_round_paper_defaults_and_flux_memory_adaptations_are_explicit() -> None:
    text = all_source(load_notebook())
    expected = (
        'CHIMERA_ROUND_SPECS = [\n'
        '    {"midpoint_count": 10},\n'
        "]"
    )
    assert expected in text
    assert "CHIMERA_INVERSION_STEPS = 50" in text
    assert "CHIMERA_DENOISING_STEPS = 50" in text
    assert "CHIMERA_ACI_WEIGHT = 0.4" in text
    assert "CHIMERA_SAP_ACTIVE_RATIO = 0.2" in text
    assert "CHIMERA_ANCHOR_RELIABILITY_THRESHOLD = 0.45" in text
    assert 'CHIMERA_LTM_MODE = "fft"' in text
    assert "CHIMERA_LTM_BANDS = 16" in text
    assert "CHIMERA_LTM_CALIBRATION_ANCHORS = 4" in text
    assert "REUSE_CHIMERA_LTM_CALIBRATION = True" in text
    assert "CHIMERA_AUTO_RENDER_BATCH_SIZE = True" in text
    assert "CHIMERA_RENDER_BATCH_MAX = 10" in text
    assert "CHIMERA_BATCH_MEMORY_RESERVE_FRACTION = 0.10" in text
    assert "CHIMERA_BATCH_MEMORY_RESERVE_GIB = 2.0" in text
    assert "CHIMERA_DECODE_BATCH_SIZE = 10" in text
    assert 'CHIMERA_CACHE_STORAGE = "int8"' in text
    assert "CHIMERA_CACHE_STRIDE = 2" in text
    assert "measures their timestep correspondence" in text


def test_fft_ltm_is_calibrated_persisted_and_part_of_pair_identity() -> None:
    code = code_source(load_notebook())
    assert "LTMCalibration," in code
    assert 'RUN_DIRECTORY / "metadata" / "chimera_ltm_calibration.json"' in code
    assert "CHIMERA_SESSION.calibrate_ltm(" in code
    assert "CHIMERA_SESSION.set_ltm_calibration(candidate)" in code
    assert '"descriptor_normalized": False' in code
    assert '"calibration_fingerprint": CHIMERA_LTM_CALIBRATION.fingerprint' in code
    assert '"ltm_fingerprint": CHIMERA_LTM_FINGERPRINT' in code


def test_all_prompts_are_active_and_original_anchor_init_is_restored() -> None:
    notebook = load_notebook()
    stages_source = "".join(notebook["cells"][4]["source"])
    stages_tree = ast.parse(stages_source)
    assignment = next(
        node
        for node in stages_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BASE_STAGES"
            for target in node.targets
        )
    )
    stages = ast.literal_eval(assignment.value)
    code = code_source(notebook)
    assert len(stages) == 12
    assert stages[0]["id"] == "01_astronomy"
    assert stages[-1]["id"] == "12_computation_math"
    assert all("the candle the only light" in stage["prompt"] for stage in stages)
    assert "nuclear_atomic_optical_physics" not in stages_source
    assert "BASE_REFERENCE_STRENGTH = 0.3" in code
    assert "REFERENCE_BACKGROUND = (116, 105, 91)" in code
    assert "reference_blend=BASE_REFERENCE_STRENGTH" in code
    assert "background_rgb=REFERENCE_BACKGROUND" in code
    assert 'kwargs["image"] = reference' in code
    assert "prepare_flux2_klein_img2img_inputs" not in code
    assert "BASE_REFERENCE_DENOISE_STRENGTH" not in code


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
    assert '"render_batch_report": CHIMERA_SESSION.render_batch_report' in code
    assert "binary backoff" in all_source(load_notebook())


def test_sap_triplet_is_image_aware_reliable_and_bounded() -> None:
    code = code_source(load_notebook())
    assert "class ChimeraPromptTriplet(BaseModel):" in code
    assert "anchor_prompt:" in code
    assert "prompt_a:" in code
    assert "prompt_b:" in code
    assert '"type": "input_image"' in code
    assert "prompt_anchor_reliability(" in code
    assert "CHIMERA_SAP_MAX_REQUERIES" in code
    assert "reliability >= CHIMERA_ANCHOR_RELIABILITY_THRESHOLD" in code
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
    assert "RIFE_MULTIPLIER = int(round(2 * VIDEO_SLOWDOWN_FACTOR))" in code
    assert "RIFE_FINAL_FPS = 24.0" in code
    assert "recursive_chimera_prompt_only_rife_ssim_loop.mp4" in code
    assert "diagnose_cyclic_flicker(" in code
