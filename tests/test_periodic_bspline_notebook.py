from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "StillLife_Periodic_BSpline_FlowMorph.ipynb"
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
    assert "StillLife_Periodic_BSpline_FlowMorph.ipynb" in "".join(
        notebook["cells"][0]["source"]
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell.get("source", [])))


def test_path_is_isolated_prompt_first_and_has_no_remote_prompt_dependency() -> None:
    code = code_source(load_notebook())
    assert "BASE_STAGES = [" in code
    assert "BASE_PROMPT_COUNT = None" in code
    assert "if BASE_PROMPT_COUNT is None:" in code
    assert "BASE_PROMPT_COUNT = len(BASE_STAGES)" in code
    assert "PeriodicSplineFlowMorphRenderer" in code
    assert "FLOWMORPH_ROUND_SPECS" not in code
    assert "OPENAI_CLIENT" not in code
    assert "OPENAI_KEY_FILENAME" not in code
    assert "MidpointProposal" not in code
    assert "MASK_" not in code
    assert "TRAJECTORY_" not in code


def test_periodic_spline_timing_is_exposed_and_regularized() -> None:
    code = code_source(load_notebook())
    assert "SPLINE_FRAMES_PER_ANCHOR = 10" in code
    assert "SPLINE_TIMING_DISTANCE_STRENGTH = 0.45" in code
    assert "SPLINE_TIMING_DISTANCE_EXPONENT = 0.50" in code
    assert "SPLINE_TIMING_MAX_SEGMENT_RATIO = 1.75" in code
    assert "periodic_thumbnail_distances(" in code
    assert "regularized_periodic_timing(" in code
    assert "allocate_periodic_segment_frames(" in code
    assert "sample_periodic_timeline(" in code
    assert "np.eye(len(BASE_RECORDS))" in code
    assert "for derivative in (0, 1, 2):" in code
    assert '"runtime_seam_audit": "C2 passed"' in code
    assert "SPLINE_TOTAL_FRAMES = BASE_PROMPT_COUNT * SPLINE_FRAMES_PER_ANCHOR" in code
    assert '"terminal_duplicate": False' in code
    assert '"seam_continuity": "C2"' in code


def test_anchor_generation_has_weak_continuity_without_flat_canvas() -> None:
    code = code_source(load_notebook())
    assert "make_soft_reference(" in code
    assert "prepare_flux2_klein_img2img_inputs(" in code
    assert "reference_blend=1.0" in code
    assert "BASE_REFERENCE_BLUR = 16.0" in code
    assert "BASE_REFERENCE_GRAIN_STRENGTH = 0.035" in code
    assert "BASE_REFERENCE_DENOISE_STRENGTH = 0.75" in code
    assert "REFERENCE_BACKGROUND" not in code
    assert "background_rgb=" not in code
    assert 'kwargs["sigmas"] = list(generation_inputs.sigmas)' in code
    assert 'kwargs["latents"] = generation_inputs.latents' in code


def test_unique_endpoint_fit_and_canonical_anchor_contract_are_retained() -> None:
    code = code_source(load_notebook())
    assert "ENDPOINT_CACHE = {}" in code
    assert "def fit_sequence_endpoints(records, progress_label):" in code
    assert 'if record["uid"] not in ENDPOINT_CACHE' in code
    assert "ensure_sequence_assets(BASE_RECORDS)" in code
    assert 'fit_sequence_endpoints(BASE_RECORDS, "Periodic endpoint fit")' in code
    assert "SEQUENCE_SESSION.render_endpoint_reconstructions(" in code
    assert "ENDPOINT_RECONSTRUCTION_PATHS[anchor[\"uid\"]]" in code
    assert '"model_loads": 1' in code
    assert '"backward_probes": 1' in code


def test_full_render_streams_global_spline_and_preserves_exact_knots() -> None:
    code = code_source(load_notebook())
    assert "PeriodicFlowMorphSpline(" in code
    assert "PeriodicConditioningSpline(" in code
    assert "SPLINE_RENDERER.render(" in code
    assert "for start in range(0, len(interior_jobs), SPLINE_STREAM_CHUNK_SIZE):" in code
    assert "SEQUENCE_SESSION.decode_frames_to_paths(frames, paths)" in code
    assert '"kind": kind' in code
    assert '"exact_canonical_anchor_knots": True' in code
    assert "final_periodic_bspline_flowmorph_sequence.json" in code


def test_tone_flicker_and_rife_treat_the_sequence_as_circular() -> None:
    code = code_source(load_notebook())
    assert "TEMPORAL_TONE_STABILIZATION_ENABLED = False" in code
    assert "stabilize_cyclic_tone(" in code
    assert "diagnose_cyclic_flicker(" in code
    assert "RUN_RIFE_POSTPROCESS = True" in code
    assert "periodic_bspline_flowmorph_rife_ssim_loop.mp4" in code
    assert (
        'RIFE_INPUT_DIRECTORY / f"{len(EXPORT_FRAME_PATHS):07d}.png"'
        in code
    )
    assert "if not np.array_equal(first_array, last_array):" in code
    assert "RIFE_DENSE_PATHS = dense_with_duplicate[:-1]" in code
    assert "dense_luma[index - 1], dense_luma[index]" in code
    assert '"terminal_duplicate_in_video": False' in code
