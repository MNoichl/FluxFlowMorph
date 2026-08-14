from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from flowmorph_klein.h3_workflow import (
    DEFAULT_H3_MOTION_DIRECTIVE,
    build_default_h3_prompt,
    cyclic_h3_pairs,
    discover_h3_finishing_source,
    h3_ui_workflow_controls,
    load_h3_anchor_records,
    patch_h3_ui_workflow,
    recover_h3_finishing_source,
    snap_h3_frame_count,
    stable_h3_fingerprint,
    strip_h3_source_only_tokens,
    validate_h3_canvas,
    validate_h3_prompt_byte_budget,
    wrap_openai_h3_motion,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TEMPLATE_FIXTURE = Path("/tmp/video_minimax_h3_i2v.json")


def _write_source_run(root: Path) -> Path:
    source = root / "source_run"
    (source / "metadata").mkdir(parents=True)
    (source / "base_frames").mkdir()
    records = []
    for index in range(3):
        image_path = source / "base_frames" / f"anchor_{index}.png"
        Image.new("RGB", (32, 32), (index * 20, 10, 30)).save(image_path)
        records.append(
            {
                "uid": f"base_{index:03d}",
                "path": f"/stale/colab/path/{image_path.name}",
                "prompt": f"RIJKSOIL, authored scene {index}",
                "generation_prompt": f"RIJKSOIL, generated scene {index}",
            }
        )
    (source / "metadata" / "base_manifest.json").write_text(
        json.dumps({"complete": True, "records": records}), encoding="utf-8"
    )
    return source


def _write_finishing_source(
    run_directory: Path,
    *,
    stage: str,
    timestamp: float,
) -> None:
    specifications = {
        "border_stabilized": (
            "minimax_h3_border_stabilized_cyclic_loop.mp4",
            "border_stabilization.json",
            "frame_count",
        ),
        "rife_x2": (
            "minimax_h3_rife_x2_cyclic_loop.mp4",
            "rife_report.json",
            "output_unique_frames",
        ),
        "native_h3": (
            "minimax_h3_native_cyclic_loop.mp4",
            "native_assembly.json",
            "native_unique_frames",
        ),
    }
    video_name, report_name, count_key = specifications[stage]
    video_path = run_directory / "video" / video_name
    report_path = run_directory / "metadata" / report_name
    video_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"finished-video")
    report_path.write_text(
        json.dumps({count_key: 120, "fps": 48.0}),
        encoding="utf-8",
    )
    os.utime(video_path, (timestamp, timestamp))
    os.utime(report_path, (timestamp, timestamp))


def _minimal_official_template() -> dict:
    return {
        "nodes": [
            {
                "id": 92,
                "type": "SaveVideo",
                "inputs": [{"name": "video", "type": "VIDEO", "link": 194}],
                "outputs": [{"name": "video", "type": "VIDEO", "links": None}],
                "widgets_values": ["video/MiniMax_H3", "auto", "auto"],
            },
            {
                "id": 114,
                "type": "LoadImage",
                "pos": [0, 0],
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [218]},
                    {"name": "MASK", "type": "MASK", "links": None},
                ],
                "widgets_values": ["old.png", "image"],
            },
            {
                "id": 115,
                "type": "ResolutionSelector",
                "inputs": [],
                "outputs": [
                    {"name": "width", "type": "INT", "links": [219]},
                    {"name": "height", "type": "INT", "links": [220]},
                ],
                "widgets_values": ["1:1 (Square)", 0.4, 32],
            },
            {
                "id": 105,
                "type": "4c314f31-ecda-4b08-ae98-faaba1bf613f",
                "inputs": [
                    {"name": "first_frame", "type": "IMAGE", "link": 218},
                    {"name": "last_frame", "type": "IMAGE", "link": None},
                    {"name": "width", "type": "INT", "link": 219},
                    {"name": "height", "type": "INT", "link": 220},
                ],
                "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": [194]}],
                "widgets_values": ["old", 1344, 768, 5, 1, "u", "c", "v", "a"],
            },
        ],
        "links": [
            [194, 105, 0, 92, 0, "VIDEO"],
            [218, 114, 0, 105, 0, "IMAGE"],
            [219, 115, 0, 105, 2, "INT"],
            [220, 115, 1, 105, 3, "INT"],
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": "4c314f31-ecda-4b08-ae98-faaba1bf613f",
                    "inputs": [],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 104,
                            "type": "MiniMaxH3ImageToVideo",
                            "widgets_values": [
                                "Vaporwave Greek statue title sequence, STARRING LATENT CONTROLNET",
                                1344,
                                768,
                                73,
                            ],
                        },
                        {"id": 111, "type": "PrimitiveFloat", "widgets_values": [2]},
                        {"id": 15, "type": "RandomNoise", "widgets_values": [1, "randomize"]},
                        {
                            "id": 6,
                            "type": "UNETLoader",
                            "widgets_values": ["old_diffusion.safetensors", "default"],
                        },
                        {
                            "id": 13,
                            "type": "CLIPLoader",
                            "widgets_values": ["old_encoder.safetensors", "minimax", "default"],
                        },
                        {
                            "id": 11,
                            "type": "VAELoader",
                            "widgets_values": ["old_video_vae.safetensors"],
                        },
                        {
                            "id": 24,
                            "type": "VAELoader",
                            "widgets_values": ["old_audio_vae.safetensors"],
                        },
                    ],
                    "links": [],
                }
            ]
        },
    }


def test_load_records_recovers_stale_colab_paths_and_closes_cycle(tmp_path: Path) -> None:
    records = load_h3_anchor_records(_write_source_run(tmp_path))
    assert [record["uid"] for record in records] == ["base_000", "base_001", "base_002"]
    assert records[0]["resolved_path"].endswith("base_frames/anchor_0.png")
    assert records[0]["authored_prompt"] == "RIJKSOIL, generated scene 0"
    pairs = cyclic_h3_pairs(records)
    assert len(pairs) == 3
    assert pairs[-1]["left"]["uid"] == "base_002"
    assert pairs[-1]["right"]["uid"] == "base_000"


def test_load_records_falls_back_to_natural_sorted_plain_image_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plain images"
    nested = source / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (32, 32), "red").save(source / "frame_10.PNG")
    Image.new("RGB", (32, 32), "green").save(source / "frame_2.jpg")
    Image.new("RGB", (32, 32), "blue").save(nested / "frame_20.webp")
    (source / "notes.txt").write_text("not an anchor", encoding="utf-8")
    hidden = source / ".thumbnails"
    hidden.mkdir()
    Image.new("RGB", (32, 32), "black").save(hidden / "ignored.png")

    records = load_h3_anchor_records(source)

    assert [record["path"] for record in records] == [
        "frame_2.jpg",
        "frame_10.PNG",
        "nested/frame_20.webp",
    ]
    assert [record["source_index"] for record in records] == [0, 1, 2]
    assert all(record["source_kind"] == "directory_scan" for record in records)
    assert all(record["authored_prompt"] == "" for record in records)
    assert all(Path(record["resolved_path"]).is_absolute() for record in records)
    assert len({record["uid"] for record in records}) == 3


def test_manifest_prompts_are_optional_for_image_only_sources(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    manifest_path = source / "metadata" / "base_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0].pop("prompt")
    payload["records"][0].pop("generation_prompt")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_h3_anchor_records(source)

    assert records[0]["authored_prompt"] == ""
    assert records[0]["source_kind"] == "base_manifest"
    assert records[1]["authored_prompt"] == "RIJKSOIL, generated scene 1"


def test_plain_image_directory_requires_two_supported_images(tmp_path: Path) -> None:
    source = tmp_path / "one_image"
    source.mkdir()
    Image.new("RGB", (32, 32), "white").save(source / "only.png")

    with pytest.raises(ValueError, match="at least two supported images; found 1"):
        load_h3_anchor_records(source)


def test_h3_frame_grid_and_canvas_contract() -> None:
    assert snap_h3_frame_count(4.0) == 107
    assert snap_h3_frame_count(5.0) == 124
    assert snap_h3_frame_count(6.0) == 158
    validate_h3_canvas(768, 768)
    validate_h3_canvas(1344, 768)
    with pytest.raises(ValueError, match="multiples of 32"):
        validate_h3_canvas(750, 768)
    with pytest.raises(ValueError, match="capped"):
        validate_h3_canvas(1024, 1024)


def test_finishing_source_recovery_prefers_current_then_newest_completed_run(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "minimax_h3_interpolations"
    empty_current_run = project_root / "source_a" / "h3_fl2va_0003"
    empty_current_run.mkdir(parents=True)
    older_run = project_root / "source_a" / "h3_fl2va_0001"
    newest_run = project_root / "source_b" / "h3_fl2va_0002"
    _write_finishing_source(older_run, stage="border_stabilized", timestamp=10.0)
    _write_finishing_source(newest_run, stage="native_h3", timestamp=20.0)

    recovered = discover_h3_finishing_source(project_root, preferred_run=empty_current_run)
    assert recovered is not None
    assert recovered["selection"] == "latest_completed_h3_run"
    assert recovered["run_directory"] == newest_run
    assert recovered["stage"] == "native_h3"
    assert recovered["frame_count"] == 120
    assert recovered["fps"] == 48.0

    _write_finishing_source(empty_current_run, stage="rife_x2", timestamp=5.0)
    recovered = discover_h3_finishing_source(project_root, preferred_run=empty_current_run)
    assert recovered is not None
    assert recovered["selection"] == "current_run"
    assert recovered["run_directory"] == empty_current_run
    assert recovered["stage"] == "rife_x2"


def test_finishing_source_recovery_rejects_incomplete_persistent_artifacts(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    report_path = run_directory / "metadata" / "native_assembly.json"
    video_path = run_directory / "video" / "minimax_h3_native_cyclic_loop.mp4"
    report_path.parent.mkdir(parents=True)
    video_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"native_unique_frames": 120, "fps": 24}), encoding="utf-8")
    video_path.write_bytes(b"")

    assert recover_h3_finishing_source(run_directory) is None
    assert discover_h3_finishing_source(tmp_path, preferred_run=run_directory) is None


def test_default_and_openai_prompts_remove_flux_token_and_lock_scene_content() -> None:
    prompt = build_default_h3_prompt(
        duration_seconds=6.0,
        motion_directive="RIJKSOIL, " + DEFAULT_H3_MOTION_DIRECTIVE,
        trigger="RIJKSOIL",
    )
    assert DEFAULT_H3_MOTION_DIRECTIVE.startswith("A static locked-off view")
    assert "RIJKSOIL" not in prompt
    assert "<Picture 1>" in prompt and "<Picture 2>" in prompt
    assert "<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark" in prompt
    assert "<Picture 2> (from [Shot 1]) aligns with the 6.00-second mark" in prompt
    assert "Static Shot with unchanged framing, lens, and viewpoint" in prompt
    assert "left, center, and right transform concurrently" in prompt
    assert "no frame-spanning dividing line, moving sheet or band" in prompt
    assert "background planes and broad color, shadow, or texture fields blend" in prompt
    assert "overlapping opaque intermediate states" in prompt
    assert "adjacent-frame increments are nearly imperceptible in isolation" in prompt
    assert "progressively smaller adjustments" in prompt
    assert "boundary advances" not in prompt
    assert "overall_soundscape: N/A" in prompt
    assert strip_h3_source_only_tokens("RIJKSOIL, a sparse scene") == "a sparse scene"

    openai_prompt = wrap_openai_h3_motion(
        "integrated_multimodal_description: [Shot 1] RIJKSOIL, the sparse sphere in Picture 1 "
        "changes through small local silhouette and surface adjustments into the faceted vessel "
        "in Picture 2 while the background and tabletop remain registered and every visible form "
        "follows its shortest coherent path through the static shot.",
        duration_seconds=6.0,
        trigger="RIJKSOIL",
    )
    assert "RIJKSOIL" not in openai_prompt
    assert openai_prompt.count("integrated_multimodal_description:") == 1
    assert openai_prompt.count("[Shot 1]") == 3  # Two alignment anchors plus one shot body.
    assert "<Picture 1>" in openai_prompt and "<Picture 2>" in openai_prompt
    assert "#Image1" not in openai_prompt and "#Image2" not in openai_prompt
    assert "small local silhouette and surface adjustments" in openai_prompt
    assert "Static Shot with unchanged framing, lens, and viewpoint" in openai_prompt
    assert "concurrently at overlapping but slightly offset rates" in openai_prompt
    assert "Adjacent frames differ through minute coherent increments" in openai_prompt
    assert "nearly imperceptible in isolation" in openai_prompt
    assert "no individual frame change calls attention to itself" in openai_prompt
    assert "no frame-spanning dividing line, moving sheet or band" in openai_prompt
    assert "background planes and broad color, shadow, or texture fields blend" in openai_prompt
    assert "overlapping opaque intermediate states" in openai_prompt
    assert "overall_soundscape: N/A" in openai_prompt


def test_openai_prompt_wrapper_rejects_mid_sentence_truncation() -> None:
    with pytest.raises(ValueError, match="complete sentence"):
        wrap_openai_h3_motion(
            "[Shot 1] Picture 1 changes continuously into Picture 2 while the Static Shot "
            "keeps all forms registered, and the final glass boundary narrows into the clear-s",
            duration_seconds=6.0,
        )


def test_h3_prompt_budget_uses_utf8_upper_bound_and_reserves_conditioning() -> None:
    assert validate_h3_prompt_byte_budget(
        "one",
        max_utf8_bytes=4,
        model_context_tokens=20,
        reserved_condition_tokens=10,
    ) == 3
    with pytest.raises(ValueError, match="operational maximum"):
        validate_h3_prompt_byte_budget(
            "three",
            max_utf8_bytes=4,
            model_context_tokens=20,
            reserved_condition_tokens=10,
        )
    assert validate_h3_prompt_byte_budget(
        "é",
        max_utf8_bytes=4,
        model_context_tokens=20,
        reserved_condition_tokens=10,
    ) == 2
    assert validate_h3_prompt_byte_budget(
        " one ",
        max_utf8_bytes=5,
        model_context_tokens=20,
        reserved_condition_tokens=10,
    ) == 5
    with pytest.raises(ValueError, match="worst-case prompt budget exceeds"):
        validate_h3_prompt_byte_budget(
            "one",
            max_utf8_bytes=12,
            model_context_tokens=20,
            reserved_condition_tokens=10,
        )


def test_official_ui_workflow_is_patched_with_two_images_and_direct_dimensions() -> None:
    patched = patch_h3_ui_workflow(
        _minimal_official_template(),
        first_image="run_pair_first.png",
        last_image="run_pair_last.png",
        prompt="RIJKSOIL, continuous sparse-object transformation",
        width=768,
        height=768,
        duration_seconds=6.0,
        seed=123,
        output_prefix="h3/pair_0000",
        diffusion_model="diffusion.safetensors",
        text_encoder="encoder.safetensors",
        video_vae="video.safetensors",
        audio_vae="audio.safetensors",
    )
    load_nodes = [node for node in patched["nodes"] if node["type"] == "LoadImage"]
    assert len(load_nodes) == 2
    assert {node["widgets_values"][0] for node in load_nodes} == {
        "run_pair_first.png",
        "run_pair_last.png",
    }
    main = next(node for node in patched["nodes"] if node["id"] == 105)
    assert main["inputs"][0]["link"] is not None
    assert main["inputs"][1]["link"] is not None
    assert main["inputs"][2]["link"] is None
    assert main["inputs"][3]["link"] is None
    assert main["widgets_values"][:5] == [
        "continuous sparse-object transformation",
        768,
        768,
        6.0,
        123,
    ]
    assert not any(link[0] in {219, 220} for link in patched["links"])
    save = next(node for node in patched["nodes"] if node["type"] == "SaveVideo")
    assert save["widgets_values"][0] == "h3/pair_0000"
    assert h3_ui_workflow_controls(patched) == {
        "prompt": "continuous sparse-object transformation",
        "width": 768,
        "height": 768,
        "frame_count_fallback": 158,
        "duration_seconds": 6.0,
        "seed": 123,
        "seed_control": "fixed",
        "diffusion_model": "diffusion.safetensors",
        "text_encoder": "encoder.safetensors",
        "video_vae": "video.safetensors",
        "audio_vae": "audio.safetensors",
    }


@pytest.mark.skipif(not OFFICIAL_TEMPLATE_FIXTURE.is_file(), reason="official template fixture unavailable")
def test_real_official_template_demo_defaults_are_fully_replaced() -> None:
    template = json.loads(OFFICIAL_TEMPLATE_FIXTURE.read_text(encoding="utf-8"))
    prompt = build_default_h3_prompt(duration_seconds=6.0)
    patched = patch_h3_ui_workflow(
        template,
        first_image="first.png",
        last_image="last.png",
        prompt=prompt,
        width=768,
        height=768,
        duration_seconds=6.0,
        seed=456,
        output_prefix="h3/real_fixture",
        diffusion_model="diffusion.safetensors",
        text_encoder="encoder.safetensors",
        video_vae="video_vae.safetensors",
        audio_vae="audio_vae.safetensors",
    )
    serialized = json.dumps(patched)
    assert "Vaporwave" not in serialized
    assert "LATENT CONTROLNET" not in serialized
    assert "DIRECTED BY COMFYUI" not in serialized
    controls = h3_ui_workflow_controls(patched)
    assert controls["duration_seconds"] == 6.0
    assert (controls["width"], controls["height"]) == (768, 768)


def test_fingerprints_are_stable_and_sensitive() -> None:
    assert stable_h3_fingerprint({"b": 2, "a": 1}) == stable_h3_fingerprint({"a": 1, "b": 2})
    assert stable_h3_fingerprint({"a": 1}) != stable_h3_fingerprint({"a": 2})
