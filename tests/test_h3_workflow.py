from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from flowmorph_klein.h3_workflow import (
    DEFAULT_H3_MOTION_DIRECTIVE,
    build_default_h3_prompt,
    cyclic_h3_pairs,
    load_h3_anchor_records,
    patch_h3_ui_workflow,
    snap_h3_frame_count,
    stable_h3_fingerprint,
    validate_h3_canvas,
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
                    "nodes": [],
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


def test_h3_frame_grid_and_canvas_contract() -> None:
    assert snap_h3_frame_count(4.0) == 107
    assert snap_h3_frame_count(5.0) == 124
    validate_h3_canvas(768, 768)
    validate_h3_canvas(1344, 768)
    with pytest.raises(ValueError, match="multiples of 32"):
        validate_h3_canvas(750, 768)
    with pytest.raises(ValueError, match="capped"):
        validate_h3_canvas(1024, 1024)


def test_default_and_openai_prompts_keep_picture_markers_trigger_and_locked_camera() -> None:
    prompt = build_default_h3_prompt(duration_seconds=4.0)
    assert DEFAULT_H3_MOTION_DIRECTIVE.startswith("The objects")
    assert prompt.count("RIJKSOIL") == 1
    assert "<Picture 1>" in prompt and "<Picture 2>" in prompt
    assert "no camera movement" in prompt.lower()
    assert "Do not invent additional objects" in prompt
    assert "overall_soundscape: Silence" in prompt

    openai_prompt = wrap_openai_h3_motion(
        "The sparse sphere at #Image1 slowly changes its silhouette and surface into the faceted "
        "vessel at #Image2 while every object follows the shortest stable path through the shot.",
        duration_seconds=4.0,
    )
    assert openai_prompt.count("RIJKSOIL") == 1
    assert "<Picture 1>" in openai_prompt and "<Picture 2>" in openai_prompt
    assert "newly invented objects" in openai_prompt


def test_official_ui_workflow_is_patched_with_two_images_and_direct_dimensions() -> None:
    patched = patch_h3_ui_workflow(
        _minimal_official_template(),
        first_image="run_pair_first.png",
        last_image="run_pair_last.png",
        prompt="RIJKSOIL transition",
        width=768,
        height=768,
        duration_seconds=4.0,
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
    assert main["widgets_values"][:5] == ["RIJKSOIL transition", 768, 768, 4.0, 123]
    assert not any(link[0] in {219, 220} for link in patched["links"])
    save = next(node for node in patched["nodes"] if node["type"] == "SaveVideo")
    assert save["widgets_values"][0] == "h3/pair_0000"


def test_fingerprints_are_stable_and_sensitive() -> None:
    assert stable_h3_fingerprint({"b": 2, "a": 1}) == stable_h3_fingerprint({"a": 1, "b": 2})
    assert stable_h3_fingerprint({"a": 1}) != stable_h3_fingerprint({"a": 2})
