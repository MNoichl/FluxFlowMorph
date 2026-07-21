from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flowmorph_klein.art_loop import (
    ArtLoopError,
    apply_prompt_prefix,
    generate_mainframes,
    load_art_loop_spec,
    make_soft_reference,
    persist_artifact_tree,
)


EXAMPLE = Path(__file__).parents[1] / "art_projects/prompts/example_still_life_loop.json"


def test_example_art_loop_is_valid_and_closed() -> None:
    spec = load_art_loop_spec(EXAMPLE)

    assert len(spec.mainframes) == 3
    assert len(spec.transitions) == 3
    assert all(len(transition.bridge_prompts) == 20 for transition in spec.transitions)
    assert spec.transitions[-1].to_id == spec.mainframes[0].id
    assert spec.generation.prompt_prefix == "RIJKSOIL"
    assert spec.lora.revision == "042a31d6cd09bf55195f820461fac60b1a358409"


def test_art_loop_rejects_wrong_bridge_prompt_count(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["transitions"][0]["bridge_prompts"].pop()
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtLoopError, match="exactly 20 bridge prompts"):
        load_art_loop_spec(invalid)


def test_soft_reference_is_faint_blurred_previous_image() -> None:
    previous = Image.new("RGB", (16, 16), (255, 0, 0))
    reference = make_soft_reference(
        previous,
        reference_blend=0.2,
        blur_radius=0.0,
        background_rgb=(100, 100, 100),
    )

    assert reference.size == previous.size
    assert tuple(np.asarray(reference)[0, 0]) == (131, 80, 80)


def test_prompt_prefix_is_applied_once() -> None:
    assert apply_prompt_prefix("RIJKSOIL", "a table") == "RIJKSOIL, a table"
    assert apply_prompt_prefix("RIJKSOIL", "RIJKSOIL, a table") == "RIJKSOIL, a table"


def test_mainframe_generation_conditions_only_after_first_frame(tmp_path: Path) -> None:
    spec = load_art_loop_spec(EXAMPLE)

    class FakePipeline:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            value = 40 * len(self.calls)
            return SimpleNamespace(images=[Image.new("RGB", (512, 512), (value, value, value))])

    pipeline = FakePipeline()
    records = generate_mainframes(pipeline, spec, tmp_path, generator_device="cpu")

    assert len(records) == len(spec.mainframes)
    assert "image" not in pipeline.calls[0]
    assert all("image" in call for call in pipeline.calls[1:])
    assert all(record.path.is_file() for record in records)
    assert records[0].soft_reference_path is None
    assert all(record.soft_reference_path is not None for record in records[1:])
    assert all(call["prompt"].startswith("RIJKSOIL, ") for call in pipeline.calls)


def test_persistent_artifacts_are_timestamped_and_auto_numbered(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "frame.png").write_bytes(b"frame")
    persistent_root = tmp_path / "drive"
    now = datetime(2026, 7, 21, 9, 45, tzinfo=timezone.utc)

    first = persist_artifact_tree(
        source,
        persistent_root,
        project_name="still_life",
        label="pears_to_tulips",
        now=now,
    )
    second = persist_artifact_tree(
        source,
        persistent_root,
        project_name="still_life",
        label="tulips_to_lemons",
        now=now,
    )

    assert first.destination.name == "still_life_0001_20260721T094500Z_pears_to_tulips"
    assert second.destination.name == "still_life_0002_20260721T094500Z_tulips_to_lemons"
    assert (first.destination / "frame.png").read_bytes() == b"frame"
    marker = json.loads((first.destination / "COPY_COMPLETE.json").read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["file_count"] == 1
    assert not list(persistent_root.rglob("*.partial"))
