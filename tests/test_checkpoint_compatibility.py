from __future__ import annotations

import json

import pytest
import torch

from flowmorph_klein.checkpoints import (
    CheckpointError,
    CheckpointCompatibilityError,
    load_endpoint_checkpoint,
    save_endpoint_checkpoint,
)


def _metadata() -> dict:
    return {
        "endpoint": "source",
        "model_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "model_revision": "model-revision",
        "lora_source": "org/adapter",
        "lora_revision": "adapter-revision",
        "lora_file_sha256": "a" * 64,
        "lora_scale": 1.0,
        "prompt_checksum": "prompt",
        "source_image_checksum": "source",
        "processed_image_checksum": "processed",
        "preprocessing_hash": "preprocess",
        "scheduler_configuration": {"points": 100, "start": 35},
        "start_timestep_index": 35,
        "latent_shape": [1, 8, 4],
        "precision_configuration": {"master": "float32"},
        "diffusers_commit": "diffusers-revision",
        "flowmorph_commit": "flowmorph-revision",
        "flux2_commit": "flux2-revision",
    }


def test_checkpoint_save_load_roundtrip(tmp_path):
    z = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
    tensors = {"z": z, "delta": z / 10, "u": torch.zeros_like(z)}
    save_endpoint_checkpoint(tmp_path / "source", tensors, _metadata())

    loaded = load_endpoint_checkpoint(tmp_path / "source", expected_metadata=_metadata())
    assert torch.equal(loaded.tensors["z"], z)
    assert json.loads((tmp_path / "source" / "metadata.json").read_text())["latent_shape"] == [1, 8, 4]


def test_checkpoint_detects_interrupted_tensor_metadata_pair(tmp_path):
    z = torch.zeros(1, 8, 4)
    directory = tmp_path / "source"
    save_endpoint_checkpoint(directory, {"z": z, "delta": z, "u": z}, _metadata())
    generation = (directory / "LATEST").read_text(encoding="utf-8").strip()
    for metadata_path in (
        directory / "metadata.json",
        directory / ".generations" / generation / "metadata.json",
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["transaction_id"] += 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CheckpointError, match="transaction mismatch"):
        load_endpoint_checkpoint(directory)


def test_checkpoint_recovers_previous_complete_generation(tmp_path):
    directory = tmp_path / "source"
    first = torch.zeros(1, 8, 4)
    second = torch.ones(1, 8, 4)
    first_metadata = {**_metadata(), "completed_steps": 25}
    second_metadata = {**_metadata(), "completed_steps": 50}
    save_endpoint_checkpoint(
        directory,
        {"z": first, "delta": first, "u": first},
        first_metadata,
    )
    first_generation = (directory / "LATEST").read_text(encoding="utf-8").strip()
    save_endpoint_checkpoint(
        directory,
        {"z": second, "delta": second, "u": second},
        second_metadata,
    )
    newest_generation = (directory / "LATEST").read_text(encoding="utf-8").strip()
    assert newest_generation != first_generation

    # Simulate a disconnect/torn publication affecting both the current
    # generation and its compatibility-layout copy.
    for metadata_path in (
        directory / "metadata.json",
        directory / ".generations" / newest_generation / "metadata.json",
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["transaction_id"] += 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    recovered = load_endpoint_checkpoint(directory, expected_metadata=_metadata())
    assert recovered.metadata["completed_steps"] == 25
    assert torch.equal(recovered.tensors["z"], first)


def test_checkpoint_recovers_when_newest_safetensors_is_truncated(tmp_path):
    directory = tmp_path / "source"
    first = torch.zeros(1, 8, 4)
    second = torch.ones(1, 8, 4)
    save_endpoint_checkpoint(
        directory,
        {"z": first, "delta": first, "u": first},
        {**_metadata(), "completed_steps": 25},
    )
    save_endpoint_checkpoint(
        directory,
        {"z": second, "delta": second, "u": second},
        {**_metadata(), "completed_steps": 50},
    )
    newest_generation = (directory / "LATEST").read_text(encoding="utf-8").strip()
    (directory / "tensors.safetensors").write_bytes(b"truncated")
    (
        directory
        / ".generations"
        / newest_generation
        / "tensors.safetensors"
    ).write_bytes(b"truncated")

    recovered = load_endpoint_checkpoint(directory, expected_metadata=_metadata())
    assert recovered.metadata["completed_steps"] == 25
    assert torch.equal(recovered.tensors["z"], first)


def test_checkpoint_refuses_non_finite_endpoint_or_optimizer_tensor(tmp_path):
    z = torch.zeros(1, 8, 4)
    invalid = z.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(CheckpointError, match="non-finite"):
        save_endpoint_checkpoint(
            tmp_path / "source",
            {"z": z, "delta": invalid, "u": z},
            _metadata(),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_revision", "changed"),
        ("lora_file_sha256", "b" * 64),
        ("lora_scale", 0.8),
        ("prompt_checksum", "changed"),
        ("processed_image_checksum", "changed"),
        ("preprocessing_hash", "changed"),
        ("scheduler_configuration", {"points": 99, "start": 35}),
        ("start_timestep_index", 34),
        ("latent_shape", [1, 7, 4]),
        ("endpoint", "target"),
        ("flowmorph_commit", "changed"),
        ("flux2_commit", "changed"),
    ],
)
def test_checkpoint_rejects_compatibility_change(tmp_path, field, replacement):
    z = torch.zeros(1, 8, 4)
    save_endpoint_checkpoint(tmp_path / "source", {"z": z, "delta": z, "u": z}, _metadata())
    expected = _metadata()
    expected[field] = replacement
    with pytest.raises(CheckpointCompatibilityError) as error:
        load_endpoint_checkpoint(tmp_path / "source", expected_metadata=expected)
    assert field in error.value.mismatches
