from __future__ import annotations

import json
import zipfile

import pytest

from flowmorph_klein.packaging import PackagingError, create_run_archive, sha256_file


def _minimal_run(root):
    root.mkdir()
    (root / "config.resolved.yaml").write_text("model: base-9b\n")
    (root / "run_manifest.json").write_text(json.dumps({"run_id": "run_001"}))
    (root / "raw_frames").mkdir()
    (root / "raw_frames" / "frame_000.png").write_bytes(b"fake-png")
    (root / "checkpoints" / "source").mkdir(parents=True)
    (root / "checkpoints" / "source" / "tensors.safetensors").write_bytes(b"endpoint-state")
    pending = root / "checkpoints" / "source" / ".generations" / ".pending-interrupted"
    pending.mkdir(parents=True)
    (pending / "tensors.safetensors").write_bytes(b"partial-endpoint-state")
    (root / "hf_cache").mkdir()
    (root / "hf_cache" / "model.safetensors").write_bytes(b"model-weights")
    (root / "inputs").mkdir()
    (root / "inputs" / "adapter.safetensors").write_bytes(b"adapter-weights")


def test_archive_uses_allowlist_and_excludes_model_and_lora(tmp_path):
    run = tmp_path / "run"
    _minimal_run(run)
    report = create_run_archive(run, "run_001")
    assert report.sha256 == sha256_file(report.path)
    with zipfile.ZipFile(report.path) as archive:
        names = set(archive.namelist())
    assert "raw_frames/frame_000.png" in names
    assert "checkpoints/source/tensors.safetensors" in names
    assert not any(".pending-" in name for name in names)
    assert not any("hf_cache" in name for name in names)
    assert "inputs/adapter.safetensors" not in names
    assert not any(name.endswith(".flowmorph-klein.zip") for name in names)


def test_archive_includes_experimental_conditioning_comparison(tmp_path):
    run = tmp_path / "run"
    _minimal_run(run)
    comparison = run / "conditioning_comparison"
    (comparison / "source_conditioning_frames").mkdir(parents=True)
    (comparison / "comparison.json").write_text("{}")
    (comparison / "source_conditioning_frames/frame_000.png").write_bytes(b"frame")

    report = create_run_archive(run, "run_001")
    with zipfile.ZipFile(report.path) as archive:
        names = set(archive.namelist())

    assert "conditioning_comparison/comparison.json" in names
    assert (
        "conditioning_comparison/source_conditioning_frames/frame_000.png" in names
    )


def test_archive_rejects_token_in_text_artifact(tmp_path):
    run = tmp_path / "run"
    _minimal_run(run)
    (run / "execution.log").write_text("accident hf_" + "A" * 32)
    with pytest.raises(PackagingError, match="token"):
        create_run_archive(run, "run_001")
