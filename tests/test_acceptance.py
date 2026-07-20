from __future__ import annotations

import json

from flowmorph_klein import MODEL_ID
from flowmorph_klein.acceptance import audit_completed_run


def test_completed_run_audit_requires_exact_contract(tmp_path):
    for path in ("checkpoints/source", "checkpoints/target", "raw_frames", "display_frames"):
        (tmp_path / path).mkdir(parents=True)
    (tmp_path / "checkpoints/source/tensors.safetensors").write_bytes(b"source")
    (tmp_path / "checkpoints/target/tensors.safetensors").write_bytes(b"target")
    for index in range(20):
        (tmp_path / f"raw_frames/frame_{index:03d}.png").write_bytes(b"raw")
        (tmp_path / f"display_frames/frame_{index:03d}.png").write_bytes(b"display")
    (tmp_path / "metrics.json").write_text("{}")
    (tmp_path / "schedule.json").write_text("{}")
    (tmp_path / "environment.json").write_text("{}")
    manifest = {
        "model_id": MODEL_ID,
        "allow_degraded_run": False,
        "backward_probe_status": "passed",
        "source_completed_steps": 100,
        "target_completed_steps": 100,
        "lora_status": "verified",
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    report = audit_completed_run(tmp_path, require_lora=True)
    assert report.passed


def test_audit_does_not_accept_inference_only_or_short_fit(tmp_path):
    (tmp_path / "run_manifest.json").write_text("{}")
    report = audit_completed_run(tmp_path)
    assert not report.passed
    assert "backward_probe" in report.failures
    assert "source_100_steps" in report.failures


def test_audit_rejects_non_finite_metric_evidence(tmp_path):
    (tmp_path / "run_manifest.json").write_text("{}")
    (tmp_path / "metrics.json").write_text(
        json.dumps({"transition": {"adjacent_lpips_mean": "NaN"}})
    )
    report = audit_completed_run(tmp_path)
    assert "metrics" in report.failures


def test_interpolated_conditioning_requires_saved_source_comparison(tmp_path):
    manifest = {
        "model_id": MODEL_ID,
        "allow_degraded_run": False,
        "backward_probe_status": "passed",
        "source_completed_steps": 1,
        "target_completed_steps": 1,
    }
    for path in (
        "checkpoints/source",
        "checkpoints/target",
        "raw_frames",
        "display_frames",
        "conditioning_comparison/source_conditioning_frames",
    ):
        (tmp_path / path).mkdir(parents=True)
    (tmp_path / "checkpoints/source/tensors.safetensors").write_bytes(b"source")
    (tmp_path / "checkpoints/target/tensors.safetensors").write_bytes(b"target")
    for directory in (
        tmp_path / "raw_frames",
        tmp_path / "display_frames",
        tmp_path / "conditioning_comparison/source_conditioning_frames",
    ):
        for index in range(3):
            (directory / f"frame_{index:03d}.png").write_bytes(b"frame")
    for name in ("metrics.json", "schedule.json", "environment.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "conditioning_comparison/comparison.json").write_text("{}")
    (tmp_path / "conditioning_comparison/interpolated_vs_source.png").write_bytes(
        b"sheet"
    )

    report = audit_completed_run(
        tmp_path,
        manifest,
        expected_frames=3,
        expected_source_steps=1,
        expected_target_steps=1,
        require_conditioning_comparison=True,
    )
    assert report.passed

    (tmp_path / "conditioning_comparison/interpolated_vs_source.png").unlink()
    report = audit_completed_run(
        tmp_path,
        manifest,
        expected_frames=3,
        expected_source_steps=1,
        expected_target_steps=1,
        require_conditioning_comparison=True,
    )
    assert "conditioning_comparison" in report.failures
