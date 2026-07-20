"""Mechanical audit of a run before it is labeled or archived."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from . import MODEL_ID


class RunPhase(str, Enum):
    CREATED = "created"
    INPUTS_VALIDATED = "inputs_validated"
    MODEL_READY = "model_ready"
    ADAPTER_VERIFIED = "adapter_verified"
    BACKWARD_PREFLIGHT_PASSED = "backward_preflight_passed"
    SOURCE_CHECKPOINTED = "source_checkpointed"
    TARGET_CHECKPOINTED = "target_checkpointed"
    FRAMES_RENDERED = "frames_rendered"
    METRICS_COMPLETE = "metrics_complete"
    ARCHIVE_VALIDATED = "archive_validated"
    FAILED = "failed"


PHASE_ORDER = {
    phase: index
    for index, phase in enumerate(
        (
            RunPhase.CREATED,
            RunPhase.INPUTS_VALIDATED,
            RunPhase.MODEL_READY,
            RunPhase.ADAPTER_VERIFIED,
            RunPhase.BACKWARD_PREFLIGHT_PASSED,
            RunPhase.SOURCE_CHECKPOINTED,
            RunPhase.TARGET_CHECKPOINTED,
            RunPhase.FRAMES_RENDERED,
            RunPhase.METRICS_COMPLETE,
            RunPhase.ARCHIVE_VALIDATED,
        )
    )
}


class AcceptanceError(RuntimeError):
    """Raised when output is incomplete or contradicts the reference contract."""


@dataclass(frozen=True)
class AcceptanceReport:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks, "failures": list(self.failures)}


def validate_phase_transition(current: RunPhase, target: RunPhase) -> None:
    if target is RunPhase.FAILED:
        return
    if current is RunPhase.FAILED:
        raise AcceptanceError("A failed run cannot advance without an explicit new run")
    if PHASE_ORDER[target] != PHASE_ORDER[current] + 1:
        raise AcceptanceError(f"Invalid phase transition {current.value!r} -> {target.value!r}")


def _count_png(directory: Path) -> int:
    return len(list(directory.glob("frame_*.png"))) if directory.is_dir() else 0


def _metrics_payload_is_numerically_valid(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> bool:
    """Reject serialized NaN/Inf evidence except exact-image infinite PSNR."""

    if isinstance(value, Mapping):
        return all(
            _metrics_payload_is_numerically_valid(item, path=(*path, str(key)))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(
            _metrics_payload_is_numerically_valid(item, path=(*path, str(index)))
            for index, item in enumerate(value)
        )
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value) or (
            bool(path) and path[-1] == "psnr" and value == math.inf
        )
    if isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
        return bool(path) and path[-1] == "psnr" and value == "Infinity"
    return True


def _metrics_file_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and _metrics_payload_is_numerically_valid(
        payload
    )


def audit_completed_run(
    run_directory: str | Path,
    manifest: Mapping[str, Any] | None = None,
    *,
    expected_frames: int = 20,
    expected_source_steps: int = 100,
    expected_target_steps: int = 100,
    expected_model_id: str = MODEL_ID,
    require_lora: bool = False,
    require_conditioning_comparison: bool = False,
) -> AcceptanceReport:
    root = Path(run_directory)
    if manifest is None:
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise AcceptanceError("run_manifest.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    checks["exact_model"] = manifest.get("model_id") == expected_model_id
    checks["not_degraded"] = manifest.get("allow_degraded_run") is False
    checks["backward_probe"] = manifest.get("backward_probe_status") == "passed"
    # Stable check names are retained for existing report consumers; the
    # compared values are configurable so explicitly diagnostic runs can be
    # audited against their own declared step counts.
    checks["source_100_steps"] = (
        manifest.get("source_completed_steps") == expected_source_steps
    )
    checks["target_100_steps"] = (
        manifest.get("target_completed_steps") == expected_target_steps
    )
    checks["source_checkpoint"] = (root / "checkpoints/source/tensors.safetensors").is_file()
    checks["target_checkpoint"] = (root / "checkpoints/target/tensors.safetensors").is_file()
    checks["raw_frame_count"] = _count_png(root / "raw_frames") == expected_frames
    checks["display_frame_count"] = _count_png(root / "display_frames") == expected_frames
    checks["metrics"] = _metrics_file_is_valid(root / "metrics.json")
    checks["schedule"] = (root / "schedule.json").is_file()
    checks["environment"] = (root / "environment.json").is_file()
    checks["lora"] = (not require_lora) or manifest.get("lora_status") == "verified"
    checks["conditioning_comparison"] = (
        not require_conditioning_comparison
        or (
            _count_png(
                root / "conditioning_comparison" / "source_conditioning_frames"
            )
            == expected_frames
            and (root / "conditioning_comparison" / "comparison.json").is_file()
            and (
                root / "conditioning_comparison" / "interpolated_vs_source.png"
            ).is_file()
        )
    )
    failures = tuple(name for name, passed in checks.items() if not passed)
    return AcceptanceReport(passed=not failures, checks=checks, failures=failures)


def require_completed_run(*args: Any, **kwargs: Any) -> AcceptanceReport:
    report = audit_completed_run(*args, **kwargs)
    if not report.passed:
        raise AcceptanceError(f"Run acceptance failed: {', '.join(report.failures)}")
    return report
