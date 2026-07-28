"""Conservative temporal tone stabilization for cyclic image sequences."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class TemporalToneConfig:
    """Settings for local cyclic tone-outlier detection and correction."""

    window_radius: int = 2
    strength: float = 0.7
    mean_threshold: float = 0.02
    contrast_threshold: float = 0.10
    mad_multiplier: float = 3.5
    max_mean_shift: float = 0.06
    max_contrast_scale_delta: float = 0.15
    analysis_max_side: int = 256

    def validate(self, frame_count: int) -> None:
        if frame_count < 3:
            raise ValueError("temporal tone stabilization requires at least three frames")
        if not 1 <= self.window_radius <= max(1, (frame_count - 1) // 2):
            raise ValueError(
                "window_radius must lie between 1 and half the cyclic sequence length"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must lie in [0, 1]")
        if self.mean_threshold < 0.0 or self.contrast_threshold < 0.0:
            raise ValueError("tone thresholds cannot be negative")
        if self.mad_multiplier < 0.0:
            raise ValueError("mad_multiplier cannot be negative")
        if not 0.0 <= self.max_mean_shift <= 1.0:
            raise ValueError("max_mean_shift must lie in [0, 1]")
        if not 0.0 <= self.max_contrast_scale_delta < 1.0:
            raise ValueError("max_contrast_scale_delta must lie in [0, 1)")
        if self.analysis_max_side < 32:
            raise ValueError("analysis_max_side must be at least 32")


@dataclass(frozen=True, slots=True)
class TemporalToneResult:
    """Paths and audit data produced by temporal tone stabilization."""

    output_paths: tuple[Path, ...]
    report: dict[str, Any]
    report_path: Path
    cache_hit: bool


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * np.float32(0.2126)
        + rgb[..., 1] * np.float32(0.7152)
        + rgb[..., 2] * np.float32(0.0722)
    )


def _tone_statistics_array(rgb: np.ndarray) -> tuple[float, float]:
    luminance = _luminance(rgb)
    return float(luminance.mean()), float(luminance.std())


def image_tone_statistics(path: str | Path, *, max_side: int = 256) -> tuple[float, float]:
    """Measure gamma-space luminance mean and RMS contrast on a thumbnail."""

    with Image.open(path) as opened:
        sample = opened.convert("RGB")
        sample.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        rgb = np.asarray(sample, dtype=np.float32) / np.float32(255.0)
    return _tone_statistics_array(rgb)


def _cyclic_neighbor_median(values: np.ndarray, radius: int) -> np.ndarray:
    neighbors = [
        np.roll(values, offset)
        for offset in range(-radius, radius + 1)
        if offset != 0
    ]
    return np.median(np.stack(neighbors, axis=0), axis=0)


def _robust_limit(
    residuals: np.ndarray,
    absolute_floor: float,
    multiplier: float,
) -> tuple[float, float]:
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    robust_sigma = 1.4826 * mad
    return center, max(float(absolute_floor), float(multiplier) * robust_sigma)


def _source_contract(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _fingerprint(paths: Sequence[Path], config: TemporalToneConfig) -> tuple[str, dict[str, Any]]:
    contract = {
        "algorithm": "cyclic_neighbor_median_luminance_affine_v1",
        "config": asdict(config),
        "sources": [_source_contract(path) for path in paths],
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), contract


def _correct_tone(
    source_path: Path,
    output_path: Path,
    *,
    target_mean: float,
    target_contrast: float,
    config: TemporalToneConfig,
) -> dict[str, float]:
    with Image.open(source_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / np.float32(255.0)

    luminance = _luminance(rgb)
    source_mean = float(luminance.mean())
    source_contrast = float(luminance.std())
    mean_shift = float(
        np.clip(
            target_mean - source_mean,
            -config.max_mean_shift,
            config.max_mean_shift,
        )
    )
    if source_contrast <= 1e-6:
        contrast_scale = 1.0
    else:
        contrast_scale = float(target_contrast / source_contrast)
    contrast_scale = float(
        np.clip(
            contrast_scale,
            1.0 - config.max_contrast_scale_delta,
            1.0 + config.max_contrast_scale_delta,
        )
    )

    corrected_luminance = (
        (luminance - np.float32(source_mean)) * np.float32(contrast_scale)
        + np.float32(source_mean + mean_shift)
    )
    corrected_luminance = (
        luminance
        + np.float32(config.strength) * (corrected_luminance - luminance)
    )
    # Adding only the luminance delta retains the original RGB channel
    # differences except where gamut clipping is unavoidable.
    corrected_rgb = np.clip(
        rgb + (corrected_luminance - luminance)[..., None],
        0.0,
        1.0,
    )
    output = Image.fromarray(
        np.rint(corrected_rgb * np.float32(255.0)).astype(np.uint8)
    )
    output.save(output_path, format="PNG", compress_level=4)
    output.close()

    after_mean, after_contrast = image_tone_statistics(
        output_path,
        max_side=config.analysis_max_side,
    )
    return {
        "full_resolution_source_mean": source_mean,
        "full_resolution_source_contrast": source_contrast,
        "requested_mean_shift": mean_shift,
        "requested_contrast_scale": contrast_scale,
        "applied_strength": config.strength,
        "output_mean": after_mean,
        "output_contrast": after_contrast,
    }


def stabilize_cyclic_tone(
    image_paths: Sequence[str | Path],
    output_directory: str | Path,
    *,
    config: TemporalToneConfig | None = None,
    report_path: str | Path | None = None,
    reuse_existing: bool = True,
) -> TemporalToneResult:
    """Correct isolated cyclic luminance/contrast excursions without replacing raw files.

    The detector compares every frame with the median tone of its cyclic
    neighbors. Only robust outliers are corrected. Correction is an explicitly
    capped affine change to luminance; spatial structure and RGB differences are
    retained, and all source files remain untouched.
    """

    paths = tuple(Path(path) for path in image_paths)
    settings = config or TemporalToneConfig()
    settings.validate(len(paths))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"temporal tone source frame does not exist: {missing[0]}")

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = (
        Path(report_path)
        if report_path is not None
        else output_root / "temporal_tone_report.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint, contract = _fingerprint(paths, settings)

    if reuse_existing and audit_path.is_file():
        cached = json.loads(audit_path.read_text(encoding="utf-8"))
        cached_paths = tuple(Path(path) for path in cached.get("output_paths", ()))
        if (
            cached.get("fingerprint") == fingerprint
            and len(cached_paths) == len(paths)
            and all(path.is_file() for path in cached_paths)
        ):
            return TemporalToneResult(
                output_paths=cached_paths,
                report={**cached, "cache_hit": True},
                report_path=audit_path,
                cache_hit=True,
            )

    statistics = np.asarray(
        [
            image_tone_statistics(path, max_side=settings.analysis_max_side)
            for path in paths
        ],
        dtype=np.float64,
    )
    means = statistics[:, 0]
    contrasts = np.maximum(statistics[:, 1], 1e-6)
    target_means = _cyclic_neighbor_median(means, settings.window_radius)
    log_contrasts = np.log(contrasts)
    target_log_contrasts = _cyclic_neighbor_median(
        log_contrasts,
        settings.window_radius,
    )
    mean_residuals = means - target_means
    contrast_residuals = log_contrasts - target_log_contrasts
    mean_center, mean_limit = _robust_limit(
        mean_residuals,
        settings.mean_threshold,
        settings.mad_multiplier,
    )
    contrast_center, contrast_limit = _robust_limit(
        contrast_residuals,
        settings.contrast_threshold,
        settings.mad_multiplier,
    )
    mean_outliers = np.abs(mean_residuals - mean_center) > mean_limit
    contrast_outliers = np.abs(contrast_residuals - contrast_center) > contrast_limit
    corrected_flags = mean_outliers | contrast_outliers

    output_paths: list[Path] = []
    frame_reports: list[dict[str, Any]] = []
    for index, source_path in enumerate(paths):
        corrected = bool(corrected_flags[index])
        correction: dict[str, float] | None
        if corrected and settings.strength > 0.0:
            output_path = output_root / f"{index:07d}_{source_path.name}"
            correction = _correct_tone(
                source_path,
                output_path,
                target_mean=float(target_means[index]),
                target_contrast=float(math.exp(target_log_contrasts[index])),
                config=settings,
            )
        else:
            # Reuse the untouched raw frame instead of duplicating a potentially
            # large PNG on persistent storage.
            output_path = source_path
            correction = None
        output_paths.append(output_path)
        frame_reports.append(
            {
                "index": index,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "corrected": corrected and settings.strength > 0.0,
                "mean_outlier": bool(mean_outliers[index]),
                "contrast_outlier": bool(contrast_outliers[index]),
                "source_mean": float(means[index]),
                "source_contrast": float(contrasts[index]),
                "neighbor_target_mean": float(target_means[index]),
                "neighbor_target_contrast": float(
                    math.exp(target_log_contrasts[index])
                ),
                "mean_residual": float(mean_residuals[index]),
                "log_contrast_residual": float(contrast_residuals[index]),
                "correction": correction,
            }
        )

    report = {
        "fingerprint": fingerprint,
        "contract": contract,
        "cyclic": True,
        "raw_sources_preserved": True,
        "frame_count": len(paths),
        "corrected_count": int(sum(frame["corrected"] for frame in frame_reports)),
        "corrected_indices": [
            frame["index"] for frame in frame_reports if frame["corrected"]
        ],
        "detector": {
            "mean_residual_center": mean_center,
            "mean_outlier_limit": mean_limit,
            "log_contrast_residual_center": contrast_center,
            "log_contrast_outlier_limit": contrast_limit,
        },
        "frames": frame_reports,
        "output_paths": [str(path) for path in output_paths],
        "cache_hit": False,
    }
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return TemporalToneResult(
        output_paths=tuple(output_paths),
        report=report,
        report_path=audit_path,
        cache_hit=False,
    )


__all__ = [
    "TemporalToneConfig",
    "TemporalToneResult",
    "image_tone_statistics",
    "stabilize_cyclic_tone",
]
