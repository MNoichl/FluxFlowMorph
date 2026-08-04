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

    luminance_enabled: bool = True
    window_radius: int = 2
    strength: float = 0.7
    mean_threshold: float = 0.02
    contrast_threshold: float = 0.10
    mad_multiplier: float = 3.5
    max_mean_shift: float = 0.06
    max_contrast_scale_delta: float = 0.15
    analysis_max_side: int = 256
    chroma_enabled: bool = False
    chroma_strength: float = 0.5
    chroma_threshold: float = 0.01
    max_chroma_gain: float = 0.08
    chroma_smoothing_passes: int = 4

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
        if not 0.0 <= self.chroma_strength <= 1.0:
            raise ValueError("chroma_strength must lie in [0, 1]")
        if not 0.0 <= self.chroma_threshold < 1.0:
            raise ValueError("chroma_threshold must lie in [0, 1)")
        if not 0.0 <= self.max_chroma_gain < 1.0:
            raise ValueError("max_chroma_gain must lie in [0, 1)")
        if self.chroma_smoothing_passes < 0:
            raise ValueError("chroma_smoothing_passes cannot be negative")


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


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= np.float32(0.04045),
        rgb / np.float32(12.92),
        ((rgb + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    positive = np.maximum(rgb, np.float32(0.0))
    return np.where(
        positive <= np.float32(0.0031308),
        positive * np.float32(12.92),
        np.float32(1.055) * positive ** np.float32(1.0 / 2.4) - np.float32(0.055),
    )


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB to OKLab using Björn Ottosson's matrices."""

    linear = _srgb_to_linear(rgb.astype(np.float32, copy=False))
    red, green, blue = np.moveaxis(linear, -1, 0)
    light = (
        np.float32(0.4122214708) * red
        + np.float32(0.5363325363) * green
        + np.float32(0.0514459929) * blue
    )
    medium = (
        np.float32(0.2119034982) * red
        + np.float32(0.6806995451) * green
        + np.float32(0.1073969566) * blue
    )
    short = (
        np.float32(0.0883024619) * red
        + np.float32(0.2817188376) * green
        + np.float32(0.6299787005) * blue
    )
    light_root = np.cbrt(light)
    medium_root = np.cbrt(medium)
    short_root = np.cbrt(short)
    return np.stack(
        (
            np.float32(0.2104542553) * light_root
            + np.float32(0.7936177850) * medium_root
            - np.float32(0.0040720468) * short_root,
            np.float32(1.9779984951) * light_root
            - np.float32(2.4285922050) * medium_root
            + np.float32(0.4505937099) * short_root,
            np.float32(0.0259040371) * light_root
            + np.float32(0.7827717662) * medium_root
            - np.float32(0.8086757660) * short_root,
        ),
        axis=-1,
    )


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lightness, opponent_a, opponent_b = np.moveaxis(lab, -1, 0)
    light_root = (
        lightness
        + np.float32(0.3963377774) * opponent_a
        + np.float32(0.2158037573) * opponent_b
    )
    medium_root = (
        lightness
        - np.float32(0.1055613458) * opponent_a
        - np.float32(0.0638541728) * opponent_b
    )
    short_root = (
        lightness
        - np.float32(0.0894841775) * opponent_a
        - np.float32(1.2914855480) * opponent_b
    )
    light = light_root**3
    medium = medium_root**3
    short = short_root**3
    linear = np.stack(
        (
            np.float32(4.0767416621) * light
            - np.float32(3.3077115913) * medium
            + np.float32(0.2309699292) * short,
            -np.float32(1.2684380046) * light
            + np.float32(2.6097574011) * medium
            - np.float32(0.3413193965) * short,
            -np.float32(0.0041960863) * light
            - np.float32(0.7034186147) * medium
            + np.float32(1.7076147010) * short,
        ),
        axis=-1,
    )
    return np.clip(_linear_to_srgb(linear), 0.0, 1.0)


def _chroma_statistics_array(rgb: np.ndarray) -> float:
    lab = _srgb_to_oklab(rgb)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    return float(chroma.mean())


def image_tone_statistics(path: str | Path, *, max_side: int = 256) -> tuple[float, float]:
    """Measure gamma-space luminance mean and RMS contrast on a thumbnail."""

    with Image.open(path) as opened:
        sample = opened.convert("RGB")
        sample.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        rgb = np.asarray(sample, dtype=np.float32) / np.float32(255.0)
    return _tone_statistics_array(rgb)


def image_chroma_statistics(path: str | Path, *, max_side: int = 256) -> float:
    """Measure mean OKLab chroma on a thumbnail."""

    with Image.open(path) as opened:
        sample = opened.convert("RGB")
        sample.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        rgb = np.asarray(sample, dtype=np.float32) / np.float32(255.0)
    return _chroma_statistics_array(rgb)


def _cyclic_neighbor_median(values: np.ndarray, radius: int) -> np.ndarray:
    neighbors = [
        np.roll(values, offset)
        for offset in range(-radius, radius + 1)
        if offset != 0
    ]
    return np.median(np.stack(neighbors, axis=0), axis=0)


def _validate_chroma_anchor_indices(
    anchor_indices: Sequence[int] | None,
    frame_count: int,
) -> tuple[int, ...]:
    if anchor_indices is None:
        raise ValueError("chroma correction requires explicit endpoint anchor indices")
    anchors = tuple(sorted(int(index) for index in anchor_indices))
    if len(anchors) < 2:
        raise ValueError("chroma correction requires at least two endpoint anchors")
    if len(set(anchors)) != len(anchors):
        raise ValueError("chroma endpoint anchor indices must be unique")
    if anchors[0] < 0 or anchors[-1] >= frame_count:
        raise ValueError("chroma endpoint anchor index lies outside the sequence")
    return anchors


def _chroma_correction_trajectory(
    chromas: np.ndarray,
    anchor_indices: Sequence[int],
    config: TemporalToneConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a cyclic endpoint-anchored target and smoothly faded gain curve."""

    frame_count = len(chromas)
    anchors = tuple(anchor_indices)
    targets = np.empty(frame_count, dtype=np.float64)
    envelopes = np.zeros(frame_count, dtype=np.float64)
    segments: list[list[int]] = []
    for anchor_position, left_index in enumerate(anchors):
        right_index = anchors[(anchor_position + 1) % len(anchors)]
        unwrapped_right = right_index if right_index > left_index else right_index + frame_count
        span = unwrapped_right - left_index
        segment: list[int] = []
        for offset in range(span + 1):
            index = (left_index + offset) % frame_count
            progress = offset / span
            targets[index] = (
                (1.0 - progress) * chromas[left_index]
                + progress * chromas[right_index]
            )
            envelopes[index] = math.sin(math.pi * progress) ** 2
            segment.append(index)
        segments.append(segment)

    denominators = np.maximum(chromas, np.finfo(np.float64).eps)
    relative_deficits = np.maximum(targets / denominators - 1.0, 0.0)
    effective_deficits = np.maximum(relative_deficits - config.chroma_threshold, 0.0)
    raw_gains = np.minimum(
        config.max_chroma_gain,
        config.chroma_strength * effective_deficits * envelopes,
    )
    gains = np.zeros(frame_count, dtype=np.float64)
    for segment in segments:
        segment_gains = raw_gains[segment].copy()
        segment_gains[0] = 0.0
        segment_gains[-1] = 0.0
        for _ in range(config.chroma_smoothing_passes):
            smoothed = segment_gains.copy()
            if len(segment_gains) > 2:
                smoothed[1:-1] = (
                    np.float64(0.25) * segment_gains[:-2]
                    + np.float64(0.50) * segment_gains[1:-1]
                    + np.float64(0.25) * segment_gains[2:]
                )
            smoothed[0] = 0.0
            smoothed[-1] = 0.0
            # Smoothing may reduce or gently spread a correction, but it must
            # never exceed the deficit-driven gain allowed for that frame.
            segment_gains = np.minimum(smoothed, raw_gains[segment])
        for offset, index in enumerate(segment):
            if offset not in {0, len(segment) - 1}:
                gains[index] = min(
                    config.max_chroma_gain,
                    float(raw_gains[index]),
                    float(segment_gains[offset]),
                )
    return targets, envelopes, gains


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


def _fingerprint(
    paths: Sequence[Path],
    config: TemporalToneConfig,
    *,
    chroma_anchor_indices: Sequence[int] | None,
) -> tuple[str, dict[str, Any]]:
    contract = {
        "algorithm": "cyclic_luminance_affine_and_endpoint_oklab_chroma_v2",
        "config": asdict(config),
        "chroma_anchor_indices": (
            list(chroma_anchor_indices) if chroma_anchor_indices is not None else None
        ),
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
    apply_luminance: bool,
    chroma_gain: float,
    config: TemporalToneConfig,
) -> dict[str, float]:
    with Image.open(source_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / np.float32(255.0)

    luminance = _luminance(rgb)
    source_mean = float(luminance.mean())
    source_contrast = float(luminance.std())
    mean_shift = (
        float(
            np.clip(
                target_mean - source_mean,
                -config.max_mean_shift,
                config.max_mean_shift,
            )
        )
        if apply_luminance
        else 0.0
    )
    if not apply_luminance or source_contrast <= 1e-6:
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

    if apply_luminance:
        corrected_luminance = (
            (luminance - np.float32(source_mean)) * np.float32(contrast_scale)
            + np.float32(source_mean + mean_shift)
        )
        corrected_luminance = (
            luminance
            + np.float32(config.strength) * (corrected_luminance - luminance)
        )
    else:
        corrected_luminance = luminance
    # Adding only the luminance delta retains the original RGB channel
    # differences except where gamut clipping is unavoidable.
    corrected_rgb = np.clip(
        rgb + (corrected_luminance - luminance)[..., None],
        0.0,
        1.0,
    )
    if chroma_gain > 0.0:
        corrected_lab = _srgb_to_oklab(corrected_rgb)
        corrected_lab[..., 1:] *= np.float32(1.0 + chroma_gain)
        corrected_rgb = _oklab_to_srgb(corrected_lab)
    output = Image.fromarray(
        np.rint(corrected_rgb * np.float32(255.0)).astype(np.uint8)
    )
    output.save(output_path, format="PNG", compress_level=4)
    output.close()

    after_mean, after_contrast = image_tone_statistics(
        output_path,
        max_side=config.analysis_max_side,
    )
    after_chroma = image_chroma_statistics(
        output_path,
        max_side=config.analysis_max_side,
    )
    return {
        "full_resolution_source_mean": source_mean,
        "full_resolution_source_contrast": source_contrast,
        "requested_mean_shift": mean_shift,
        "requested_contrast_scale": contrast_scale,
        "applied_luminance_strength": config.strength if apply_luminance else 0.0,
        "requested_chroma_gain": chroma_gain,
        "output_mean": after_mean,
        "output_contrast": after_contrast,
        "output_chroma": after_chroma,
    }


def stabilize_cyclic_tone(
    image_paths: Sequence[str | Path],
    output_directory: str | Path,
    *,
    config: TemporalToneConfig | None = None,
    chroma_anchor_indices: Sequence[int] | None = None,
    report_path: str | Path | None = None,
    reuse_existing: bool = True,
) -> TemporalToneResult:
    """Correct cyclic tone/chroma excursions without replacing raw files.

    The detector compares every frame with the median tone of its cyclic
    neighbors. Only robust outliers are corrected. Correction is an explicitly
    capped affine change to luminance. Optional chroma correction follows a
    smooth endpoint-anchored OKLab trajectory, only lifts deficits, and fades to
    exactly zero at every endpoint. All source files remain untouched.
    """

    paths = tuple(Path(path) for path in image_paths)
    settings = config or TemporalToneConfig()
    settings.validate(len(paths))
    anchors = (
        _validate_chroma_anchor_indices(chroma_anchor_indices, len(paths))
        if settings.chroma_enabled
        else None
    )
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
    fingerprint, contract = _fingerprint(
        paths,
        settings,
        chroma_anchor_indices=anchors,
    )

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
    chromas = np.asarray(
        [
            image_chroma_statistics(path, max_side=settings.analysis_max_side)
            for path in paths
        ],
        dtype=np.float64,
    )
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
    luminance_corrected_flags = (
        mean_outliers | contrast_outliers
        if settings.luminance_enabled and settings.strength > 0.0
        else np.zeros(len(paths), dtype=bool)
    )
    if settings.chroma_enabled:
        assert anchors is not None
        chroma_targets, chroma_envelopes, chroma_gains = _chroma_correction_trajectory(
            chromas,
            anchors,
            settings,
        )
    else:
        chroma_targets = chromas.copy()
        chroma_envelopes = np.zeros(len(paths), dtype=np.float64)
        chroma_gains = np.zeros(len(paths), dtype=np.float64)
    chroma_corrected_flags = chroma_gains > np.finfo(np.float64).eps
    corrected_flags = luminance_corrected_flags | chroma_corrected_flags

    output_paths: list[Path] = []
    frame_reports: list[dict[str, Any]] = []
    for index, source_path in enumerate(paths):
        corrected = bool(corrected_flags[index])
        luminance_corrected = bool(luminance_corrected_flags[index])
        chroma_corrected = bool(chroma_corrected_flags[index])
        correction: dict[str, float] | None
        if corrected:
            output_path = output_root / f"{index:07d}_{source_path.name}"
            correction = _correct_tone(
                source_path,
                output_path,
                target_mean=float(target_means[index]),
                target_contrast=float(math.exp(target_log_contrasts[index])),
                apply_luminance=luminance_corrected,
                chroma_gain=float(chroma_gains[index]),
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
                "corrected": corrected,
                "luminance_corrected": luminance_corrected,
                "chroma_corrected": chroma_corrected,
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
                "source_chroma": float(chromas[index]),
                "chroma_target": float(chroma_targets[index]),
                "chroma_envelope": float(chroma_envelopes[index]),
                "chroma_gain": float(chroma_gains[index]),
                "output_chroma": (
                    float(correction["output_chroma"])
                    if correction is not None
                    else float(chromas[index])
                ),
                "correction": correction,
            }
        )

    output_chromas = np.asarray(
        [float(frame["output_chroma"]) for frame in frame_reports],
        dtype=np.float64,
    )
    cyclic_gain_steps = chroma_gains - np.roll(chroma_gains, 1)
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
        "luminance_corrected_indices": [
            frame["index"] for frame in frame_reports if frame["luminance_corrected"]
        ],
        "chroma_corrected_indices": [
            frame["index"] for frame in frame_reports if frame["chroma_corrected"]
        ],
        "detector": {
            "mean_residual_center": mean_center,
            "mean_outlier_limit": mean_limit,
            "log_contrast_residual_center": contrast_center,
            "log_contrast_outlier_limit": contrast_limit,
        },
        "chroma_trajectory": {
            "color_space": "OKLab",
            "anchors": list(anchors) if anchors is not None else [],
            "source": [float(value) for value in chromas],
            "target": [float(value) for value in chroma_targets],
            "gain": [float(value) for value in chroma_gains],
            "output": [float(value) for value in output_chromas],
            "source_target_mae": float(np.mean(np.abs(chromas - chroma_targets))),
            "output_target_mae": float(np.mean(np.abs(output_chromas - chroma_targets))),
            "maximum_gain": float(np.max(chroma_gains)),
            "maximum_adjacent_gain_step": float(np.max(np.abs(cyclic_gain_steps))),
            "endpoint_gain_is_zero": bool(
                anchors is None
                or all(chroma_gains[index] == 0.0 for index in anchors)
            ),
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
    "image_chroma_statistics",
    "image_tone_statistics",
    "stabilize_cyclic_tone",
]
