"""Conservative anchor-aware border flicker correction for cyclic PNG sequences."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class BorderStabilizationConfig:
    """Settings for low-frequency color correction near the image margins only."""

    border_width_fraction: float = 0.025
    feather_fraction: float = 0.040
    strength: float = 0.65
    max_rgb_shift: float = 0.025

    def validate(self, frame_count: int) -> None:
        if frame_count < 3:
            raise ValueError("border stabilization requires at least three frames")
        if not 0.0 < self.border_width_fraction <= 0.25:
            raise ValueError("border_width_fraction must lie in (0, 0.25]")
        if not 0.0 <= self.feather_fraction <= 0.25:
            raise ValueError("feather_fraction must lie in [0, 0.25]")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must lie in [0, 1]")
        if not 0.0 <= self.max_rgb_shift <= 0.25:
            raise ValueError("max_rgb_shift must lie in [0, 0.25]")


@dataclass(frozen=True, slots=True)
class BorderStabilizationResult:
    """Corrected paths and an audit of the border-only operation."""

    output_paths: tuple[Path, ...]
    report: dict[str, Any]
    report_path: Path


def _validate_anchors(anchor_indices: Sequence[int], frame_count: int) -> tuple[int, ...]:
    anchors = tuple(sorted(int(index) for index in anchor_indices))
    if len(anchors) < 2:
        raise ValueError("border stabilization requires at least two anchor indices")
    if len(set(anchors)) != len(anchors):
        raise ValueError("border anchor indices must be unique")
    if anchors[0] < 0 or anchors[-1] >= frame_count:
        raise ValueError("border anchor index lies outside the sequence")
    return anchors


def _border_geometry(
    width: int,
    height: int,
    config: BorderStabilizationConfig,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    short_edge = min(width, height)
    border_width = max(1, round(short_edge * config.border_width_fraction))
    feather_width = max(0, round(short_edge * config.feather_fraction))
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    distance = np.minimum.reduce(
        (
            np.broadcast_to(rows, (height, width)),
            np.broadcast_to(columns, (height, width)),
            np.broadcast_to(height - 1 - rows, (height, width)),
            np.broadcast_to(width - 1 - columns, (height, width)),
        )
    )
    sample_mask = distance < border_width
    if feather_width == 0:
        correction_mask = sample_mask.astype(np.float32)
    else:
        progress = np.clip((distance - border_width) / feather_width, 0.0, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        correction_mask = (1.0 - smooth).astype(np.float32)
        correction_mask[distance >= border_width + feather_width] = 0.0
    return sample_mask, correction_mask, border_width, feather_width


def _piecewise_cyclic_targets(
    source: np.ndarray,
    anchors: Sequence[int],
) -> np.ndarray:
    frame_count = len(source)
    target = np.empty_like(source, dtype=np.float64)
    for position, left_index in enumerate(anchors):
        right_index = anchors[(position + 1) % len(anchors)]
        unwrapped_right = right_index if right_index > left_index else right_index + frame_count
        span = unwrapped_right - left_index
        for offset in range(span + 1):
            index = (left_index + offset) % frame_count
            progress = offset / span
            target[index] = (
                (1.0 - progress) * source[left_index] + progress * source[right_index]
            )
    return target


def _border_median(rgb: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    return np.median(rgb[sample_mask], axis=0).astype(np.float64)


def stabilize_cyclic_borders(
    frame_paths: Sequence[str | Path],
    output_directory: str | Path,
    *,
    anchor_indices: Sequence[int],
    config: BorderStabilizationConfig | None = None,
) -> BorderStabilizationResult:
    """Smooth border color drift while preserving pixels at every anchor frame."""

    paths = tuple(Path(path) for path in frame_paths)
    settings = config or BorderStabilizationConfig()
    settings.validate(len(paths))
    anchors = _validate_anchors(anchor_indices, len(paths))
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"border stabilization input is missing: {missing[:3]}")

    with Image.open(paths[0]) as opened:
        width, height = opened.size
    sample_mask, correction_mask, border_width, feather_width = _border_geometry(
        width, height, settings
    )

    source_statistics = []
    for path in paths:
        with Image.open(path) as opened:
            if opened.size != (width, height):
                raise ValueError(f"mismatched frame size at {path}: {opened.size}")
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        source_statistics.append(_border_median(rgb, sample_mask))
    source = np.asarray(source_statistics, dtype=np.float64)
    target = _piecewise_cyclic_targets(source, anchors)
    shifts = settings.strength * (target - source)
    shifts = np.clip(shifts, -settings.max_rgb_shift, settings.max_rgb_shift)
    shifts[list(anchors)] = 0.0

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    output_paths = []
    output_statistics = []
    changed_indices = []
    alpha = correction_mask[..., None]
    for index, (path, shift) in enumerate(zip(paths, shifts, strict=True)):
        output_path = destination / f"{index:07d}.png"
        if np.max(np.abs(shift)) < 0.5 / 255.0:
            with Image.open(path) as opened:
                rgb_u8 = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            shutil.copy2(path, output_path)
        else:
            with Image.open(path) as opened:
                rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / np.float32(255.0)
            corrected = np.clip(rgb + alpha * shift.astype(np.float32), 0.0, 1.0)
            rgb_u8 = np.rint(corrected * 255.0).astype(np.uint8)
            Image.fromarray(rgb_u8, mode="RGB").save(output_path, compress_level=4)
            changed_indices.append(index)
        output_paths.append(output_path)
        output_statistics.append(
            _border_median(rgb_u8.astype(np.float32) / np.float32(255.0), sample_mask)
        )

    output = np.asarray(output_statistics, dtype=np.float64)
    report = {
        "method": "anchor-interpolated cyclic border RGB trajectory with feathered spatial mask",
        "config": asdict(settings),
        "frame_count": len(paths),
        "width": width,
        "height": height,
        "border_width_pixels": border_width,
        "feather_width_pixels": feather_width,
        "anchor_indices": list(anchors),
        "anchor_pixels_unchanged": True,
        "center_pixels_unchanged": True,
        "changed_count": len(changed_indices),
        "changed_indices": changed_indices,
        "maximum_applied_rgb_shift": float(np.max(np.abs(shifts))),
        "source_border_rgb": source.tolist(),
        "target_border_rgb": target.tolist(),
        "output_border_rgb": output.tolist(),
        "source_target_mae": float(np.mean(np.abs(source - target))),
        "output_target_mae": float(np.mean(np.abs(output - target))),
        "raw_sources_preserved": True,
    }
    report_path = destination / "border_stabilization_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return BorderStabilizationResult(tuple(output_paths), report, report_path)
