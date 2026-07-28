from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from flowmorph_klein.temporal_tone import (
    TemporalToneConfig,
    image_tone_statistics,
    stabilize_cyclic_tone,
)


def _write_frame(path: Path, mean: float, contrast: float = 0.08) -> None:
    gradient = np.linspace(-1.0, 1.0, 96, dtype=np.float32)
    luminance = np.clip(mean + contrast * gradient[None, :], 0.0, 1.0)
    luminance = np.repeat(luminance, 64, axis=0)
    rgb = np.stack(
        [
            np.clip(luminance + 0.03, 0.0, 1.0),
            luminance,
            np.clip(luminance - 0.02, 0.0, 1.0),
        ],
        axis=-1,
    )
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)


def test_temporal_tone_corrects_isolated_cyclic_exposure_outlier(tmp_path: Path) -> None:
    paths = []
    for index, mean in enumerate((0.50, 0.51, 0.50, 0.68, 0.49, 0.50, 0.51)):
        path = tmp_path / f"frame_{index:02d}.png"
        _write_frame(path, mean)
        paths.append(path)

    result = stabilize_cyclic_tone(
        paths,
        tmp_path / "stabilized",
        config=TemporalToneConfig(
            mean_threshold=0.025,
            contrast_threshold=0.2,
            mad_multiplier=3.0,
            strength=0.8,
        ),
    )

    assert result.report["corrected_indices"] == [3]
    before_mean, _ = image_tone_statistics(paths[3])
    after_mean, _ = image_tone_statistics(result.output_paths[3])
    neighbor_target = result.report["frames"][3]["neighbor_target_mean"]
    assert abs(after_mean - neighbor_target) < abs(before_mean - neighbor_target)
    assert paths[3].read_bytes() != result.output_paths[3].read_bytes()
    assert result.output_paths[2] == paths[2]
    assert paths[2].read_bytes() == result.output_paths[2].read_bytes()


def test_temporal_tone_corrects_isolated_contrast_outlier(tmp_path: Path) -> None:
    paths = []
    for index, contrast in enumerate((0.08, 0.08, 0.08, 0.20, 0.08, 0.08, 0.08)):
        path = tmp_path / f"frame_{index:02d}.png"
        _write_frame(path, 0.5, contrast)
        paths.append(path)

    result = stabilize_cyclic_tone(
        paths,
        tmp_path / "stabilized",
        config=TemporalToneConfig(
            mean_threshold=0.2,
            contrast_threshold=0.10,
            mad_multiplier=3.0,
            strength=1.0,
            max_contrast_scale_delta=0.25,
        ),
    )

    assert result.report["corrected_indices"] == [3]
    _, before_contrast = image_tone_statistics(paths[3])
    _, after_contrast = image_tone_statistics(result.output_paths[3])
    target = result.report["frames"][3]["neighbor_target_contrast"]
    assert abs(after_contrast - target) < abs(before_contrast - target)


def test_temporal_tone_reuses_matching_audit_and_outputs(tmp_path: Path) -> None:
    paths = []
    for index, mean in enumerate((0.5, 0.5, 0.65, 0.5, 0.5)):
        path = tmp_path / f"frame_{index:02d}.png"
        _write_frame(path, mean)
        paths.append(path)
    output_directory = tmp_path / "stabilized"

    first = stabilize_cyclic_tone(paths, output_directory)
    second = stabilize_cyclic_tone(paths, output_directory)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.output_paths == second.output_paths
    saved = json.loads(second.report_path.read_text(encoding="utf-8"))
    assert saved["raw_sources_preserved"] is True


def test_temporal_tone_leaves_smooth_sequence_unchanged(tmp_path: Path) -> None:
    paths = []
    for index, mean in enumerate((0.49, 0.50, 0.51, 0.51, 0.50, 0.49)):
        path = tmp_path / f"frame_{index:02d}.png"
        _write_frame(path, mean)
        paths.append(path)

    result = stabilize_cyclic_tone(paths, tmp_path / "stabilized")

    assert result.report["corrected_count"] == 0
    assert all(
        source.read_bytes() == output.read_bytes()
        for source, output in zip(paths, result.output_paths, strict=True)
    )
