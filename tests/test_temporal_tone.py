from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flowmorph_klein.temporal_tone import (
    TemporalToneConfig,
    image_chroma_statistics,
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


def _write_chroma_frame(path: Path, amount: float) -> None:
    horizontal = np.linspace(-0.02, 0.02, 96, dtype=np.float32)[None, :, None]
    base = np.asarray(
        [0.50 + amount, 0.49 - 0.35 * amount, 0.47 - 0.25 * amount],
        dtype=np.float32,
    )[None, None, :]
    rgb = np.clip(base + horizontal, 0.0, 1.0)
    rgb = np.repeat(rgb, 64, axis=0)
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
        source.read_bytes() == output.read_bytes() for source, output in zip(paths, result.output_paths, strict=True)
    )


def test_temporal_chroma_lifts_midpoint_deficit_with_smooth_zero_endpoint_gain(
    tmp_path: Path,
) -> None:
    paths = []
    for index, amount in enumerate((0.16, 0.14, 0.10, 0.07, 0.10, 0.14, 0.16)):
        path = tmp_path / f"chroma_{index:02d}.png"
        _write_chroma_frame(path, amount)
        paths.append(path)

    result = stabilize_cyclic_tone(
        paths,
        tmp_path / "chroma_stabilized",
        config=TemporalToneConfig(
            luminance_enabled=False,
            chroma_enabled=True,
            chroma_strength=0.7,
            chroma_threshold=0.0,
            max_chroma_gain=None,
            max_chroma_decrease=None,
            chroma_smoothness=6.0,
        ),
        chroma_anchor_indices=[0, 6],
    )

    trajectory = result.report["chroma_trajectory"]
    gains = np.asarray(trajectory["gain"])
    source = np.asarray(trajectory["source"])
    target = np.asarray(trajectory["target"])
    output = np.asarray(trajectory["output"])
    assert gains[0] == 0.0
    assert gains[6] == 0.0
    assert gains == pytest.approx(gains[::-1])
    assert output[3] > source[3]
    assert np.mean(np.abs(output - target)) < np.mean(np.abs(source - target))
    assert trajectory["desired_output_mae"] < 0.001
    assert output == pytest.approx(trajectory["desired"], abs=0.0015)
    assert trajectory["desired_curvature_rms"] < trajectory["source_curvature_rms"]
    assert result.output_paths[0] == paths[0]
    assert result.output_paths[6] == paths[6]
    assert image_chroma_statistics(result.output_paths[3]) > image_chroma_statistics(paths[3])
    before_mean, _ = image_tone_statistics(paths[3])
    after_mean, _ = image_tone_statistics(result.output_paths[3])
    # This synthetic midpoint requires almost a 2x chroma scale and touches
    # the sRGB gamut boundary; lightness remains close despite gamut mapping.
    assert after_mean == pytest.approx(before_mean, abs=0.015)
    assert trajectory["endpoint_gain_is_zero"] is True


def test_temporal_chroma_smooths_output_with_signed_upward_and_downward_changes(
    tmp_path: Path,
) -> None:
    paths = []
    for index, amount in enumerate((0.08, 0.22, 0.18, 0.06, 0.05, 0.16, 0.20)):
        path = tmp_path / f"signed_chroma_{index:02d}.png"
        _write_chroma_frame(path, amount)
        paths.append(path)

    result = stabilize_cyclic_tone(
        paths,
        tmp_path / "signed_chroma_stabilized",
        config=TemporalToneConfig(
            luminance_enabled=False,
            chroma_enabled=True,
            chroma_strength=0.7,
            chroma_threshold=0.0,
            max_chroma_gain=None,
            max_chroma_decrease=None,
            chroma_smoothness=6.0,
        ),
        chroma_anchor_indices=[0, 6],
    )

    trajectory = result.report["chroma_trajectory"]
    gains = np.asarray(trajectory["gain"])
    source = np.asarray(trajectory["source"])
    target = np.asarray(trajectory["target"])
    output = np.asarray(trajectory["output"])
    assert np.any(gains < 0.0)
    assert np.any(gains > 0.0)
    assert gains[0] == 0.0
    assert gains[-1] == 0.0
    assert np.mean(np.abs(output - target)) < np.mean(np.abs(source - target))
    assert trajectory["desired_output_mae"] < 0.001
    assert output == pytest.approx(trajectory["desired"], abs=0.0015)
    assert trajectory["output_curvature_rms"] < trajectory["source_curvature_rms"]
    assert trajectory["minimum_gain"] < 0.0
    assert trajectory["maximum_gain"] > 0.0


def test_temporal_chroma_optional_emergency_limits_remain_available(
    tmp_path: Path,
) -> None:
    paths = []
    for index, amount in enumerate((0.18, 0.04, 0.02, 0.01, 0.02, 0.04, 0.18)):
        path = tmp_path / f"limited_chroma_{index:02d}.png"
        _write_chroma_frame(path, amount)
        paths.append(path)

    result = stabilize_cyclic_tone(
        paths,
        tmp_path / "limited_chroma_stabilized",
        config=TemporalToneConfig(
            luminance_enabled=False,
            chroma_enabled=True,
            chroma_strength=1.0,
            max_chroma_gain=0.15,
            max_chroma_decrease=0.10,
            chroma_smoothness=6.0,
        ),
        chroma_anchor_indices=[0, 6],
    )

    gains = np.asarray(result.report["chroma_trajectory"]["gain"])
    assert np.max(gains) <= 0.15
    assert np.min(gains) >= -0.10
