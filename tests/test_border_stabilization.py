from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from flowmorph_klein.border_stabilization import (
    BorderStabilizationConfig,
    stabilize_cyclic_borders,
)


def _write_frame(path: Path, border_value: float, center_value: float) -> None:
    array = np.full((64, 64, 3), center_value, dtype=np.float32)
    array[:8] = border_value
    array[-8:] = border_value
    array[:, :8] = border_value
    array[:, -8:] = border_value
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB").save(path)


def test_border_stabilization_corrects_margin_without_touching_center_or_anchors(
    tmp_path: Path,
) -> None:
    paths = []
    for index, value in enumerate((0.30, 0.62, 0.34, 0.38, 0.42)):
        path = tmp_path / f"source_{index:02d}.png"
        _write_frame(path, value, 0.77)
        paths.append(path)

    result = stabilize_cyclic_borders(
        paths,
        tmp_path / "corrected",
        anchor_indices=[0, 4],
        config=BorderStabilizationConfig(
            border_width_fraction=0.125,
            feather_fraction=0.0625,
            strength=1.0,
            max_rgb_shift=0.25,
        ),
    )

    assert result.report["output_target_mae"] < result.report["source_target_mae"]
    assert result.output_paths[0].read_bytes() == paths[0].read_bytes()
    assert result.output_paths[4].read_bytes() == paths[4].read_bytes()
    with Image.open(paths[1]) as opened:
        before = np.asarray(opened.convert("RGB"))
    with Image.open(result.output_paths[1]) as opened:
        after = np.asarray(opened.convert("RGB"))
    assert np.array_equal(before[32, 32], after[32, 32])
    assert not np.array_equal(before[0, 0], after[0, 0])
    assert result.report["anchor_pixels_unchanged"] is True
    assert result.report["center_pixels_unchanged"] is True


def test_border_stabilization_preserves_source_files(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate((0.25, 0.55, 0.35, 0.45)):
        path = tmp_path / f"source_{index:02d}.png"
        _write_frame(path, value, 0.70)
        paths.append(path)
    original = [path.read_bytes() for path in paths]

    result = stabilize_cyclic_borders(
        paths,
        tmp_path / "corrected",
        anchor_indices=[0, 2],
    )

    assert [path.read_bytes() for path in paths] == original
    assert result.report_path.is_file()
    assert len(result.output_paths) == len(paths)
