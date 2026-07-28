from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from flowmorph_klein.flicker_diagnostics import (
    FlickerDiagnosticConfig,
    diagnose_cyclic_flicker,
    format_flicker_diagnostic_markdown,
)


def _write_frame(path: Path, mean: float, contrast: float = 0.08) -> None:
    gradient = np.linspace(-1.0, 1.0, 96, dtype=np.float32)
    luminance = np.clip(mean + contrast * gradient[None, :], 0.0, 1.0)
    luminance = np.repeat(luminance, 64, axis=0)
    rgb = np.stack([luminance + 0.02, luminance, luminance - 0.02], axis=-1)
    Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def test_flicker_diagnostic_finds_repeated_gap_phase(tmp_path: Path) -> None:
    gap_size = 11
    records = []
    for index in range(gap_size * 4):
        phase = index % gap_size
        path = tmp_path / f"frame_{index:03d}.png"
        _write_frame(path, 0.68 if phase == 3 else 0.50)
        records.append(
            {
                "uid": f"frame_{index:03d}",
                "path": str(path),
                "kind": "flowmorph_midpoint" if phase else "base",
                "fraction": phase / gap_size if phase else None,
            }
        )

    result = diagnose_cyclic_flicker(
        records,
        tmp_path / "diagnostics",
        config=FlickerDiagnosticConfig(
            gap_size=gap_size,
            render_batch_size=4,
            minimum_outlier_score=2.0,
        ),
    )

    assert result.report_path.is_file()
    assert result.plot_path.is_file()
    assert result.report["outlier_indices"] == [3, 14, 25, 36]
    gap_hypothesis = next(
        item
        for item in result.report["hypotheses"]
        if item["mechanism"] == "repeated_position_within_final_flowmorph_gap"
    )
    assert gap_hypothesis["peak_phase"] == 3
    assert gap_hypothesis["support"] == "strong"
    markdown = format_flicker_diagnostic_markdown(result.report)
    assert "### Flicker diagnosis summary" in markdown
    assert "#### Detected pulse centers" in markdown
    assert "| 3 | frame_003 | 3 | 2 |" in markdown
    assert "#### Mean score by render-batch slot" in markdown
    assert "#### Strongest repeated lags" in markdown
    assert "#### Dominant spectral periods" in markdown


def test_flicker_diagnostic_prefers_raw_flowmorph_path(tmp_path: Path) -> None:
    records = []
    for index in range(6):
        raw_path = tmp_path / f"raw_{index:02d}.png"
        alternate_path = tmp_path / f"alternate_{index:02d}.png"
        _write_frame(raw_path, 0.5)
        _write_frame(alternate_path, 0.8)
        records.append(
            {
                "uid": f"frame_{index:02d}",
                "path": str(alternate_path),
                "raw_flowmorph_path": str(raw_path),
            }
        )

    result = diagnose_cyclic_flicker(records, tmp_path / "diagnostics")

    assert result.report["read_only"] is True
    assert result.report["raw_flowmorph_paths_preferred"] is True
    assert all(
        row["path"].endswith(f"raw_{index:02d}.png")
        for index, row in enumerate(result.report["frames"])
    )
