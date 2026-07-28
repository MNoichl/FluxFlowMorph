"""Read-only cyclic flicker diagnostics for generated image sequences."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class FlickerDiagnosticConfig:
    """Settings for cyclic flicker measurement and periodicity analysis."""

    analysis_max_side: int = 256
    outlier_mad_multiplier: float = 3.5
    minimum_outlier_score: float = 3.0
    max_lag: int = 64
    gap_size: int | None = None
    render_batch_size: int | None = None

    def validate(self, frame_count: int) -> None:
        if frame_count < 5:
            raise ValueError("flicker diagnosis requires at least five frames")
        if self.analysis_max_side < 32:
            raise ValueError("analysis_max_side must be at least 32")
        if self.outlier_mad_multiplier < 0:
            raise ValueError("outlier_mad_multiplier cannot be negative")
        if self.minimum_outlier_score < 0:
            raise ValueError("minimum_outlier_score cannot be negative")
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive")
        if self.gap_size is not None and self.gap_size < 2:
            raise ValueError("gap_size must be at least two")
        if self.render_batch_size is not None and self.render_batch_size < 1:
            raise ValueError("render_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class FlickerDiagnosticResult:
    """Audit paths and in-memory report returned by the diagnostic."""

    report: dict[str, Any]
    report_path: Path
    plot_path: Path


def _metric_array(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as opened:
        sample = opened.convert("RGB")
        sample.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(sample, dtype=np.float32) / np.float32(255.0)


def _frame_metrics(path: Path, max_side: int) -> tuple[float, float, float, float]:
    rgb = _metric_array(path, max_side)
    luminance = (
        rgb[..., 0] * np.float32(0.2126)
        + rgb[..., 1] * np.float32(0.7152)
        + rgb[..., 2] * np.float32(0.0722)
    )
    horizontal = np.abs(np.diff(luminance, axis=1)).mean()
    vertical = np.abs(np.diff(luminance, axis=0)).mean()
    return (
        float(luminance.mean()),
        float(luminance.std()),
        float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
        float((horizontal + vertical) * 0.5),
    )


def _cyclic_neighbor_residual(values: np.ndarray) -> np.ndarray:
    expected = (np.roll(values, 1, axis=0) + np.roll(values, -1, axis=0)) * 0.5
    return values - expected


def _robust_z_scores(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.median(residuals, axis=0)
    mad = np.median(np.abs(residuals - centers), axis=0)
    scales = np.maximum(1.4826 * mad, np.asarray([0.003, 0.003, 0.003, 0.001]))
    return (residuals - centers) / scales, centers, scales


def _autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    centered = signal - signal.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return np.zeros(max_lag, dtype=np.float64)
    return np.asarray(
        [
            float(np.dot(centered, np.roll(centered, -lag)) / denominator)
            for lag in range(1, max_lag + 1)
        ],
        dtype=np.float64,
    )


def _dominant_periods(signal: np.ndarray, limit: int = 8) -> list[dict[str, float]]:
    centered = signal - signal.mean()
    amplitudes = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(len(centered))
    candidates = [
        (index, float(amplitudes[index]))
        for index in range(1, len(amplitudes))
        if frequencies[index] > 0
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        {
            "frequency_cycles_per_frame": float(frequencies[index]),
            "period_frames": float(1.0 / frequencies[index]),
            "amplitude": amplitude,
        }
        for index, amplitude in candidates[:limit]
    ]


def _phase_summary(
    scores: np.ndarray,
    outliers: np.ndarray,
    phases: np.ndarray,
) -> list[dict[str, float | int]]:
    rows = []
    for phase in sorted(set(int(value) for value in phases)):
        selected = phases == phase
        rows.append(
            {
                "phase": phase,
                "count": int(selected.sum()),
                "mean_score": float(scores[selected].mean()),
                "median_score": float(np.median(scores[selected])),
                "outlier_count": int(outliers[selected].sum()),
                "outlier_rate": float(outliers[selected].mean()),
            }
        )
    return rows


def _safe_ratio(maximum: float, baseline: float) -> float:
    return float(maximum / baseline) if baseline > 1e-12 else 0.0


def _pattern_hypotheses(
    *,
    gap_summary: list[dict[str, float | int]],
    batch_summary: list[dict[str, float | int]],
    autocorrelation: np.ndarray,
    dominant_periods: list[dict[str, float]],
    config: FlickerDiagnosticConfig,
) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    if gap_summary:
        scores = np.asarray([float(row["mean_score"]) for row in gap_summary])
        outlier_rates = np.asarray(
            [float(row["outlier_rate"]) for row in gap_summary]
        )
        peak = int(np.argmax(scores))
        ratio = _safe_ratio(float(scores[peak]), float(np.median(scores)))
        concentrated_outliers = (
            outlier_rates[peak] >= 0.5
            and outlier_rates[peak] > float(np.median(outlier_rates))
        )
        hypotheses.append(
            {
                "mechanism": "repeated_position_within_final_flowmorph_gap",
                "peak_phase": int(gap_summary[peak]["phase"]),
                "peak_to_median_score_ratio": ratio,
                "support": (
                    "strong" if ratio >= 1.5 or concentrated_outliers else "weak"
                ),
            }
        )
    if batch_summary:
        scores = np.asarray([float(row["mean_score"]) for row in batch_summary])
        outlier_rates = np.asarray(
            [float(row["outlier_rate"]) for row in batch_summary]
        )
        peak = int(np.argmax(scores))
        ratio = _safe_ratio(float(scores[peak]), float(np.median(scores)))
        concentrated_outliers = (
            outlier_rates[peak] >= 0.5
            and outlier_rates[peak] > float(np.median(outlier_rates))
        )
        hypotheses.append(
            {
                "mechanism": "render_batch_slot",
                "peak_slot": int(batch_summary[peak]["phase"]),
                "peak_to_median_score_ratio": ratio,
                "support": (
                    "strong" if ratio >= 1.5 or concentrated_outliers else "weak"
                ),
            }
        )
    for label, period in (
        ("final_gap_period", config.gap_size),
        ("render_batch_period", config.render_batch_size),
    ):
        if period is None or period < 1 or period > len(autocorrelation):
            continue
        correlation = float(autocorrelation[period - 1])
        hypotheses.append(
            {
                "mechanism": label,
                "period_frames": period,
                "autocorrelation": correlation,
                "support": "strong" if correlation >= 0.35 else "weak",
            }
        )
    if dominant_periods:
        hypotheses.append(
            {
                "mechanism": "strongest_spectral_period",
                **dominant_periods[0],
                "support": "descriptive_only",
            }
        )
    return hypotheses


def diagnose_cyclic_flicker(
    records: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
    *,
    config: FlickerDiagnosticConfig | None = None,
) -> FlickerDiagnosticResult:
    """Measure frame-local tone pulses and test common repeated phases.

    This is deliberately read-only: it creates a JSON audit and plot but never
    rewrites, normalizes, or otherwise alters an input image.
    """

    settings = config or FlickerDiagnosticConfig()
    settings.validate(len(records))
    paths = tuple(
        Path(record.get("raw_flowmorph_path", record["path"]))
        for record in records
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"flicker source frame does not exist: {missing[0]}")

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    names = ("luminance_mean", "luminance_contrast", "saturation", "edge_energy")
    metrics = np.asarray(
        [_frame_metrics(path, settings.analysis_max_side) for path in paths],
        dtype=np.float64,
    )
    residuals = _cyclic_neighbor_residual(metrics)
    z_scores, residual_centers, residual_scales = _robust_z_scores(residuals)
    flicker_scores = np.sqrt(np.mean(np.square(z_scores), axis=1))
    score_center = float(np.median(flicker_scores))
    score_mad = float(np.median(np.abs(flicker_scores - score_center)))
    score_limit = max(
        settings.minimum_outlier_score,
        score_center + settings.outlier_mad_multiplier * 1.4826 * score_mad,
    )
    affected = flicker_scores > score_limit
    outliers = (
        affected
        & (flicker_scores >= np.roll(flicker_scores, 1))
        & (flicker_scores > np.roll(flicker_scores, -1))
    )
    max_lag = min(settings.max_lag, max(1, len(records) // 2))
    autocorrelation = _autocorrelation(flicker_scores, max_lag)
    dominant_periods = _dominant_periods(flicker_scores)

    gap_phases = (
        np.arange(len(records), dtype=np.int64) % settings.gap_size
        if settings.gap_size is not None
        else None
    )
    gap_summary = (
        _phase_summary(flicker_scores, outliers, gap_phases)
        if gap_phases is not None
        else []
    )
    batch_summary: list[dict[str, float | int]] = []
    if gap_phases is not None and settings.render_batch_size is not None:
        midpoint_frames = gap_phases > 0
        batch_slots = (gap_phases[midpoint_frames] - 1) % settings.render_batch_size
        batch_summary = _phase_summary(
            flicker_scores[midpoint_frames],
            outliers[midpoint_frames],
            batch_slots,
        )

    frame_rows = []
    for index, (record, path) in enumerate(zip(records, paths, strict=True)):
        row = {
            "index": index,
            "uid": record.get("uid"),
            "path": str(path),
            "kind": record.get("kind"),
            "round": record.get("round"),
            "fraction": record.get("fraction"),
            "alpha": record.get("alpha"),
            "left_uid": record.get("left_uid"),
            "right_uid": record.get("right_uid"),
            "flicker_score": float(flicker_scores[index]),
            "outlier": bool(outliers[index]),
            "affected_by_local_pulse": bool(affected[index]),
            "gap_phase": (
                int(gap_phases[index]) if gap_phases is not None else None
            ),
            "render_batch_slot": (
                int((gap_phases[index] - 1) % settings.render_batch_size)
                if gap_phases is not None
                and settings.render_batch_size is not None
                and gap_phases[index] > 0
                else None
            ),
        }
        for metric_index, name in enumerate(names):
            row[name] = float(metrics[index, metric_index])
            row[f"{name}_neighbor_residual"] = float(
                residuals[index, metric_index]
            )
            row[f"{name}_robust_z"] = float(z_scores[index, metric_index])
        frame_rows.append(row)

    hypotheses = _pattern_hypotheses(
        gap_summary=gap_summary,
        batch_summary=batch_summary,
        autocorrelation=autocorrelation,
        dominant_periods=dominant_periods,
        config=settings,
    )
    report = {
        "diagnostic": "cyclic_frame_local_flicker_v1",
        "read_only": True,
        "raw_flowmorph_paths_preferred": True,
        "config": asdict(settings),
        "frame_count": len(records),
        "metric_names": list(names),
        "residual_centers": dict(zip(names, residual_centers.tolist(), strict=True)),
        "residual_scales": dict(zip(names, residual_scales.tolist(), strict=True)),
        "outlier_score_limit": score_limit,
        "outlier_count": int(outliers.sum()),
        "outlier_indices": np.flatnonzero(outliers).astype(int).tolist(),
        "affected_count": int(affected.sum()),
        "affected_indices": np.flatnonzero(affected).astype(int).tolist(),
        "gap_phase_summary": gap_summary,
        "render_batch_slot_summary": batch_summary,
        "autocorrelation_by_lag": [
            {"lag": lag, "correlation": float(value)}
            for lag, value in enumerate(autocorrelation, start=1)
        ],
        "dominant_periods": dominant_periods,
        "hypotheses": hypotheses,
        "frames": frame_rows,
    }
    report_path = output_root / "flicker_diagnosis.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - Colab requirements pin matplotlib
        raise RuntimeError("matplotlib is required to render flicker diagnostics") from error

    plot_path = output_root / "flicker_diagnosis.png"
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    frame_indices = np.arange(len(records))
    axes[0, 0].plot(metrics[:, 0], label="mean luminance", linewidth=0.9)
    axes[0, 0].plot(metrics[:, 1], label="luminance contrast", linewidth=0.9)
    axes[0, 0].set_title("Raw FlowMorph tone metrics")
    axes[0, 0].legend()

    axes[0, 1].plot(flicker_scores, color="#4b6a88", linewidth=0.9)
    axes[0, 1].axhline(score_limit, color="#c43d32", linestyle="--", label="outlier limit")
    axes[0, 1].scatter(
        frame_indices[outliers],
        flicker_scores[outliers],
        color="#c43d32",
        s=22,
        zorder=3,
        label="detected outlier",
    )
    axes[0, 1].set_title("Neighbor-relative flicker score")
    axes[0, 1].legend()

    if gap_summary:
        phases = [int(row["phase"]) for row in gap_summary]
        phase_scores = [float(row["mean_score"]) for row in gap_summary]
        axes[1, 0].bar(phases, phase_scores, color="#7a8f6a")
        axes[1, 0].set_xticks(phases)
        axes[1, 0].set_title("Mean flicker score by position within final gap")
        axes[1, 0].set_xlabel("0 = fitted endpoint; 1…N = FlowMorph midpoint")
    else:
        axes[1, 0].axis("off")

    lags = np.arange(1, len(autocorrelation) + 1)
    axes[1, 1].plot(lags, autocorrelation, color="#7a5535", linewidth=0.9)
    if settings.gap_size is not None and settings.gap_size <= len(autocorrelation):
        axes[1, 1].axvline(
            settings.gap_size,
            color="#466b8f",
            linestyle="--",
            label=f"gap period {settings.gap_size}",
        )
    if (
        settings.render_batch_size is not None
        and settings.render_batch_size <= len(autocorrelation)
    ):
        axes[1, 1].axvline(
            settings.render_batch_size,
            color="#8f5f46",
            linestyle=":",
            label=f"render batch {settings.render_batch_size}",
        )
    axes[1, 1].set_title("Flicker-score autocorrelation")
    axes[1, 1].set_xlabel("lag in generated frames")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend()

    for axis in axes.flat:
        if axis.axison:
            axis.grid(alpha=0.2)
    figure.savefig(plot_path, dpi=160, facecolor="white")
    plt.close(figure)

    return FlickerDiagnosticResult(
        report=report,
        report_path=report_path,
        plot_path=plot_path,
    )


__all__ = [
    "FlickerDiagnosticConfig",
    "FlickerDiagnosticResult",
    "diagnose_cyclic_flicker",
]
