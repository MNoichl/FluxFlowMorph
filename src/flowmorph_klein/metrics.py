"""Endpoint, transition, and optimization metrics.

Metric provenance is stored alongside values.  Nothing in this module labels
locally computed scores as a reproduction of a paper table.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image


def _rgb_float(image: Image.Image | np.ndarray) -> np.ndarray:
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image array, received shape {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / np.iinfo(array.dtype).max
    array = array.astype(np.float32)
    if array.min() < 0 or array.max() > 1:
        raise ValueError("Floating image values must be in [0, 1]")
    return array


def psnr(reference: Image.Image | np.ndarray, generated: Image.Image | np.ndarray) -> float:
    lhs, rhs = _rgb_float(reference), _rgb_float(generated)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Image shapes differ: {lhs.shape} versus {rhs.shape}")
    mse = float(np.mean(np.square(lhs - rhs), dtype=np.float64))
    return math.inf if mse == 0 else float(10.0 * math.log10(1.0 / mse))


def ssim(reference: Image.Image | np.ndarray, generated: Image.Image | np.ndarray) -> float:
    lhs, rhs = _rgb_float(reference), _rgb_float(generated)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Image shapes differ: {lhs.shape} versus {rhs.shape}")
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(lhs, rhs, channel_axis=-1, data_range=1.0))
    except ImportError:
        # Global SSIM fallback. The implementation name is recorded by callers.
        c1, c2 = 0.01**2, 0.03**2
        mu_x, mu_y = float(lhs.mean()), float(rhs.mean())
        var_x, var_y = float(lhs.var()), float(rhs.var())
        covariance = float(np.mean((lhs - mu_x) * (rhs - mu_y)))
        numerator = (2 * mu_x * mu_y + c1) * (2 * covariance + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        return float(numerator / denominator)


def _lpips_tensor(image: Image.Image | np.ndarray, device: torch.device) -> torch.Tensor:
    array = _rgb_float(image)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) * 2 - 1


def lpips_distance(
    reference: Image.Image | np.ndarray,
    generated: Image.Image | np.ndarray,
    *,
    model: torch.nn.Module,
    device: str | torch.device | None = None,
) -> float:
    """Compute LPIPS using an explicitly supplied model (no hidden download)."""

    selected_device = torch.device(device or next(model.parameters()).device)
    with torch.inference_mode():
        value = model(_lpips_tensor(reference, selected_device), _lpips_tensor(generated, selected_device))
    return float(value.detach().float().mean().cpu())


def endpoint_reconstruction_metrics(
    reference: Image.Image | np.ndarray,
    generated: Image.Image | np.ndarray,
    reference_latent: torch.Tensor,
    generated_latent: torch.Tensor,
    *,
    lpips_model: torch.nn.Module | None = None,
    lpips_device: str | torch.device | None = None,
) -> dict[str, Any]:
    residual = generated_latent.detach().float().cpu() - reference_latent.detach().float().cpu()
    result: dict[str, Any] = {
        "psnr": psnr(reference, generated),
        "ssim": ssim(reference, generated),
        "latent_l1": float(residual.abs().mean()),
        "latent_l2": float(torch.linalg.vector_norm(residual)),
        "lpips": None,
    }
    if lpips_model is not None:
        result["lpips"] = lpips_distance(
            reference,
            generated,
            model=lpips_model,
            device=lpips_device,
        )
    return result


def transition_metrics(
    frames: Sequence[Image.Image | np.ndarray],
    *,
    lpips_model: torch.nn.Module | None = None,
    lpips_device: str | torch.device | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(frames) < 2:
        raise ValueError("At least two frames are required for transition metrics")
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(frames[:-1], frames[1:])):
        left_array, right_array = _rgb_float(left), _rgb_float(right)
        if left_array.shape != right_array.shape:
            raise ValueError("All transition frames must have identical dimensions")
        pixel_change = float(np.mean(np.abs(right_array - left_array), dtype=np.float64))
        lpips_value = (
            lpips_distance(left, right, model=lpips_model, device=lpips_device) if lpips_model is not None else None
        )
        rows.append(
            {
                "from_frame": index,
                "to_frame": index + 1,
                "pixel_l1_mean": pixel_change,
                "lpips": lpips_value,
            }
        )

    pixel_values = [row["pixel_l1_mean"] for row in rows]
    lpips_values = [row["lpips"] for row in rows if row["lpips"] is not None]
    summary: dict[str, Any] = {
        "adjacent_pixel_change_sum": float(sum(pixel_values)),
        "adjacent_pixel_change_mean": float(np.mean(pixel_values)),
        "adjacent_pixel_change_max": float(max(pixel_values)),
        "adjacent_lpips_sum": float(sum(lpips_values)) if lpips_values else None,
        "adjacent_lpips_mean": float(np.mean(lpips_values)) if lpips_values else None,
        "adjacent_lpips_max": float(max(lpips_values)) if lpips_values else None,
        "perceptual_path_length_style": (
            float(np.mean(np.square(lpips_values), dtype=np.float64)) if lpips_values else None
        ),
        "frame_count": len(frames),
    }
    return summary, rows


def summarize_optimization(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Optimization history is empty")
    losses = [float(row["total_loss"]) for row in rows]
    runtimes = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    best_index = int(np.argmin(losses))
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "relative_loss_reduction": (losses[0] - losses[-1]) / losses[0] if losses[0] else None,
        "best_loss": losses[best_index],
        "best_step": int(rows[best_index].get("step", best_index)),
        "runtime_seconds": max(runtimes) if runtimes else None,
        "mean_step_time_seconds": (
            (max(runtimes) - min(runtimes)) / max(1, len(runtimes) - 1) if len(runtimes) > 1 else None
        ),
        "peak_allocated_vram_bytes": max(int(row.get("peak_allocated_vram_bytes", 0)) for row in rows),
        "peak_reserved_vram_bytes": max(int(row.get("peak_reserved_vram_bytes", 0)) for row in rows),
    }


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    if not records:
        raise ValueError("Cannot write an empty CSV")
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return output


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return "Infinity" if float(value) > 0 else "-Infinity" if float(value) < 0 else "NaN"
        if isinstance(value, np.generic):
            return value.item()
        return value

    output.write_text(
        json.dumps(json_safe(metrics), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
