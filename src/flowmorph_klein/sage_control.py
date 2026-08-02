"""Dense spatial initialization for SAGE-guided FLUX.2 rendering.

SAGE supplies corresponding endpoint line segments and their interpolated
positions.  A colored line raster alone is too sparse to initialize an image
generator, so this module turns those correspondences into two dense
piecewise-affine endpoint warps and blends the detailed warps at the same time
coordinate.  The result is a conventional full-resolution img2img canvas, not
an overlay of the control colors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class SageWarpResult:
    """Detailed warped initialization and its two endpoint contributions."""

    image: Image.Image
    warped_left: Image.Image
    warped_right: Image.Image
    effective_alpha: float
    source_control_points: int
    target_control_points: int


def smoothstep(value: float) -> float:
    """Return a cubic ease with zero endpoint slope."""

    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _border_points(width: int, height: int, samples_per_edge: int) -> np.ndarray:
    if width < 2 or height < 2:
        raise ValueError("warp dimensions must be at least 2x2")
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least 2")
    xs = np.linspace(0.0, width - 1.0, samples_per_edge)
    ys = np.linspace(0.0, height - 1.0, samples_per_edge)
    points = []
    points.extend((float(x), 0.0) for x in xs)
    points.extend((float(x), float(height - 1)) for x in xs)
    points.extend((0.0, float(y)) for y in ys[1:-1])
    points.extend((float(width - 1), float(y)) for y in ys[1:-1])
    return np.asarray(points, dtype=np.float64)


def _select_lines(lines: np.ndarray, maximum: int | None) -> np.ndarray:
    array = np.asarray(lines, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (2, 2):
        raise ValueError("line arrays must have shape (line, endpoint, xy)")
    if maximum is None or len(array) <= maximum:
        return np.arange(len(array), dtype=np.int64)
    if maximum < 3:
        raise ValueError("maximum control lines must be at least 3")
    return np.unique(
        np.rint(np.linspace(0, len(array) - 1, maximum)).astype(np.int64)
    )


def _deduplicate_mapping(
    source: np.ndarray,
    destination: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident source points and average their destinations."""

    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("control point arrays must have identical shape (point, xy)")
    source[:, 0] = np.clip(source[:, 0], 0.0, width - 1.0)
    source[:, 1] = np.clip(source[:, 1], 0.0, height - 1.0)
    destination[:, 0] = np.clip(destination[:, 0], 0.0, width - 1.0)
    destination[:, 1] = np.clip(destination[:, 1], 0.0, height - 1.0)
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    representatives: dict[tuple[int, int], np.ndarray] = {}
    for point, mapped in zip(source, destination):
        key = tuple(np.rint(point * 4.0).astype(np.int64))
        representatives.setdefault(key, point)
        buckets.setdefault(key, []).append(mapped)
    unique_source = np.stack([representatives[key] for key in buckets])
    unique_destination = np.stack(
        [np.mean(buckets[key], axis=0) for key in buckets]
    )
    if len(unique_source) < 3:
        raise ValueError("at least three unique control points are required")
    return unique_source, unique_destination


def _warp_image(
    image: Image.Image,
    source_points: np.ndarray,
    destination_points: np.ndarray,
) -> Image.Image:
    try:
        from skimage.transform import PiecewiseAffineTransform, warp
    except ImportError as error:  # pragma: no cover - Colab dependency contract
        raise RuntimeError(
            "scikit-image==0.26.0 is required for dense SAGE endpoint warping"
        ) from error
    width, height = image.size
    try:
        constructor = getattr(PiecewiseAffineTransform, "from_estimate", None)
        if callable(constructor):
            transform = constructor(source_points, destination_points)
        else:
            # scikit-image <=0.25 exposes only the mutable instance API. Some
            # Colab images can retain that version until the kernel is fully
            # restarted even after the pinned requirements cell has run.
            transform = PiecewiseAffineTransform()
            if not transform.estimate(source_points, destination_points):
                raise RuntimeError("piecewise-affine estimator returned false")
    except (ValueError, RuntimeError) as error:
        raise RuntimeError("piecewise-affine SAGE warp estimation failed") from error
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    warped = warp(
        source,
        inverse_map=transform.inverse,
        output_shape=(height, width),
        order=1,
        mode="edge",
        preserve_range=True,
    )
    return Image.fromarray(
        np.clip(np.rint(warped * 255.0), 0, 255).astype(np.uint8),
        mode="RGB",
    )


def warp_sage_endpoints(
    left: Image.Image,
    right: Image.Image,
    source_lines: np.ndarray,
    target_lines: np.ndarray,
    intermediate_lines: np.ndarray,
    alpha: float,
    *,
    maximum_control_lines: int | None = 96,
    border_samples_per_edge: int = 7,
    eased_blend: bool = True,
    grain_strength: float = 0.0,
    seed: int = 0,
) -> SageWarpResult:
    """Warp both detailed endpoints to one SAGE structure and blend them."""

    if left.size != right.size:
        raise ValueError("SAGE endpoint images must have identical dimensions")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not 0.0 <= grain_strength <= 0.25:
        raise ValueError("grain_strength must lie in [0, 0.25]")
    source_lines = np.asarray(source_lines, dtype=np.float64)
    target_lines = np.asarray(target_lines, dtype=np.float64)
    intermediate_lines = np.asarray(intermediate_lines, dtype=np.float64)
    if source_lines.shape != target_lines.shape or source_lines.shape != intermediate_lines.shape:
        raise ValueError("source, target, and intermediate line arrays must match")
    selected = _select_lines(source_lines, maximum_control_lines)
    width, height = left.size
    border = _border_points(width, height, border_samples_per_edge)
    middle_points = intermediate_lines[selected].reshape(-1, 2)
    left_source, left_destination = _deduplicate_mapping(
        np.concatenate([source_lines[selected].reshape(-1, 2), border], axis=0),
        np.concatenate([middle_points, border], axis=0),
        width=width,
        height=height,
    )
    right_source, right_destination = _deduplicate_mapping(
        np.concatenate([target_lines[selected].reshape(-1, 2), border], axis=0),
        np.concatenate([middle_points, border], axis=0),
        width=width,
        height=height,
    )
    warped_left = _warp_image(left, left_source, left_destination)
    warped_right = _warp_image(right, right_source, right_destination)
    effective_alpha = smoothstep(alpha) if eased_blend else float(alpha)
    blended = Image.blend(warped_left, warped_right, effective_alpha)
    if grain_strength:
        array = np.asarray(blended, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(
            0.0,
            255.0 * grain_strength,
            size=(height, width, 1),
        )
        blended.close()
        blended = Image.fromarray(
            np.clip(np.rint(array + noise), 0, 255).astype(np.uint8),
            mode="RGB",
        )
    return SageWarpResult(
        image=blended,
        warped_left=warped_left,
        warped_right=warped_right,
        effective_alpha=effective_alpha,
        source_control_points=len(left_source),
        target_control_points=len(right_source),
    )


def make_canny_reference(condition: Image.Image, *, dilation_pixels: int = 1) -> Image.Image:
    """Convert a colored SAGE line raster to RefControl's black/white format."""

    if dilation_pixels < 0:
        raise ValueError("dilation_pixels cannot be negative")
    array = np.asarray(condition.convert("RGB"), dtype=np.uint8)
    binary = np.where(array.max(axis=2) > 0, 255, 0).astype(np.uint8)
    output = Image.fromarray(binary, mode="L")
    if dilation_pixels:
        output = output.filter(ImageFilter.MaxFilter(dilation_pixels * 2 + 1))
    return output.convert("RGB")


def make_structure_lock_mask(
    condition: Image.Image,
    *,
    dilation_pixels: int = 14,
    feather_radius: float = 4.0,
    strength: float = 0.55,
) -> Image.Image:
    """Create a grayscale white-lock mask around SAGE structural lines."""

    if dilation_pixels < 0 or feather_radius < 0:
        raise ValueError("lock dilation and feather must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("lock strength must lie in [0, 1]")
    array = np.asarray(condition.convert("RGB"), dtype=np.uint8)
    mask = Image.fromarray(
        np.where(array.max(axis=2) > 0, 255, 0).astype(np.uint8),
        mode="L",
    )
    if dilation_pixels:
        mask = mask.filter(ImageFilter.MaxFilter(dilation_pixels * 2 + 1))
    if feather_radius:
        mask = mask.filter(ImageFilter.GaussianBlur(feather_radius))
    if strength < 1.0:
        mask = mask.point(lambda value: int(round(value * strength)))
    return mask


__all__ = [
    "SageWarpResult",
    "make_canny_reference",
    "make_structure_lock_mask",
    "smoothstep",
    "warp_sage_endpoints",
]
