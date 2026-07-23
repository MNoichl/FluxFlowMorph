"""Safe, deterministic keyframe sampling from trajectory image archives."""

from __future__ import annotations

import hashlib
import io
import math
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from .flow_schedule import compute_empirical_mu, klein_custom_sigmas


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class TrajectoryArchiveError(RuntimeError):
    """Raised when a trajectory ZIP is unsafe, empty, or contains invalid images."""


@dataclass(frozen=True)
class Flux2KleinImg2ImgInputs:
    """True img2img inputs for the pinned FLUX.2 Klein pipeline."""

    latents: torch.Tensor
    sigmas: tuple[float, ...]
    requested_strength: float
    effective_start_sigma: float
    denoising_steps: int


def natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Return a deterministic numeric-aware key for nested member names."""

    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.casefold())
        for token in re.split(r"(\d+)", value)
        if token
    )


def regular_sample_indices(image_count: int, keyframe_count: int) -> tuple[int, ...]:
    """Select the first member of each equal-width interval.

    For 180 images and 18 keyframes this returns ``0, 10, ..., 170``.
    """

    if image_count < 1 or keyframe_count < 1:
        raise ValueError("image_count and keyframe_count must be positive")
    if keyframe_count > image_count:
        raise ValueError("keyframe_count cannot exceed image_count")
    indices = tuple(index * image_count // keyframe_count for index in range(keyframe_count))
    if len(set(indices)) != keyframe_count:
        raise AssertionError("regular sampling unexpectedly produced duplicate indices")
    return indices


def _validate_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if "\\" in name:
        raise TrajectoryArchiveError(f"ZIP member uses an unsafe backslash path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise TrajectoryArchiveError(f"ZIP member escapes the archive root: {name!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise TrajectoryArchiveError(f"ZIP member is a symbolic link: {name!r}")
    if info.flag_bits & 0x1:
        raise TrajectoryArchiveError(f"ZIP member is encrypted: {name!r}")


def list_image_members(
    archive: str | Path,
    *,
    member_prefix: str = "",
) -> tuple[str, ...]:
    """List safe image members in natural trajectory order."""

    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"trajectory ZIP does not exist: {archive_path}")
    normalized_prefix = member_prefix.strip().strip("/")
    names: list[str] = []
    with zipfile.ZipFile(archive_path) as handle:
        for info in handle.infolist():
            _validate_member(info)
            if info.is_dir():
                continue
            name = info.filename
            path = PurePosixPath(name)
            if "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            if normalized_prefix and not (
                name == normalized_prefix or name.startswith(normalized_prefix + "/")
            ):
                continue
            if path.suffix.casefold() in IMAGE_SUFFIXES:
                names.append(name)
    if not names:
        suffix = f" under {normalized_prefix!r}" if normalized_prefix else ""
        raise TrajectoryArchiveError(f"trajectory ZIP contains no supported images{suffix}")
    if len(names) != len(set(names)):
        raise TrajectoryArchiveError("trajectory ZIP contains duplicate image member names")
    return tuple(sorted(names, key=natural_sort_key))


def stage_regular_keyframes(
    archive: str | Path,
    output_directory: str | Path,
    *,
    keyframe_count: int,
    width: int,
    height: int,
    member_prefix: str = "",
    frame_offset: int = 0,
    reverse_order: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Select, normalize, and persist regularly spaced trajectory frames."""

    if width < 1 or height < 1:
        raise ValueError("trajectory output dimensions must be positive")
    members = list(list_image_members(archive, member_prefix=member_prefix))
    if reverse_order:
        members.reverse()
    if frame_offset:
        offset = int(frame_offset) % len(members)
        members = members[offset:] + members[:offset]
    selected_indices = regular_sample_indices(len(members), keyframe_count)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as handle:
        for keyframe_index, trajectory_index in enumerate(selected_indices):
            member = members[trajectory_index]
            data = handle.read(member)
            try:
                with Image.open(io.BytesIO(data)) as opened:
                    oriented = ImageOps.exif_transpose(opened)
                    original_size = tuple(oriented.size)
                    normalized = ImageOps.fit(
                        oriented.convert("RGB"),
                        (width, height),
                        method=Image.Resampling.LANCZOS,
                        centering=(0.5, 0.5),
                    )
            except (UnidentifiedImageError, OSError, ValueError) as error:
                raise TrajectoryArchiveError(
                    f"cannot decode selected trajectory member {member!r}: {error}"
                ) from error
            output_path = destination / f"selected_{keyframe_index:03d}.png"
            normalized.save(output_path, format="PNG", compress_level=4)
            records.append(
                {
                    "keyframe_index": keyframe_index,
                    "trajectory_index": trajectory_index,
                    "member": member,
                    "member_sha256": hashlib.sha256(data).hexdigest(),
                    "original_size": list(original_size),
                    "staged_size": [width, height],
                    "path": str(output_path),
                }
            )
    return tuple(records)


def make_strong_trajectory_reference(
    image: Image.Image,
    *,
    detail_strength: float = 1.0,
    blur_radius: float = 0.0,
    grain_strength: float = 0.0,
    grain_seed: int | None = None,
) -> Image.Image:
    """Create a strong, color-preserving init without blending into grey."""

    if not 0.0 <= detail_strength <= 1.0:
        raise ValueError("detail_strength must lie in [0, 1]")
    if blur_radius < 0.0:
        raise ValueError("blur_radius cannot be negative")
    if not 0.0 <= grain_strength <= 0.25:
        raise ValueError("grain_strength must lie in [0, 0.25]")
    original = image.convert("RGB")
    softened = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    reference = Image.blend(softened, original, detail_strength)
    if grain_strength == 0.0:
        return reference
    array = np.asarray(reference, dtype=np.float32)
    rng = np.random.default_rng(grain_seed)
    grain = rng.normal(
        loc=0.0,
        scale=255.0 * grain_strength,
        size=(array.shape[0], array.shape[1], 1),
    )
    return Image.fromarray(np.clip(array + grain, 0.0, 255.0).astype(np.uint8), mode="RGB")


def prepare_flux2_klein_img2img_inputs(
    pipeline: Any,
    image: Image.Image,
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    strength: float,
    generator: torch.Generator,
) -> Flux2KleinImg2ImgInputs:
    """Encode an image into the output latents and start from a late sigma.

    ``Flux2KleinPipeline(image=...)`` treats the image as reference context; it
    does not initialize the output canvas from that image. This helper supplies
    both the image latent and a truncated scheduler tail, giving conventional
    img2img behavior. Lower strength preserves more of the source composition.
    """

    if width < 1 or height < 1:
        raise ValueError("img2img output dimensions must be positive")
    if num_inference_steps < 2:
        raise ValueError("img2img num_inference_steps must be at least 2")
    if not 0.0 < strength <= 1.0:
        raise ValueError("img2img strength must lie in (0, 1]")
    scheduler_config = getattr(getattr(pipeline, "scheduler", None), "config", {})
    use_flow_sigmas = (
        scheduler_config.get("use_flow_sigmas", False)
        if isinstance(scheduler_config, dict)
        else getattr(scheduler_config, "use_flow_sigmas", False)
    )
    if use_flow_sigmas:
        raise ValueError(
            "true trajectory img2img requires a scheduler that accepts custom sigmas"
        )

    device = pipeline._execution_device
    multiple_of = int(pipeline.vae_scale_factor) * 2
    normalized_width = max(multiple_of, (int(width) // multiple_of) * multiple_of)
    normalized_height = max(multiple_of, (int(height) // multiple_of) * multiple_of)
    processed = pipeline.image_processor.preprocess(
        image.convert("RGB"),
        height=normalized_height,
        width=normalized_width,
        resize_mode="crop",
    ).to(device=device, dtype=pipeline.vae.dtype)
    with torch.no_grad():
        clean_latents = pipeline._encode_vae_image(
            image=processed,
            generator=generator,
        )

    denoising_steps = max(
        2,
        min(num_inference_steps, math.ceil(num_inference_steps * strength)),
    )
    sigmas = klein_custom_sigmas(num_inference_steps)[-denoising_steps:]
    image_seq_len = int(clean_latents.shape[-2] * clean_latents.shape[-1])
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    pipeline.scheduler.set_timesteps(
        sigmas=list(sigmas),
        device=device,
        mu=mu,
    )
    effective_start_sigma = float(pipeline.scheduler.sigmas[0].item())
    noise = torch.randn(
        clean_latents.shape,
        generator=generator,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    latents = (
        (1.0 - effective_start_sigma) * clean_latents
        + effective_start_sigma * noise
    )
    return Flux2KleinImg2ImgInputs(
        latents=latents,
        sigmas=tuple(float(value) for value in sigmas),
        requested_strength=float(strength),
        effective_start_sigma=effective_start_sigma,
        denoising_steps=denoising_steps,
    )


__all__ = [
    "Flux2KleinImg2ImgInputs",
    "IMAGE_SUFFIXES",
    "TrajectoryArchiveError",
    "list_image_members",
    "make_strong_trajectory_reference",
    "natural_sort_key",
    "prepare_flux2_klein_img2img_inputs",
    "regular_sample_indices",
    "stage_regular_keyframes",
]
