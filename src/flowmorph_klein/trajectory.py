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
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from scipy.ndimage import maximum_filter

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


@dataclass(frozen=True)
class TrajectoryActivityGuide:
    """Neutral spatial guide derived from non-background source activity."""

    image: Image.Image
    mask: Image.Image
    estimated_background_rgb: tuple[int, int, int]
    coverage_fraction: float


@dataclass(frozen=True)
class BackgroundEditMask:
    """A continuous white-edit/black-protect mask."""

    mask: Image.Image
    background_rgb: tuple[int, int, int] | None
    editable_fraction: float


@dataclass(frozen=True)
class Flux2KleinMaskedInpaintInputs:
    """Latents and callback for background-locked FLUX.2 Klein inpainting."""

    latents: torch.Tensor
    sigmas: tuple[float, ...]
    callback_on_step_end: Callable[[Any, int, torch.Tensor, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    requested_strength: float
    effective_start_sigma: float
    denoising_steps: int
    used_init_image: bool


@dataclass(frozen=True)
class Flux2KleinSpatialLockInputs:
    """Img2img latents plus a callback that retains selected init structure."""

    latents: torch.Tensor
    sigmas: tuple[float, ...]
    callback_on_step_end: Callable[[Any, int, torch.Tensor, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
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


def make_trajectory_activity_guide(
    image: Image.Image,
    *,
    threshold: float = 0.1,
    softness: float = 0.12,
    blur_radius: float = 20.0,
    expansion_radius: int = 6,
    contrast: float = 0.25,
    background_rgb: tuple[int, int, int] = (238, 233, 218),
    active_is_light: bool = True,
) -> TrajectoryActivityGuide:
    """Convert a trajectory frame into a color-free soft occupancy guide.

    The source background is estimated from its border. Pixel distance from
    that background becomes an activity mask; the original color and texture
    are discarded so they cannot survive as an underpainting.
    """

    if not 0.0 <= threshold < 1.0:
        raise ValueError("activity threshold must lie in [0, 1)")
    if not 0.0 < softness <= 1.0:
        raise ValueError("activity softness must lie in (0, 1]")
    if blur_radius < 0.0:
        raise ValueError("activity blur_radius cannot be negative")
    if not isinstance(expansion_radius, int) or not 0 <= expansion_radius <= 128:
        raise ValueError("activity expansion_radius must be an integer in [0, 128]")
    if not 0.0 < contrast <= 1.0:
        raise ValueError("activity guide contrast must lie in (0, 1]")
    if len(background_rgb) != 3 or any(not 0 <= channel <= 255 for channel in background_rgb):
        raise ValueError("activity guide background_rgb must contain three values in [0, 255]")

    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = source.shape[:2]
    border_width = max(1, min(height, width) // 50)
    border = np.concatenate(
        [
            source[:border_width, :, :].reshape(-1, 3),
            source[-border_width:, :, :].reshape(-1, 3),
            source[:, :border_width, :].reshape(-1, 3),
            source[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    estimated_background = np.median(border, axis=0)
    distance = np.sqrt(np.mean(np.square(source - estimated_background), axis=2))
    mask_array = np.clip((distance - threshold) / softness, 0.0, 1.0)
    mask_array = mask_array * mask_array * (3.0 - 2.0 * mask_array)
    mask = Image.fromarray(np.round(mask_array * 255.0).astype(np.uint8), mode="L")
    if expansion_radius:
        mask = mask.filter(ImageFilter.MaxFilter(2 * expansion_radius + 1))
    if blur_radius:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    softened_mask = np.asarray(mask, dtype=np.float32) / 255.0
    background = np.asarray(background_rgb, dtype=np.float32)
    shadow_tone = background * (1.0 - contrast)
    if active_is_light:
        inactive_tone, active_tone = shadow_tone, background
    else:
        inactive_tone, active_tone = background, shadow_tone
    guide_array = (
        inactive_tone[None, None, :] * (1.0 - softened_mask[:, :, None])
        + active_tone[None, None, :] * softened_mask[:, :, None]
    )
    guide = Image.fromarray(np.round(guide_array).clip(0, 255).astype(np.uint8), mode="RGB")
    return TrajectoryActivityGuide(
        image=guide,
        mask=mask,
        estimated_background_rgb=tuple(
            int(round(channel * 255.0)) for channel in estimated_background
        ),
        coverage_fraction=float(np.mean(softened_mask >= 0.5)),
    )


def make_background_edit_mask(
    image: Image.Image,
    *,
    background_rgb: tuple[int, int, int],
    tolerance: float = 0.08,
    softness: float = 0.05,
    expansion_radius: int = 0,
    feather_radius: float = 3.0,
) -> BackgroundEditMask:
    """Make a mask where non-background pixels are editable.

    Pixels sufficiently close to ``background_rgb`` become black (protected);
    everything else becomes white (editable). ``softness`` and
    ``feather_radius`` create a gradual boundary without changing the polarity.
    """

    if len(background_rgb) != 3 or any(not 0 <= channel <= 255 for channel in background_rgb):
        raise ValueError("background_rgb must contain three values in [0, 255]")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("background tolerance must lie in [0, 1)")
    if not 0.0 < softness <= 1.0:
        raise ValueError("background softness must lie in (0, 1]")
    if not isinstance(expansion_radius, int) or not 0 <= expansion_radius <= 128:
        raise ValueError("mask expansion_radius must be an integer in [0, 128]")
    if feather_radius < 0.0:
        raise ValueError("mask feather_radius cannot be negative")

    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    background = np.asarray(background_rgb, dtype=np.float32) / 255.0
    distance = np.sqrt(np.mean(np.square(source - background), axis=2))
    editable = np.clip((distance - tolerance) / softness, 0.0, 1.0)
    editable = editable * editable * (3.0 - 2.0 * editable)
    mask = Image.fromarray(np.round(editable * 255.0).astype(np.uint8), mode="L")
    if expansion_radius:
        mask = mask.filter(ImageFilter.MaxFilter(2 * expansion_radius + 1))
    if feather_radius:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    return BackgroundEditMask(
        mask=mask,
        background_rgb=tuple(int(channel) for channel in background_rgb),
        editable_fraction=float(np.mean(mask_array)),
    )


def prepare_grayscale_edit_mask(
    image: Image.Image,
    *,
    invert: bool = False,
    gamma: float = 1.0,
    expansion_radius: int = 0,
    feather_radius: float = 0.0,
) -> BackgroundEditMask:
    """Load a continuous grayscale mask without binarizing it.

    Black pixels remain protected, white pixels remain editable, and intermediate
    values retain proportional influence. ``gamma`` is optional tone shaping:
    values above one reduce mid-gray editability and values below one increase it.
    """

    if gamma <= 0.0:
        raise ValueError("mask gamma must be positive")
    if not isinstance(expansion_radius, int) or not 0 <= expansion_radius <= 128:
        raise ValueError("mask expansion_radius must be an integer in [0, 128]")
    if feather_radius < 0.0:
        raise ValueError("mask feather_radius cannot be negative")

    mask_array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if invert:
        mask_array = 1.0 - mask_array
    if gamma != 1.0:
        mask_array = np.power(mask_array, gamma)
    mask = Image.fromarray(
        np.clip(np.round(mask_array * 255.0), 0.0, 255.0).astype(np.uint8),
        mode="L",
    )
    if expansion_radius:
        yy, xx = np.ogrid[
            -expansion_radius : expansion_radius + 1,
            -expansion_radius : expansion_radius + 1,
        ]
        circular_footprint = (xx * xx + yy * yy) <= expansion_radius * expansion_radius
        expanded = maximum_filter(
            np.asarray(mask, dtype=np.uint8),
            footprint=circular_footprint,
            mode="nearest",
        )
        if expanded.shape != (mask.height, mask.width):
            raise RuntimeError("circular mask expansion produced an invalid image")
        mask = Image.fromarray(expanded, mode="L")
    if feather_radius:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    final_array = np.asarray(mask, dtype=np.float32) / 255.0
    return BackgroundEditMask(
        mask=mask,
        background_rgb=None,
        editable_fraction=float(np.mean(final_array)),
    )


def composite_generated_activity(
    generated: Image.Image,
    mask: Image.Image,
    *,
    background_rgb: tuple[int, int, int] = (238, 233, 218),
    outside_opacity: float = 0.0,
) -> Image.Image:
    """Keep generated content inside activity while neutralizing empty regions."""

    if len(background_rgb) != 3 or any(not 0 <= channel <= 255 for channel in background_rgb):
        raise ValueError("activity composite background_rgb must contain values in [0, 255]")
    if not 0.0 <= outside_opacity <= 1.0:
        raise ValueError("activity composite outside_opacity must lie in [0, 1]")
    output = generated.convert("RGB")
    resized_mask = mask.convert("L").resize(output.size, Image.Resampling.LANCZOS)
    mask_array = np.asarray(resized_mask, dtype=np.float32) / 255.0
    alpha = outside_opacity + (1.0 - outside_opacity) * mask_array
    alpha_image = Image.fromarray(
        np.round(alpha * 255.0).clip(0, 255).astype(np.uint8),
        mode="L",
    )
    background = Image.new("RGB", output.size, background_rgb)
    return Image.composite(output, background, alpha_image)


def composite_generated_on_background(
    generated: Image.Image,
    edit_mask: Image.Image,
    *,
    background_rgb: tuple[int, int, int],
) -> Image.Image:
    """Place generated pixels only in white/editable mask regions."""

    if len(background_rgb) != 3 or any(not 0 <= channel <= 255 for channel in background_rgb):
        raise ValueError("background_rgb must contain three values in [0, 255]")
    output = generated.convert("RGB")
    resized_mask = edit_mask.convert("L").resize(output.size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", output.size, background_rgb)
    return Image.composite(output, background, resized_mask)


def prepare_flux2_klein_masked_inpaint_inputs(
    pipeline: Any,
    edit_mask: Image.Image,
    *,
    background_rgb: tuple[int, int, int],
    init_image: Image.Image | None = None,
    width: int,
    height: int,
    num_inference_steps: int,
    strength: float,
    generator: torch.Generator,
) -> Flux2KleinMaskedInpaintInputs:
    """Prepare full repainting inside a mask while locking the background.

    Generation starts from either noise over a flat canvas or a noised init image
    inside editable regions. After every denoising step, black mask regions are
    restored to the correctly noised flat-background latent; white regions remain
    free for prompt-driven generation. Gray values blend those behaviors.
    """

    if len(background_rgb) != 3 or any(not 0 <= channel <= 255 for channel in background_rgb):
        raise ValueError("background_rgb must contain three values in [0, 255]")
    if width < 1 or height < 1:
        raise ValueError("masked inpaint output dimensions must be positive")
    if num_inference_steps < 2:
        raise ValueError("masked inpaint num_inference_steps must be at least 2")
    if not 0.0 < strength <= 1.0:
        raise ValueError("masked inpaint strength must lie in (0, 1]")
    scheduler_config = getattr(getattr(pipeline, "scheduler", None), "config", {})
    use_flow_sigmas = (
        scheduler_config.get("use_flow_sigmas", False)
        if isinstance(scheduler_config, dict)
        else getattr(scheduler_config, "use_flow_sigmas", False)
    )
    if use_flow_sigmas:
        raise ValueError(
            "masked trajectory generation requires a scheduler that accepts custom sigmas"
        )

    device = pipeline._execution_device
    multiple_of = int(pipeline.vae_scale_factor) * 2
    normalized_width = max(multiple_of, (int(width) // multiple_of) * multiple_of)
    normalized_height = max(multiple_of, (int(height) // multiple_of) * multiple_of)
    background_image = Image.new(
        "RGB",
        (normalized_width, normalized_height),
        tuple(int(channel) for channel in background_rgb),
    )
    processed = pipeline.image_processor.preprocess(
        background_image,
        height=normalized_height,
        width=normalized_width,
        resize_mode="crop",
    ).to(device=device, dtype=pipeline.vae.dtype)
    with torch.no_grad():
        background_latents = pipeline._encode_vae_image(
            image=processed,
            generator=generator,
        )

    latent_height, latent_width = background_latents.shape[-2:]
    resized_mask = edit_mask.convert("L").resize(
        (latent_width, latent_height),
        Image.Resampling.LANCZOS,
    )
    mask_array = np.asarray(resized_mask, dtype=np.float32) / 255.0
    edit_mask_latents = torch.from_numpy(mask_array).to(
        device=background_latents.device,
        dtype=background_latents.dtype,
    )[None, None, :, :]
    # FLUX.2 may pack 2x2 spatial neighborhoods into the channel dimension.
    # Packing an expanded mask through the pipeline itself guarantees that the
    # callback mask has exactly the same token/channel shape as its latents.
    edit_mask_packed = pipeline._pack_latents(
        edit_mask_latents.expand(-1, background_latents.shape[1], -1, -1)
    )

    clean_latents = background_latents
    if init_image is not None:
        processed_init = pipeline.image_processor.preprocess(
            init_image.convert("RGB"),
            height=normalized_height,
            width=normalized_width,
            resize_mode="crop",
        ).to(device=device, dtype=pipeline.vae.dtype)
        with torch.no_grad():
            init_latents = pipeline._encode_vae_image(
                image=processed_init,
                generator=generator,
            )
        if init_latents.shape != background_latents.shape:
            raise ValueError(
                "encoded init image and background latent shapes do not match"
            )
        clean_latents = (
            edit_mask_latents * init_latents
            + (1.0 - edit_mask_latents) * background_latents
        )

    denoising_steps = max(
        2,
        min(num_inference_steps, math.ceil(num_inference_steps * strength)),
    )
    sigmas = klein_custom_sigmas(num_inference_steps)[-denoising_steps:]
    image_seq_len = int(background_latents.shape[-2] * background_latents.shape[-1])
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    pipeline.scheduler.set_timesteps(
        sigmas=list(sigmas),
        device=device,
        mu=mu,
    )
    effective_start_sigma = float(pipeline.scheduler.sigmas[0].item())
    fixed_noise = torch.randn(
        background_latents.shape,
        generator=generator,
        device=background_latents.device,
        dtype=background_latents.dtype,
    )
    latents = (
        (1.0 - effective_start_sigma) * clean_latents
        + effective_start_sigma * fixed_noise
    )

    def lock_background_callback(
        current_pipeline: Any,
        step_index: int,
        _timestep: torch.Tensor,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        current_latents = callback_kwargs["latents"]
        scheduler_sigmas = current_pipeline.scheduler.sigmas
        next_index = min(step_index + 1, len(scheduler_sigmas) - 1)
        next_sigma = scheduler_sigmas[next_index].to(
            device=background_latents.device,
            dtype=background_latents.dtype,
        )
        protected_latents = (
            (1.0 - next_sigma) * background_latents
            + next_sigma * fixed_noise
        )
        packed_protected = current_pipeline._pack_latents(protected_latents).to(
            device=current_latents.device,
            dtype=current_latents.dtype,
        )
        packed_mask = edit_mask_packed.to(
            device=current_latents.device,
            dtype=current_latents.dtype,
        )
        return {
            "latents": (
                packed_mask * current_latents
                + (1.0 - packed_mask) * packed_protected
            )
        }

    return Flux2KleinMaskedInpaintInputs(
        latents=latents,
        sigmas=tuple(float(value) for value in sigmas),
        callback_on_step_end=lock_background_callback,
        requested_strength=float(strength),
        effective_start_sigma=effective_start_sigma,
        denoising_steps=denoising_steps,
        used_init_image=init_image is not None,
    )


def prepare_flux2_klein_spatial_lock_inputs(
    pipeline: Any,
    init_image: Image.Image,
    protect_mask: Image.Image,
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    strength: float,
    generator: torch.Generator,
) -> Flux2KleinSpatialLockInputs:
    """Prepare FLUX img2img while retaining selected init-image structure.

    White mask values keep the correctly noised initialization latent after
    every denoising step; black values remain fully generative. Gray values
    softly mix both. Unlike the background-mask helper, this function never
    constructs or locks a flat-color canvas.
    """

    if width < 1 or height < 1:
        raise ValueError("spatial-lock output dimensions must be positive")
    if num_inference_steps < 2:
        raise ValueError("spatial-lock num_inference_steps must be at least 2")
    if not 0.0 < strength <= 1.0:
        raise ValueError("spatial-lock strength must lie in (0, 1]")
    scheduler_config = getattr(getattr(pipeline, "scheduler", None), "config", {})
    use_flow_sigmas = (
        scheduler_config.get("use_flow_sigmas", False)
        if isinstance(scheduler_config, dict)
        else getattr(scheduler_config, "use_flow_sigmas", False)
    )
    if use_flow_sigmas:
        raise ValueError("spatial lock requires a scheduler that accepts custom sigmas")

    device = pipeline._execution_device
    multiple_of = int(pipeline.vae_scale_factor) * 2
    normalized_width = max(multiple_of, (int(width) // multiple_of) * multiple_of)
    normalized_height = max(multiple_of, (int(height) // multiple_of) * multiple_of)
    processed = pipeline.image_processor.preprocess(
        init_image.convert("RGB"),
        height=normalized_height,
        width=normalized_width,
        resize_mode="crop",
    ).to(device=device, dtype=pipeline.vae.dtype)
    with torch.no_grad():
        clean_latents = pipeline._encode_vae_image(image=processed, generator=generator)

    latent_height, latent_width = clean_latents.shape[-2:]
    resized_mask = protect_mask.convert("L").resize(
        (latent_width, latent_height), Image.Resampling.LANCZOS
    )
    mask_array = np.asarray(resized_mask, dtype=np.float32) / 255.0
    spatial_mask = torch.from_numpy(mask_array).to(
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )[None, None, :, :]
    packed_mask = pipeline._pack_latents(
        spatial_mask.expand(-1, clean_latents.shape[1], -1, -1)
    )

    denoising_steps = max(2, min(num_inference_steps, math.ceil(num_inference_steps * strength)))
    sigmas = klein_custom_sigmas(num_inference_steps)[-denoising_steps:]
    image_seq_len = int(clean_latents.shape[-2] * clean_latents.shape[-1])
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    pipeline.scheduler.set_timesteps(sigmas=list(sigmas), device=device, mu=mu)
    effective_start_sigma = float(pipeline.scheduler.sigmas[0].item())
    fixed_noise = torch.randn(
        clean_latents.shape,
        generator=generator,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    latents = (
        (1.0 - effective_start_sigma) * clean_latents
        + effective_start_sigma * fixed_noise
    )

    def lock_structure_callback(
        current_pipeline: Any,
        step_index: int,
        _timestep: torch.Tensor,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        current_latents = callback_kwargs["latents"]
        scheduler_sigmas = current_pipeline.scheduler.sigmas
        next_index = min(step_index + 1, len(scheduler_sigmas) - 1)
        next_sigma = scheduler_sigmas[next_index].to(
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )
        protected = (1.0 - next_sigma) * clean_latents + next_sigma * fixed_noise
        packed_protected = current_pipeline._pack_latents(protected).to(
            device=current_latents.device,
            dtype=current_latents.dtype,
        )
        current_mask = packed_mask.to(
            device=current_latents.device,
            dtype=current_latents.dtype,
        )
        return {
            "latents": current_mask * packed_protected + (1.0 - current_mask) * current_latents
        }

    return Flux2KleinSpatialLockInputs(
        latents=latents,
        sigmas=tuple(float(value) for value in sigmas),
        callback_on_step_end=lock_structure_callback,
        requested_strength=float(strength),
        effective_start_sigma=effective_start_sigma,
        denoising_steps=denoising_steps,
    )


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
    "BackgroundEditMask",
    "Flux2KleinImg2ImgInputs",
    "Flux2KleinMaskedInpaintInputs",
    "Flux2KleinSpatialLockInputs",
    "IMAGE_SUFFIXES",
    "TrajectoryActivityGuide",
    "TrajectoryArchiveError",
    "composite_generated_activity",
    "composite_generated_on_background",
    "list_image_members",
    "make_background_edit_mask",
    "make_strong_trajectory_reference",
    "make_trajectory_activity_guide",
    "natural_sort_key",
    "prepare_flux2_klein_img2img_inputs",
    "prepare_flux2_klein_masked_inpaint_inputs",
    "prepare_flux2_klein_spatial_lock_inputs",
    "prepare_grayscale_edit_mask",
    "regular_sample_indices",
    "stage_regular_keyframes",
]
