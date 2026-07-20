"""Exact FLUX.2 Klein latent transformations.

The helpers in this module mirror ``Flux2KleinPipeline`` at the pinned
Diffusers revision without importing Diffusers.  This keeps the algebra easy
to unit test on CPU and, more importantly, avoids reusing FLUX.1 scaling
constants.  A real pipeline's image processor and VAE are accepted through
their small duck-typed interfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


Tensor = torch.Tensor
PatchSize = int | tuple[int, int]


def _pair(value: PatchSize) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"patch_size must contain two positive integers, got {value!r}")
    return int(pair[0]), int(pair[1])


def _loaded_patch_size(vae: Any, patch_size: PatchSize | None) -> PatchSize:
    if patch_size is not None:
        return patch_size
    config = getattr(vae, "config", None)
    configured = getattr(config, "patch_size", None)
    return configured if configured is not None else 2


def _module_device_dtype(
    module: Any,
    *,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    device = getattr(module, "device", None)
    dtype = getattr(module, "dtype", None)
    parameters = getattr(module, "parameters", None)
    if (not isinstance(device, torch.device) or not isinstance(dtype, torch.dtype)) and callable(
        parameters
    ):
        for parameter in parameters():
            if not isinstance(device, torch.device):
                device = parameter.device
            if not isinstance(dtype, torch.dtype) and parameter.is_floating_point():
                dtype = parameter.dtype
            break
    return (
        device if isinstance(device, torch.device) else fallback_device,
        dtype if isinstance(dtype, torch.dtype) else fallback_dtype,
    )


def preprocess_endpoint_image(
    image: Any,
    image_processor: Any | None = None,
    *,
    height: int = 512,
    width: int = 512,
    resize_mode: str = "default",
) -> Tensor:
    """Convert an endpoint image to the VAE's normalized BCHW tensor.

    When a ``Flux2ImageProcessor`` is supplied, its public ``preprocess``
    method is used directly.  That is the production path and therefore
    inherits the pinned pipeline's RGB conversion, Lanczos resize, and
    ``[-1, 1]`` normalization exactly.  The dependency-free fallback is for
    CPU tests and already-resized endpoint assets.
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if image_processor is not None:
        result = image_processor.preprocess(
            image,
            height=height,
            width=width,
            resize_mode=resize_mode,
        )
        if not isinstance(result, torch.Tensor) or result.ndim != 4:
            raise TypeError("image_processor.preprocess() must return a BCHW torch.Tensor")
        return result

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            pil_image = opened.copy()
    elif isinstance(image, Image.Image):
        pil_image = image.copy()
    elif isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("tensor images must have shape (B, 3, H, W) or (3, H, W)")
        if tensor.shape[-2:] != (height, width):
            raise ValueError(
                "the no-Diffusers tensor fallback does not resize tensors; "
                f"expected {(height, width)}, got {tuple(tensor.shape[-2:])}"
            )
        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() and tensor.amin() >= 0 and tensor.amax() <= 1:
            tensor = tensor * 2.0 - 1.0
        return tensor
    else:
        raise TypeError(
            "image must be a PIL image, path, or tensor when no image processor is supplied"
        )

    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    if resize_mode in {"default", "stretch"}:
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    elif resize_mode == "crop":
        left = (pil_image.width - width) // 2
        top = (pil_image.height - height) // 2
        pil_image = pil_image.crop((left, top, left + width, top + height))
    else:
        raise ValueError("fallback resize_mode must be 'default', 'stretch', or 'crop'")

    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def _retrieve_deterministic_latent(encoder_output: Any) -> Tensor:
    """Reproduce Diffusers ``retrieve_latents(..., sample_mode='argmax')``."""

    if hasattr(encoder_output, "latent_dist"):
        latent_dist = encoder_output.latent_dist
        if not hasattr(latent_dist, "mode"):
            raise AttributeError("VAE latent_dist does not expose deterministic mode()")
        latent = latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        latent = encoder_output.latents
    elif isinstance(encoder_output, torch.Tensor):
        latent = encoder_output
    else:
        raise AttributeError("could not access latents from the VAE encoder output")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError("VAE encoder must produce a four-dimensional BCHW latent tensor")
    return latent


def encode_image_to_vae_latent(
    image: Tensor,
    vae: Any,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Encode an image using the posterior mode, never a random sample."""

    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        ndim = getattr(image, "ndim", None)
        raise ValueError(f"expected a four-dimensional BCHW image tensor, got ndim={ndim}")
    # ``generator`` is accepted for call-site parity, but deliberately unused:
    # the pinned pipeline requests latent_dist.mode() ("argmax"), not sample().
    del generator
    return _retrieve_deterministic_latent(vae.encode(image))


def patchify_latent(latent: Tensor, patch_size: PatchSize = 2) -> Tensor:
    """Move each spatial patch into channels using the FLUX.2 ordering."""

    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got shape {tuple(latent.shape)}")
    patch_height, patch_width = _pair(patch_size)
    batch, channels, height, width = latent.shape
    if height % patch_height or width % patch_width:
        raise ValueError(
            f"latent spatial shape {(height, width)} is not divisible by patch size "
            f"{(patch_height, patch_width)}"
        )
    latent = latent.reshape(
        batch,
        channels,
        height // patch_height,
        patch_height,
        width // patch_width,
        patch_width,
    )
    latent = latent.permute(0, 1, 3, 5, 2, 4)
    return latent.reshape(
        batch,
        channels * patch_height * patch_width,
        height // patch_height,
        width // patch_width,
    )


def unpatchify_latent(latent: Tensor, patch_size: PatchSize = 2) -> Tensor:
    """Invert :func:`patchify_latent` without assuming a channel count."""

    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got shape {tuple(latent.shape)}")
    patch_height, patch_width = _pair(patch_size)
    batch, patched_channels, height, width = latent.shape
    patch_area = patch_height * patch_width
    if patched_channels % patch_area:
        raise ValueError(
            f"patched channel count {patched_channels} is not divisible by patch area {patch_area}"
        )
    channels = patched_channels // patch_area
    latent = latent.reshape(
        batch,
        channels,
        patch_height,
        patch_width,
        height,
        width,
    )
    latent = latent.permute(0, 1, 4, 2, 5, 3)
    return latent.reshape(batch, channels, height * patch_height, width * patch_width)


def _batch_norm_statistics(
    vae_or_batch_norm: Any | None,
    *,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    eps: float | None,
) -> tuple[Tensor, Tensor, float]:
    source = vae_or_batch_norm
    batch_norm = getattr(source, "bn", source)
    if running_mean is None:
        running_mean = getattr(batch_norm, "running_mean", None)
    if running_var is None:
        running_var = getattr(batch_norm, "running_var", None)
    if eps is None:
        config = getattr(source, "config", None)
        eps = getattr(config, "batch_norm_eps", None)
    if eps is None:
        eps = getattr(batch_norm, "eps", None)
    if not isinstance(running_mean, torch.Tensor) or not isinstance(running_var, torch.Tensor):
        raise ValueError("VAE batch-normalization running_mean and running_var are required")
    if running_mean.ndim != 1 or running_var.ndim != 1 or running_mean.shape != running_var.shape:
        raise ValueError("running_mean and running_var must be equal-length one-dimensional tensors")
    if eps is None or float(eps) < 0:
        raise ValueError("a non-negative batch-normalization epsilon is required")
    return running_mean, running_var, float(eps)


def normalize_vae_latent(
    latent: Tensor,
    vae_or_batch_norm: Any | None = None,
    *,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    eps: float | None = None,
) -> Tensor:
    """Apply the loaded Flux2 VAE BN statistics to a patchified latent."""

    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got shape {tuple(latent.shape)}")
    mean, variance, epsilon = _batch_norm_statistics(
        vae_or_batch_norm,
        running_mean=running_mean,
        running_var=running_var,
        eps=eps,
    )
    if mean.numel() != latent.shape[1]:
        raise ValueError(
            f"BN statistics contain {mean.numel()} channels but latent has {latent.shape[1]}"
        )
    # The pinned pipeline takes sqrt before casting the running variance.  This
    # matters for BF16 parity and is intentionally kept in the same order.
    mean = mean.reshape(1, -1, 1, 1).to(device=latent.device, dtype=latent.dtype)
    std = torch.sqrt(variance.reshape(1, -1, 1, 1) + epsilon).to(
        device=latent.device,
        dtype=latent.dtype,
    )
    return (latent - mean) / std


def denormalize_vae_latent(
    latent: Tensor,
    vae_or_batch_norm: Any | None = None,
    *,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    eps: float | None = None,
) -> Tensor:
    """Invert :func:`normalize_vae_latent` using the same loaded statistics."""

    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got shape {tuple(latent.shape)}")
    mean, variance, epsilon = _batch_norm_statistics(
        vae_or_batch_norm,
        running_mean=running_mean,
        running_var=running_var,
        eps=eps,
    )
    if mean.numel() != latent.shape[1]:
        raise ValueError(
            f"BN statistics contain {mean.numel()} channels but latent has {latent.shape[1]}"
        )
    mean = mean.reshape(1, -1, 1, 1).to(device=latent.device, dtype=latent.dtype)
    std = torch.sqrt(variance.reshape(1, -1, 1, 1) + epsilon).to(
        device=latent.device,
        dtype=latent.dtype,
    )
    return latent * std + mean


def pack_latent_tokens(latent: Tensor) -> Tensor:
    """Pack ``(B, C, H, W)`` into ``(B, H*W, C)``."""

    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got shape {tuple(latent.shape)}")
    batch, channels, height, width = latent.shape
    return latent.reshape(batch, channels, height * width).permute(0, 2, 1)


def make_image_ids(
    latent: Tensor | None = None,
    *,
    batch_size: int | None = None,
    height: int | None = None,
    width: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create FLUX.2 ``(T,H,W,L)`` position IDs for packed latent tokens."""

    if latent is not None:
        if latent.ndim != 4:
            raise ValueError(f"expected BCHW latent, got shape {tuple(latent.shape)}")
        inferred_batch, _, inferred_height, inferred_width = latent.shape
        batch_size = inferred_batch if batch_size is None else batch_size
        height = inferred_height if height is None else height
        width = inferred_width if width is None else width
        device = latent.device if device is None else device
    if batch_size is None or height is None or width is None:
        raise ValueError("provide a BCHW latent or explicit batch_size, height, and width")
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("batch_size, height, and width must be positive")

    rows = torch.arange(height, device=device, dtype=torch.int64)
    columns = torch.arange(width, device=device, dtype=torch.int64)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    zeros = torch.zeros(height * width, device=device, dtype=torch.int64)
    ids = torch.stack(
        (
            zeros,
            row_grid.reshape(-1),
            column_grid.reshape(-1),
            zeros,
        ),
        dim=-1,
    )
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def unpack_latent_tokens(
    tokens: Tensor,
    image_ids: Tensor | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
) -> Tensor:
    """Unpack tokens, optionally scattering them according to image IDs."""

    if tokens.ndim != 3:
        raise ValueError(f"expected (B, N, C) tokens, got shape {tuple(tokens.shape)}")
    batch, token_count, channels = tokens.shape

    if image_ids is None:
        if height is None or width is None:
            raise ValueError("height and width are required when image_ids are omitted")
        if height * width != token_count:
            raise ValueError(
                f"height*width is {height * width}, but token count is {token_count}"
            )
        return tokens.permute(0, 2, 1).reshape(batch, channels, height, width)

    if image_ids.ndim != 3 or image_ids.shape != (batch, token_count, 4):
        raise ValueError(
            "image_ids must have shape (batch, token_count, 4), got "
            f"{tuple(image_ids.shape)}"
        )
    unpacked: list[Tensor] = []
    for data, positions in zip(tokens, image_ids):
        row_ids = positions[:, 1].to(device=data.device, dtype=torch.int64)
        column_ids = positions[:, 2].to(device=data.device, dtype=torch.int64)
        current_height = height
        current_width = width
        if current_height is None:
            current_height = int(row_ids.max().item()) + 1
        if current_width is None:
            current_width = int(column_ids.max().item()) + 1
        if current_height <= 0 or current_width <= 0:
            raise ValueError("height and width inferred from image_ids must be positive")
        if token_count != current_height * current_width:
            raise ValueError(
                f"image IDs describe {(current_height, current_width)} but contain {token_count} tokens"
            )
        if (
            torch.any(row_ids < 0)
            or torch.any(row_ids >= current_height)
            or torch.any(column_ids < 0)
            or torch.any(column_ids >= current_width)
        ):
            raise ValueError("image_ids contain out-of-bounds spatial coordinates")

        flat_ids = row_ids * current_width + column_ids
        output = torch.zeros(
            (current_height * current_width, channels),
            device=data.device,
            dtype=data.dtype,
        )
        output.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        unpacked.append(output.reshape(current_height, current_width, channels).permute(2, 0, 1))
    return torch.stack(unpacked, dim=0)


def decode_vae_latent(
    latent: Tensor,
    vae: Any,
    *,
    image_processor: Any | None = None,
    output_type: str = "pt",
    postprocess: bool = True,
) -> Any:
    """Decode a raw VAE latent and optionally apply pipeline postprocessing."""

    decoded_output = vae.decode(latent, return_dict=False)
    if isinstance(decoded_output, (tuple, list)):
        decoded = decoded_output[0]
    elif hasattr(decoded_output, "sample"):
        decoded = decoded_output.sample
    else:
        decoded = decoded_output
    if not isinstance(decoded, torch.Tensor):
        raise TypeError("vae.decode() did not return a tensor sample")
    if not bool(torch.isfinite(decoded).all().item()):
        raise FloatingPointError(
            "VAE decode produced non-finite pixels before image postprocessing"
        )
    if not postprocess:
        return decoded
    if image_processor is not None:
        return image_processor.postprocess(decoded, output_type=output_type)
    if output_type != "pt":
        raise ValueError("an image_processor is required for non-'pt' output")
    return (decoded * 0.5 + 0.5).clamp(0.0, 1.0)


def encode_image_to_packed_latent(
    image: Any,
    vae: Any,
    *,
    image_processor: Any | None = None,
    height: int = 512,
    width: int = 512,
    resize_mode: str = "default",
    patch_size: PatchSize | None = None,
    preprocessed: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run deterministic FLUX.2 image encoding and return tokens plus IDs."""

    image_tensor = (
        image
        if preprocessed
        else preprocess_endpoint_image(
            image,
            image_processor,
            height=height,
            width=width,
            resize_mode=resize_mode,
        )
    )
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("preprocessed image must be a torch.Tensor")
    vae_device, vae_dtype = _module_device_dtype(
        vae,
        fallback_device=image_tensor.device,
        fallback_dtype=image_tensor.dtype,
    )
    image_tensor = image_tensor.to(device=vae_device, dtype=vae_dtype)
    vae_latent = encode_image_to_vae_latent(image_tensor, vae)
    patchified = patchify_latent(
        vae_latent,
        patch_size=_loaded_patch_size(vae, patch_size),
    )
    normalized = normalize_vae_latent(patchified, vae)
    image_ids = make_image_ids(normalized)
    return pack_latent_tokens(normalized), image_ids


def decode_packed_latent(
    tokens: Tensor,
    image_ids: Tensor,
    vae: Any,
    *,
    height: int | None = None,
    width: int | None = None,
    patch_size: PatchSize | None = None,
    image_processor: Any | None = None,
    output_type: str = "pt",
    postprocess: bool = True,
) -> Any:
    """Reverse FLUX.2 packing, BN normalization, patching, and VAE decode."""

    packed_spatial = unpack_latent_tokens(
        tokens,
        image_ids,
        height=height,
        width=width,
    )
    denormalized = denormalize_vae_latent(packed_spatial, vae)
    vae_latent = unpatchify_latent(
        denormalized,
        patch_size=_loaded_patch_size(vae, patch_size),
    )
    vae_device, vae_dtype = _module_device_dtype(
        vae,
        fallback_device=vae_latent.device,
        fallback_dtype=vae_latent.dtype,
    )
    vae_latent = vae_latent.to(device=vae_device, dtype=vae_dtype)
    return decode_vae_latent(
        vae_latent,
        vae,
        image_processor=image_processor,
        output_type=output_type,
        postprocess=postprocess,
    )
