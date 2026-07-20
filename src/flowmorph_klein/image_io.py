"""Deterministic endpoint image decoding and spatial preprocessing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import TypeAlias

from PIL import Image, ImageOps, UnidentifiedImageError, __version__ as pillow_version

from .colab_io import sha256_file
from .errors import ImagePreprocessingError
from .types import PreprocessedImage, ResizeMode


ImageSource: TypeAlias = str | Path | Image.Image


def _validate_target_geometry(
    width: int,
    height: int,
    divisibility: int | tuple[int, int] | None,
) -> None:
    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        raise ImagePreprocessingError("processed image dimensions must be positive integers")
    if divisibility is None:
        return
    if isinstance(divisibility, int):
        divisors = (divisibility, divisibility)
    elif len(divisibility) == 2:
        divisors = divisibility
    else:
        raise ImagePreprocessingError("divisibility must be an integer or a width/height pair")
    if any(isinstance(value, bool) or value <= 0 for value in divisors):
        raise ImagePreprocessingError("divisibility values must be positive integers")
    if width % divisors[0] or height % divisors[1]:
        raise ImagePreprocessingError(
            f"target {width}x{height} is not divisible by required geometry "
            f"{divisors[0]}x{divisors[1]}"
        )


def _in_memory_image_hash(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("utf-8"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    try:
        digest.update(image.getexif().tobytes())
    except (AttributeError, TypeError, ValueError):
        pass
    return digest.hexdigest()


def _load_image(source: ImageSource) -> tuple[Image.Image, Path | None, str | None]:
    if isinstance(source, Image.Image):
        try:
            source.load()
            image = source.copy()
        except (OSError, ValueError) as error:
            raise ImagePreprocessingError(f"cannot read in-memory endpoint image: {error}") from error
        return image, None, _in_memory_image_hash(source)

    source_path = Path(source).expanduser().resolve(strict=False)
    if not source_path.is_file():
        raise ImagePreprocessingError(f"endpoint image is not a readable file: {source_path}")
    original_hash = sha256_file(source_path)
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = opened.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise ImagePreprocessingError(
            f"cannot decode endpoint image {source_path}: {error}"
        ) from error
    return image, source_path, original_hash


def _convert_rgb_with_exif_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and match Pillow/Diffusers ``convert('RGB')``.

    For images with alpha, Pillow's RGB conversion deterministically discards
    the alpha channel while retaining the stored RGB components.  Recording
    this policy in the preprocessing hash prevents incompatible resume.
    """

    try:
        oriented = ImageOps.exif_transpose(image)
        return oriented.convert("RGB")
    except (OSError, ValueError) as error:
        raise ImagePreprocessingError(f"cannot orient or convert endpoint image: {error}") from error


def _resize_center_crop(
    image: Image.Image,
    width: int,
    height: int,
    resample: Image.Resampling,
) -> Image.Image:
    source_width, source_height = image.size
    if source_width * height >= source_height * width:
        resized_height = height
        resized_width = math.ceil(source_width * height / source_height)
    else:
        resized_width = width
        resized_height = math.ceil(source_height * width / source_width)
    resized = image.resize((resized_width, resized_height), resample=resample)
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _round_ratio(numerator: int, denominator: int) -> int:
    return max(1, (numerator + denominator // 2) // denominator)


def _resize_contain_and_pad(
    image: Image.Image,
    width: int,
    height: int,
    resample: Image.Resampling,
    pad_color: tuple[int, int, int],
) -> Image.Image:
    source_width, source_height = image.size
    if source_width * height >= source_height * width:
        resized_width = width
        resized_height = min(height, _round_ratio(source_height * width, source_width))
    else:
        resized_height = height
        resized_width = min(width, _round_ratio(source_width * height, source_height))
    resized = image.resize((resized_width, resized_height), resample=resample)
    canvas = Image.new("RGB", (width, height), pad_color)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def resize_endpoint_image(
    image: Image.Image,
    *,
    width: int,
    height: int,
    resize_mode: ResizeMode | str,
    resample: Image.Resampling | int = Image.Resampling.LANCZOS,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize an RGB image according to an explicit deterministic policy."""

    try:
        mode = ResizeMode(resize_mode)
    except ValueError as error:
        raise ImagePreprocessingError(f"unsupported resize mode {resize_mode!r}") from error
    try:
        sampling = Image.Resampling(resample)
    except ValueError as error:
        raise ImagePreprocessingError(f"unsupported Pillow resampling value {resample!r}") from error
    if len(pad_color) != 3 or any(
        isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
        for channel in pad_color
    ):
        raise ImagePreprocessingError("pad_color must contain three integer RGB values")
    if image.mode != "RGB":
        raise ImagePreprocessingError("resize_endpoint_image expects an RGB image")

    if mode is ResizeMode.STRETCH:
        return image.resize((width, height), resample=sampling)
    if mode is ResizeMode.CENTER_CROP:
        return _resize_center_crop(image, width, height, sampling)
    return _resize_contain_and_pad(image, width, height, sampling, pad_color)


def _preprocessing_hash(
    *,
    original_sha256: str,
    width: int,
    height: int,
    resize_mode: ResizeMode,
    resample: Image.Resampling,
    pad_color: tuple[int, int, int],
) -> str:
    payload = {
        "source_sha256": original_sha256,
        "exif_orientation": "pillow_exif_transpose",
        "rgb_conversion": "pillow_convert_rgb_discard_alpha",
        "width": width,
        "height": height,
        "resize_mode": resize_mode.value,
        "resample": resample.name,
        "pad_color": list(pad_color),
        "pillow_version": pillow_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_preprocessed_image(image: Image.Image, output_path: str | Path) -> Path:
    """Atomically write a metadata-free RGB PNG."""

    destination = Path(output_path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            image.save(handle, format="PNG", optimize=False, compress_level=9)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except (OSError, ValueError) as error:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise ImagePreprocessingError(
            f"cannot save preprocessed endpoint image {destination}: {error}"
        ) from error
    return destination


def preprocess_endpoint_image(
    source: ImageSource,
    *,
    width: int = 512,
    height: int = 512,
    resize_mode: ResizeMode | str = ResizeMode.STRETCH,
    output_path: str | Path | None = None,
    divisibility: int | tuple[int, int] | None = None,
    resample: Image.Resampling | int = Image.Resampling.LANCZOS,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> PreprocessedImage:
    """Decode, orient, convert, resize, optionally save, and fingerprint an endpoint."""

    _validate_target_geometry(width, height, divisibility)
    try:
        mode = ResizeMode(resize_mode)
        sampling = Image.Resampling(resample)
    except ValueError as error:
        raise ImagePreprocessingError(f"invalid preprocessing option: {error}") from error

    loaded, source_path, original_hash = _load_image(source)
    original_size = loaded.size
    converted = _convert_rgb_with_exif_orientation(loaded)
    processed = resize_endpoint_image(
        converted,
        width=width,
        height=height,
        resize_mode=mode,
        resample=sampling,
        pad_color=pad_color,
    )
    if processed.size != (width, height) or processed.mode != "RGB":
        raise ImagePreprocessingError("internal preprocessing produced invalid geometry or mode")

    saved_path = save_preprocessed_image(processed, output_path) if output_path else None
    if original_hash is None:  # defensive: every supported source produces a fingerprint
        original_hash = _in_memory_image_hash(loaded)
    return PreprocessedImage(
        image=processed,
        source_path=source_path,
        output_path=saved_path,
        original_size=original_size,
        processed_size=processed.size,
        resize_mode=mode,
        original_sha256=original_hash,
        preprocessing_sha256=_preprocessing_hash(
            original_sha256=original_hash,
            width=width,
            height=height,
            resize_mode=mode,
            resample=sampling,
            pad_color=pad_color,
        ),
    )


def preprocess_endpoint_pair(
    source: ImageSource,
    target: ImageSource,
    *,
    width: int = 512,
    height: int = 512,
    resize_mode: ResizeMode | str = ResizeMode.STRETCH,
    output_directory: str | Path | None = None,
    divisibility: int | tuple[int, int] | None = None,
    resample: Image.Resampling | int = Image.Resampling.LANCZOS,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[PreprocessedImage, PreprocessedImage]:
    """Preprocess source and target with one shared deterministic policy."""

    output_root = (
        Path(output_directory).expanduser().resolve(strict=False)
        if output_directory is not None
        else None
    )
    source_result = preprocess_endpoint_image(
        source,
        width=width,
        height=height,
        resize_mode=resize_mode,
        output_path=output_root / "source_preprocessed.png" if output_root else None,
        divisibility=divisibility,
        resample=resample,
        pad_color=pad_color,
    )
    target_result = preprocess_endpoint_image(
        target,
        width=width,
        height=height,
        resize_mode=resize_mode,
        output_path=output_root / "target_preprocessed.png" if output_root else None,
        divisibility=divisibility,
        resample=resample,
        pad_color=pad_color,
    )
    if source_result.processed_size != target_result.processed_size:
        raise ImagePreprocessingError("source and target preprocessing dimensions differ")
    return source_result, target_result
