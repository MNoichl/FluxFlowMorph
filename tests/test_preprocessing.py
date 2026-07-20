from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageOps

from flowmorph_klein.errors import ImagePreprocessingError
from flowmorph_klein.image_io import (
    preprocess_endpoint_image,
    preprocess_endpoint_pair,
)
from flowmorph_klein.types import ResizeMode


def test_stretch_converts_rgb_and_has_exact_dimensions() -> None:
    image = Image.new("L", (2, 3), 127)
    result = preprocess_endpoint_image(
        image,
        width=8,
        height=6,
        resize_mode="stretch",
        resample=Image.Resampling.NEAREST,
    )
    assert result.image.mode == "RGB"
    assert result.original_size == (2, 3)
    assert result.processed_size == (8, 6)
    assert result.image.getpixel((0, 0)) == (127, 127, 127)


def test_center_crop_preserves_center_content() -> None:
    image = Image.new("RGB", (4, 2))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for x, color in enumerate(colors):
        for y in range(2):
            image.putpixel((x, y), color)

    result = preprocess_endpoint_image(
        image,
        width=2,
        height=2,
        resize_mode=ResizeMode.CENTER_CROP,
        resample=Image.Resampling.NEAREST,
    )
    assert result.image.getpixel((0, 0)) == (0, 255, 0)
    assert result.image.getpixel((1, 0)) == (0, 0, 255)


def test_contain_and_pad_is_centered_and_deterministic() -> None:
    image = Image.new("RGB", (4, 2), (255, 0, 0))
    result = preprocess_endpoint_image(
        image,
        width=4,
        height=4,
        resize_mode="contain_and_pad",
        resample=Image.Resampling.NEAREST,
        pad_color=(1, 2, 3),
    )
    assert result.image.getpixel((0, 0)) == (1, 2, 3)
    assert result.image.getpixel((0, 1)) == (255, 0, 0)
    assert result.image.getpixel((3, 2)) == (255, 0, 0)
    assert result.image.getpixel((3, 3)) == (1, 2, 3)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    original = Image.new("RGB", (2, 3))
    original.putpixel((0, 0), (255, 0, 0))
    original.putpixel((1, 0), (0, 255, 0))
    original.putpixel((0, 2), (0, 0, 255))
    original.putpixel((1, 2), (255, 255, 0))
    exif = original.getexif()
    exif[274] = 6  # rotate 90 degrees clockwise for display
    path = tmp_path / "oriented.jpg"
    original.save(path, format="JPEG", quality=100, subsampling=0, exif=exif)

    with Image.open(path) as opened:
        expected = ImageOps.exif_transpose(opened).convert("RGB")
        expected.load()
    result = preprocess_endpoint_image(
        path,
        width=3,
        height=2,
        resample=Image.Resampling.NEAREST,
    )
    assert result.original_size == (2, 3)
    assert result.image.tobytes() == expected.tobytes()


def test_alpha_conversion_matches_recorded_pillow_rgb_policy() -> None:
    image = Image.new("RGBA", (1, 1), (200, 10, 20, 0))
    result = preprocess_endpoint_image(image, width=1, height=1)
    assert result.image.getpixel((0, 0)) == (200, 10, 20)


def test_preprocessing_and_saved_png_are_reproducible(tmp_path: Path) -> None:
    source = Image.new("RGB", (7, 5), (20, 40, 60))
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = preprocess_endpoint_image(source, width=16, height=16, output_path=first_path)
    second = preprocess_endpoint_image(source, width=16, height=16, output_path=second_path)

    assert first.image.tobytes() == second.image.tobytes()
    assert first.preprocessing_sha256 == second.preprocessing_sha256
    assert first_path.read_bytes() == second_path.read_bytes()


def test_pair_preprocessing_uses_identical_dimensions_and_names(tmp_path: Path) -> None:
    source = Image.new("RGB", (10, 4), "red")
    target = Image.new("RGB", (3, 12), "blue")
    first, second = preprocess_endpoint_pair(
        source,
        target,
        width=32,
        height=32,
        resize_mode="center_crop",
        output_directory=tmp_path,
    )
    assert first.processed_size == second.processed_size == (32, 32)
    assert first.output_path == (tmp_path / "source_preprocessed.png").resolve()
    assert second.output_path == (tmp_path / "target_preprocessed.png").resolve()
    assert first.output_path.is_file()
    assert second.output_path.is_file()


def test_corrupt_image_and_invalid_divisibility_fail_clearly(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ImagePreprocessingError, match="cannot decode"):
        preprocess_endpoint_image(corrupt)

    with pytest.raises(ImagePreprocessingError, match="not divisible"):
        preprocess_endpoint_image(
            Image.new("RGB", (2, 2)),
            width=510,
            height=512,
            divisibility=16,
        )
