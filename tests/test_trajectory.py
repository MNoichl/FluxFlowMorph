from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flowmorph_klein.trajectory import (
    TrajectoryArchiveError,
    list_image_members,
    make_strong_trajectory_reference,
    regular_sample_indices,
    stage_regular_keyframes,
)


def image_bytes(color: tuple[int, int, int], size: tuple[int, int] = (12, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_archive(path: Path, count: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for index in reversed(range(count)):
            archive.writestr(
                f"frames/frame_{index}.png",
                image_bytes((index, 0, 0)),
            )
        archive.writestr("__MACOSX/._frame.png", b"ignored")


def test_regular_sampling_matches_requested_example() -> None:
    assert regular_sample_indices(180, 18) == tuple(range(0, 180, 10))
    assert regular_sample_indices(10, 3) == (0, 3, 6)


def test_regular_sampling_rejects_more_keyframes_than_images() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        regular_sample_indices(3, 4)


def test_archive_members_are_naturally_sorted_and_staged(tmp_path: Path) -> None:
    archive_path = tmp_path / "trajectory.zip"
    write_archive(archive_path, 12)

    members = list_image_members(archive_path)
    records = stage_regular_keyframes(
        archive_path,
        tmp_path / "selected",
        keyframe_count=3,
        width=16,
        height=16,
    )

    assert members[2].endswith("frame_2.png")
    assert [record["trajectory_index"] for record in records] == [0, 4, 8]
    assert [record["member"] for record in records] == [
        "frames/frame_0.png",
        "frames/frame_4.png",
        "frames/frame_8.png",
    ]
    assert all(Image.open(record["path"]).size == (16, 16) for record in records)


def test_archive_offset_and_reverse_are_applied_before_sampling(tmp_path: Path) -> None:
    archive_path = tmp_path / "trajectory.zip"
    write_archive(archive_path, 10)

    records = stage_regular_keyframes(
        archive_path,
        tmp_path / "selected",
        keyframe_count=2,
        width=8,
        height=8,
        frame_offset=1,
        reverse_order=True,
    )

    assert [record["member"] for record in records] == [
        "frames/frame_8.png",
        "frames/frame_3.png",
    ]


def test_archive_rejects_unsafe_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.png", image_bytes((0, 0, 0)))

    with pytest.raises(TrajectoryArchiveError, match="escapes"):
        list_image_members(archive_path)


def test_strong_reference_preserves_color_and_seeded_grain() -> None:
    source = Image.new("RGB", (16, 16), (180, 70, 20))
    exact = make_strong_trajectory_reference(source)
    first = make_strong_trajectory_reference(
        source,
        detail_strength=0.5,
        blur_radius=3.0,
        grain_strength=0.01,
        grain_seed=42,
    )
    repeated = make_strong_trajectory_reference(
        source,
        detail_strength=0.5,
        blur_radius=3.0,
        grain_strength=0.01,
        grain_seed=42,
    )

    assert np.array_equal(np.asarray(exact), np.asarray(source))
    assert np.array_equal(np.asarray(first), np.asarray(repeated))
    assert np.allclose(
        np.asarray(first, dtype=np.float32).mean(axis=(0, 1)),
        (180, 70, 20),
        atol=2.0,
    )
