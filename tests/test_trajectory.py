from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flowmorph_klein.trajectory import (
    TrajectoryArchiveError,
    list_image_members,
    make_strong_trajectory_reference,
    prepare_flux2_klein_img2img_inputs,
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


class _FakeImageProcessor:
    def preprocess(self, image, *, height, width, resize_mode):
        assert image.mode == "RGB"
        assert (width, height, resize_mode) == (32, 32, "crop")
        return torch.zeros((1, 3, height, width), dtype=torch.float32)


class _FakeScheduler:
    config = {"use_flow_sigmas": False}

    def set_timesteps(self, *, sigmas, device, mu):
        self.requested_sigmas = tuple(sigmas)
        self.device = device
        self.mu = mu
        self.sigmas = torch.tensor([0.4, *sigmas[1:], 0.0], device=device)


class _FakeVae:
    dtype = torch.float32


class _FakePipeline:
    _execution_device = torch.device("cpu")
    vae_scale_factor = 8
    image_processor = _FakeImageProcessor()
    scheduler = _FakeScheduler()
    vae = _FakeVae()

    @staticmethod
    def _encode_vae_image(*, image, generator):
        assert image.shape == (1, 3, 32, 32)
        assert generator is not None
        return torch.ones((1, 4, 2, 3), dtype=torch.float32)


def test_true_img2img_inputs_use_encoded_image_and_truncated_sigmas() -> None:
    pipeline = _FakePipeline()
    result = prepare_flux2_klein_img2img_inputs(
        pipeline,
        Image.new("RGB", (32, 32), (20, 40, 80)),
        width=32,
        height=32,
        num_inference_steps=20,
        strength=0.15,
        generator=torch.Generator(device="cpu").manual_seed(123),
    )

    assert result.denoising_steps == 3
    assert len(result.sigmas) == 3
    assert result.sigmas == pipeline.scheduler.requested_sigmas
    assert result.effective_start_sigma == pytest.approx(0.4)
    assert result.latents.shape == (1, 4, 2, 3)
    assert not torch.equal(result.latents, torch.ones_like(result.latents))


@pytest.mark.parametrize("strength", [0.0, 1.01])
def test_true_img2img_inputs_reject_invalid_strength(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        prepare_flux2_klein_img2img_inputs(
            _FakePipeline(),
            Image.new("RGB", (32, 32)),
            width=32,
            height=32,
            num_inference_steps=20,
            strength=strength,
            generator=torch.Generator(device="cpu"),
        )
