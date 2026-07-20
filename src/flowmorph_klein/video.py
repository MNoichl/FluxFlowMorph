"""Animation and broadly compatible MP4 export."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def _frame_arrays(frames: Sequence[Image.Image]) -> list[np.ndarray]:
    if not frames:
        raise ValueError("Animation requires at least one frame")
    images = [frame.convert("RGB") for frame in frames]
    if len({frame.size for frame in images}) != 1:
        raise ValueError("All animation frames must have identical dimensions")
    return [np.asarray(frame, dtype=np.uint8) for frame in images]


def _held_frames(arrays: list[np.ndarray], hold_frames: int) -> list[np.ndarray]:
    if hold_frames < 0:
        raise ValueError("hold_frames cannot be negative")
    if not hold_frames:
        return arrays
    return [arrays[0]] * hold_frames + arrays + [arrays[-1]] * hold_frames


def save_webp(
    frames: Sequence[Image.Image],
    output_path: str | Path,
    *,
    fps: float = 12,
    hold_frames: int = 0,
) -> Path:
    arrays = _held_frames(_frame_arrays(frames), hold_frames)
    images = [Image.fromarray(array) for array in arrays]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        lossless=True,
        method=4,
    )
    return output


def save_gif(
    frames: Sequence[Image.Image],
    output_path: str | Path,
    *,
    fps: float = 12,
    hold_frames: int = 0,
) -> Path:
    arrays = _held_frames(_frame_arrays(frames), hold_frames)
    images = [Image.fromarray(array) for array in arrays]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output


def save_mp4(
    frames: Sequence[Image.Image],
    output_path: str | Path,
    *,
    fps: float = 12,
    hold_frames: int = 0,
) -> Path:
    arrays = _held_frames(_frame_arrays(frames), hold_frames)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip(arrays, fps=fps)
        clip.write_videofile(
            str(output),
            codec="libx264",
            audio=False,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            logger=None,
        )
        clip.close()
    except ImportError as exc:
        raise RuntimeError("moviepy==1.0.3 and imageio-ffmpeg are required for MP4 export") from exc
    return output


def export_previews(
    frames: Sequence[Image.Image],
    output_directory: str | Path,
    *,
    fps: float = 12,
    hold_frames: int = 0,
) -> dict[str, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "webp": save_webp(frames, directory / "preview.webp", fps=fps, hold_frames=hold_frames),
        "gif": save_gif(frames, directory / "preview.gif", fps=fps, hold_frames=hold_frames),
        "mp4": save_mp4(frames, directory / "morph.mp4", fps=fps, hold_frames=hold_frames),
    }

