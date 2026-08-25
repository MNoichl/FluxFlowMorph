"""Build the standalone local video-directory stitcher notebook once."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Local_Video_Directory_Stitcher.ipynb"


def lines(value: str) -> list[str]:
    return dedent(value).strip("\n").splitlines(keepends=True)


def markdown(cell_id: str, value: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines(value),
    }


def code(cell_id: str, value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


notebook = {
    "cells": [
        markdown(
            "stitch-00-title",
            r"""
            # Local video-directory stitcher

            Point this notebook at a local directory containing only the clips you want. It scans
            that directory, natural-sorts the filenames, prints the exact order, and joins them into
            one MP4 with FFmpeg stream-copy.

            This notebook performs **no RIFE, interpolation, resizing, upscaling, filtering, or
            re-encoding**. The input clips therefore need compatible video/audio stream parameters.
            Name them `01_first.mp4`, `02_second.mp4`, and so on to control the order.
            """,
        ),
        markdown(
            "stitch-01-settings-heading",
            "## 1. Point at the directory",
        ),
        code(
            "stitch-02-settings",
            r'''
            VIDEO_DIRECTORY = "/absolute/path/to/the/video_directory"

            # The output is written into VIDEO_DIRECTORY. Existing outputs are never overwritten;
            # the notebook chooses stitched.mp4, stitched_001.mp4, stitched_002.mp4, and so on.
            OUTPUT_STEM = "stitched"
            SEARCH_RECURSIVELY = False
            KEEP_AUDIO = True
            VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}

            # Leave as None to find ffmpeg on PATH, then try the imageio-ffmpeg package.
            FFMPEG_EXECUTABLE = None
            ''',
        ),
        markdown(
            "stitch-03-discovery-heading",
            "## 2. Discover clips and verify their order",
        ),
        code(
            "stitch-04-discovery",
            r'''
            import re
            import shutil
            import subprocess
            import tempfile
            from pathlib import Path
            from IPython.display import Markdown, Video, display


            def natural_sort_key(path, root):
                relative = path.relative_to(root).as_posix().casefold()
                return tuple(
                    int(part) if part.isdigit() else part
                    for part in re.split(r"(\d+)", relative)
                )


            def is_generated_output(path):
                stem = path.stem
                return stem == OUTPUT_STEM or re.fullmatch(
                    rf"{re.escape(OUTPUT_STEM)}_\d{{3}}", stem
                ) is not None


            def find_ffmpeg():
                if FFMPEG_EXECUTABLE:
                    candidate = Path(FFMPEG_EXECUTABLE).expanduser()
                    if not candidate.is_file():
                        raise FileNotFoundError(f"FFMPEG_EXECUTABLE does not exist: {candidate}")
                    return str(candidate)
                system_ffmpeg = shutil.which("ffmpeg")
                if system_ffmpeg:
                    return system_ffmpeg
                try:
                    import imageio_ffmpeg
                except ImportError as error:
                    raise RuntimeError(
                        "FFmpeg was not found. Install it with `brew install ffmpeg`, or run "
                        "`%pip install imageio-ffmpeg`, then rerun this cell."
                    ) from error
                return imageio_ffmpeg.get_ffmpeg_exe()


            CLIP_DIRECTORY = Path(VIDEO_DIRECTORY).expanduser().resolve()
            if not CLIP_DIRECTORY.is_dir():
                raise FileNotFoundError(
                    f"VIDEO_DIRECTORY is not a directory: {CLIP_DIRECTORY}\n"
                    "Edit the settings cell above, then rerun this cell."
                )

            iterator = CLIP_DIRECTORY.rglob("*") if SEARCH_RECURSIVELY else CLIP_DIRECTORY.iterdir()
            CLIP_PATHS = sorted(
                (
                    path.resolve()
                    for path in iterator
                    if path.is_file()
                    and path.suffix.casefold() in VIDEO_SUFFIXES
                    and not path.name.startswith(".")
                    and not is_generated_output(path)
                ),
                key=lambda path: natural_sort_key(path, CLIP_DIRECTORY),
            )
            if len(CLIP_PATHS) < 2:
                raise RuntimeError(
                    f"Found {len(CLIP_PATHS)} input clip(s) in {CLIP_DIRECTORY}; at least 2 are required."
                )

            FFMPEG = find_ffmpeg()
            print(f"FFmpeg: {FFMPEG}")
            print(f"Input directory: {CLIP_DIRECTORY}")
            print("\nExact stitch order:")
            for index, path in enumerate(CLIP_PATHS, start=1):
                relative = path.relative_to(CLIP_DIRECTORY)
                size_mib = path.stat().st_size / 1024**2
                print(f"  {index:03d}. {relative}  ({size_mib:.1f} MiB)")
            ''',
        ),
        markdown(
            "stitch-05-run-heading",
            r"""
            ## 3. Stitch in the printed order

            This uses FFmpeg's concat demuxer with `-c copy`. It does not alter any frames. If
            FFmpeg reports incompatible streams, make sure the folder contains clips from the same
            render settings; this notebook intentionally does not normalize or re-encode them.
            """,
        ),
        code(
            "stitch-06-run",
            r'''
            def next_output_path(directory, stem):
                first = directory / f"{stem}.mp4"
                if not first.exists():
                    return first
                index = 1
                while True:
                    candidate = directory / f"{stem}_{index:03d}.mp4"
                    if not candidate.exists():
                        return candidate
                    index += 1


            def ffconcat_file_line(path):
                # FFmpeg concat-file quoting for absolute paths, including spaces/apostrophes.
                escaped = str(path).replace("'", "'\\''")
                return f"file '{escaped}'"


            def run_ffmpeg(command):
                print("Running FFmpeg stream-copy concat...")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                log_lines = []
                assert process.stdout is not None
                for line in process.stdout:
                    log_lines.append(line)
                    print(line, end="")
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(
                        "FFmpeg concat failed. The clips probably have incompatible codecs, "
                        "dimensions, frame rates, time bases, or audio layouts. Because this "
                        "notebook performs no re-encoding, make the input clips consistent first.\n\n"
                        + "".join(log_lines[-80:])
                    )


            OUTPUT_VIDEO_PATH = next_output_path(CLIP_DIRECTORY, OUTPUT_STEM)
            partial_output = OUTPUT_VIDEO_PATH.with_name(
                f".{OUTPUT_VIDEO_PATH.stem}.partial{OUTPUT_VIDEO_PATH.suffix}"
            )
            if partial_output.exists():
                partial_output.unlink()

            with tempfile.TemporaryDirectory(prefix="local_video_stitch_") as temporary_directory:
                concat_path = Path(temporary_directory) / "clips.ffconcat"
                concat_path.write_text(
                    "ffconcat version 1.0\n"
                    + "\n".join(ffconcat_file_line(path) for path in CLIP_PATHS)
                    + "\n",
                    encoding="utf-8",
                )
                command = [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-fflags", "+genpts",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_path),
                    "-map", "0:v:0",
                ]
                if KEEP_AUDIO:
                    command.extend(["-map", "0:a?"])
                else:
                    command.append("-an")
                command.extend([
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    "-movflags", "+faststart",
                    str(partial_output),
                ])
                try:
                    run_ffmpeg(command)
                    if not partial_output.is_file() or partial_output.stat().st_size == 0:
                        raise RuntimeError(f"FFmpeg produced no usable output: {partial_output}")
                    partial_output.replace(OUTPUT_VIDEO_PATH)
                except Exception:
                    partial_output.unlink(missing_ok=True)
                    raise

            print(f"\nStitched {len(CLIP_PATHS)} clips without re-encoding:")
            print(OUTPUT_VIDEO_PATH)
            ''',
        ),
        markdown(
            "stitch-07-preview-heading",
            "## 4. Preview the result",
        ),
        code(
            "stitch-08-preview",
            r'''
            if not OUTPUT_VIDEO_PATH.is_file():
                raise FileNotFoundError(OUTPUT_VIDEO_PATH)
            display(Markdown(f"### {OUTPUT_VIDEO_PATH.name}"))
            display(Video(str(OUTPUT_VIDEO_PATH), embed=False, width=768))
            ''',
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite tracked/user notebook: {OUTPUT}. "
            "Edit it with a narrow migration instead."
        )
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
