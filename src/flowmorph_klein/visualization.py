"""Static visual diagnostics for a FlowMorph run."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


def _as_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def difference_image(reference: Image.Image, generated: Image.Image, amplification: float = 4.0) -> Image.Image:
    lhs, rhs = _as_rgb(reference), _as_rgb(generated)
    if lhs.size != rhs.size:
        raise ValueError(f"Image sizes differ: {lhs.size} versus {rhs.size}")
    difference = np.asarray(ImageChops.difference(lhs, rhs), dtype=np.float32) * float(amplification)
    return Image.fromarray(np.clip(difference, 0, 255).astype(np.uint8), mode="RGB")


def make_contact_sheet(
    frames: Sequence[Image.Image],
    output_path: str | Path,
    *,
    columns: int = 5,
    labels: Sequence[str] | None = None,
    gutter: int = 8,
    label_height: int = 24,
    background: tuple[int, int, int] = (24, 24, 24),
) -> Path:
    if not frames:
        raise ValueError("Contact sheet requires at least one frame")
    if columns < 1:
        raise ValueError("columns must be positive")
    images = [_as_rgb(frame) for frame in frames]
    if len({frame.size for frame in images}) != 1:
        raise ValueError("Contact sheet frames must have identical dimensions")
    if labels is not None and len(labels) != len(images):
        raise ValueError("labels length must match frames length")

    width, height = images[0].size
    rows = ceil(len(images) / columns)
    cell_height = height + (label_height if labels is not None else 0)
    sheet_width = columns * width + (columns + 1) * gutter
    sheet_height = rows * cell_height + (rows + 1) * gutter
    sheet = Image.new("RGB", (sheet_width, sheet_height), background)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        x = gutter + column * (width + gutter)
        y = gutter + row * (cell_height + gutter)
        sheet.paste(image, (x, y))
        if labels is not None:
            draw.text((x + 4, y + height + 4), str(labels[index]), fill=(240, 240, 240), font=font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def save_endpoint_comparison(
    source_reference: Image.Image,
    source_generated: Image.Image,
    target_reference: Image.Image,
    target_generated: Image.Image,
    output_path: str | Path,
) -> Path:
    images = [source_reference, source_generated, target_reference, target_generated]
    labels = ["source reference", "source generated", "target reference", "target generated"]
    return make_contact_sheet(images, output_path, columns=2, labels=labels)


def save_loss_plot(
    rows: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    if not rows:
        raise ValueError("Loss plot requires at least one optimization row")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row.get("step", index)) for index, row in enumerate(rows)]
    total = [float(row["total_loss"]) for row in rows]
    reconstruction = [float(row.get("reconstruction_loss", row["total_loss"])) for row in rows]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(steps, total, label="total", linewidth=1.8)
    if reconstruction != total:
        axis.plot(steps, reconstruction, label="reconstruction", linewidth=1.2)
    axis.set(title=title, xlabel="optimization step", ylabel="loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output

