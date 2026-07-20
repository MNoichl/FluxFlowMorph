from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from flowmorph_klein.metrics import endpoint_reconstruction_metrics, psnr, transition_metrics


def _image(value: int) -> Image.Image:
    return Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8), "RGB")


def test_identical_image_has_infinite_psnr():
    assert math.isinf(psnr(_image(20), _image(20)))


def test_endpoint_and_transition_metrics_are_finite_for_distinct_images():
    endpoint = endpoint_reconstruction_metrics(
        _image(0), _image(10), torch.zeros(1, 4, 3), torch.ones(1, 4, 3)
    )
    assert endpoint["psnr"] > 0
    assert endpoint["latent_l1"] == 1

    summary, rows = transition_metrics([_image(0), _image(10), _image(20)])
    assert len(rows) == 2
    assert summary["frame_count"] == 3
    assert summary["adjacent_pixel_change_mean"] > 0

