from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

from flowmorph_klein.sage_control import (
    make_canny_reference,
    make_structure_lock_mask,
    smoothstep,
    warp_sage_endpoints,
)


def test_smoothstep_preserves_endpoints() -> None:
    assert smoothstep(0.0) == 0.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(1.0) == 1.0


def test_dense_warp_retains_endpoint_detail() -> None:
    pytest.importorskip("skimage")
    left_array = np.zeros((32, 32, 3), dtype=np.uint8)
    right_array = np.zeros((32, 32, 3), dtype=np.uint8)
    left_array[8:16, 4:12] = (255, 20, 20)
    right_array[16:24, 20:28] = (20, 20, 255)
    source = np.array([[[4, 8], [12, 16]], [[4, 16], [12, 8]]], dtype=float)
    target = np.array([[[20, 16], [28, 24]], [[20, 24], [28, 16]]], dtype=float)
    middle = (source + target) / 2.0
    result = warp_sage_endpoints(
        Image.fromarray(left_array),
        Image.fromarray(right_array),
        source,
        target,
        middle,
        0.5,
        maximum_control_lines=None,
        border_samples_per_edge=3,
    )
    output = np.asarray(result.image)
    assert output.shape == (32, 32, 3)
    assert output.std() > 5.0
    assert result.source_control_points >= 8
    result.image.close()
    result.warped_left.close()
    result.warped_right.close()


def test_dense_warp_supports_legacy_scikit_image_estimator() -> None:
    skimage = pytest.importorskip("skimage.transform")
    real_transform = skimage.PiecewiseAffineTransform

    class LegacyTransform:
        def __init__(self):
            self._delegate = None

        def estimate(self, source, destination):
            self._delegate = real_transform.from_estimate(source, destination)
            return True

        @property
        def inverse(self):
            return self._delegate.inverse

    image = Image.new("RGB", (32, 32), (80, 40, 20))
    source = np.array([[[4, 4], [12, 12]], [[4, 12], [12, 4]]], dtype=float)
    target = source + 8.0
    with patch("skimage.transform.PiecewiseAffineTransform", LegacyTransform):
        result = warp_sage_endpoints(
            image,
            image,
            source,
            target,
            (source + target) / 2.0,
            0.5,
            maximum_control_lines=None,
            border_samples_per_edge=3,
        )
    assert result.image.size == (32, 32)
    result.image.close()
    result.warped_left.close()
    result.warped_right.close()
    image.close()


def test_control_images_are_not_colored_overlays() -> None:
    condition = Image.new("RGB", (32, 32), (0, 0, 0))
    array = np.asarray(condition).copy()
    array[15:17, 4:28] = (255, 30, 180)
    condition = Image.fromarray(array)
    canny = make_canny_reference(condition, dilation_pixels=1)
    lock = make_structure_lock_mask(
        condition, dilation_pixels=2, feather_radius=1.0, strength=0.5
    )
    canny_array = np.asarray(canny)
    lock_array = np.asarray(lock)
    assert set(np.unique(canny_array)).issubset({0, 255})
    assert lock_array.max() <= 128
    assert lock_array.sum() > 0
    condition.close()
    canny.close()
    lock.close()
