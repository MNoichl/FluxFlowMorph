import pytest
import torch

from flowmorph_klein.flux2_latents import patchify_latent, unpatchify_latent


@pytest.mark.parametrize(
    ("shape", "patch_size"),
    (
        ((2, 3, 8, 10), 2),
        ((1, 7, 12, 8), (3, 2)),
        ((3, 11, 5, 9), 1),
    ),
)
def test_patchify_unpatchify_roundtrip(shape: tuple[int, ...], patch_size) -> None:
    latent = torch.randn(shape)
    patchified = patchify_latent(latent, patch_size)
    restored = unpatchify_latent(patchified, patch_size)
    assert restored.shape == latent.shape
    torch.testing.assert_close(restored, latent, atol=0, rtol=0)


def test_patchify_rejects_non_divisible_spatial_shape() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        patchify_latent(torch.randn(1, 4, 7, 8), 2)


def test_unpatchify_derives_output_channels() -> None:
    patchified = torch.randn(1, 52, 4, 5)
    restored = unpatchify_latent(patchified, 2)
    assert restored.shape == (1, 13, 8, 10)

