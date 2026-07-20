from types import SimpleNamespace

import pytest
import torch

from flowmorph_klein.flux2_latents import (
    decode_vae_latent,
    denormalize_vae_latent,
    encode_image_to_vae_latent,
    normalize_vae_latent,
)


class FakeVAE:
    def __init__(self, channels: int) -> None:
        self.bn = SimpleNamespace(
            running_mean=torch.linspace(-0.5, 0.5, channels),
            running_var=torch.linspace(0.25, 1.25, channels),
            eps=1e-4,
        )
        self.config = SimpleNamespace(batch_norm_eps=1e-4)


@pytest.mark.parametrize("shape", ((2, 7, 5, 3), (1, 12, 2, 9)))
def test_normalize_denormalize_roundtrip(shape: tuple[int, ...]) -> None:
    torch.manual_seed(4)
    latent = torch.randn(shape)
    vae = FakeVAE(shape[1])
    restored = denormalize_vae_latent(normalize_vae_latent(latent, vae), vae)
    torch.testing.assert_close(restored, latent, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(restored).all()


def test_statistics_must_match_dynamic_channel_count() -> None:
    with pytest.raises(ValueError, match="statistics contain"):
        normalize_vae_latent(torch.randn(1, 8, 2, 2), FakeVAE(7))


def test_explicit_statistics_are_supported() -> None:
    latent = torch.randn(1, 3, 2, 2)
    mean = torch.tensor([1.0, 2.0, 3.0])
    variance = torch.tensor([0.5, 1.0, 1.5])
    normalized = normalize_vae_latent(
        latent,
        running_mean=mean,
        running_var=variance,
        eps=1e-4,
    )
    restored = denormalize_vae_latent(
        normalized,
        running_mean=mean,
        running_var=variance,
        eps=1e-4,
    )
    torch.testing.assert_close(restored, latent)


def test_vae_encoding_uses_posterior_mode_not_sampling() -> None:
    expected = torch.randn(1, 5, 3, 4)

    class Distribution:
        def mode(self):
            return expected

        def sample(self, *args, **kwargs):
            raise AssertionError("the deterministic path must not sample")

    class Encoder:
        def encode(self, image):
            return SimpleNamespace(latent_dist=Distribution())

    actual = encode_image_to_vae_latent(torch.zeros(1, 3, 24, 32), Encoder())
    assert actual is expected


def test_vae_decode_rejects_non_finite_pixels_before_postprocessing() -> None:
    class Decoder:
        def decode(self, latent, return_dict=False):
            del latent, return_dict
            return (torch.full((1, 3, 4, 4), float("nan")),)

    class Processor:
        def postprocess(self, *args, **kwargs):
            raise AssertionError("non-finite decoded pixels must never reach postprocess")

    with pytest.raises(FloatingPointError, match="before image postprocessing"):
        decode_vae_latent(
            torch.zeros(1, 4, 2, 2),
            Decoder(),
            image_processor=Processor(),
            output_type="pil",
        )
