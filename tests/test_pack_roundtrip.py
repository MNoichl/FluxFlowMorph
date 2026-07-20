import torch

from flowmorph_klein.flux2_latents import (
    make_image_ids,
    pack_latent_tokens,
    unpack_latent_tokens,
)


def test_pack_unpack_roundtrip_rectangular_dynamic_shape() -> None:
    latent = torch.randn(2, 13, 5, 7)
    packed = pack_latent_tokens(latent)
    image_ids = make_image_ids(latent)
    assert packed.shape == (2, 35, 13)
    assert image_ids.shape == (2, 35, 4)
    restored = unpack_latent_tokens(packed, image_ids)
    torch.testing.assert_close(restored, latent, atol=0, rtol=0)


def test_unpack_scatter_respects_permuted_ids() -> None:
    latent = torch.arange(1 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 4, 6)
    packed = pack_latent_tokens(latent)
    ids = make_image_ids(latent)
    permutation = torch.randperm(packed.shape[1])
    restored = unpack_latent_tokens(packed[:, permutation], ids[:, permutation])
    torch.testing.assert_close(restored, latent, atol=0, rtol=0)


def test_512_klein_token_and_id_count() -> None:
    # At 512px the VAE downsamples by eight and FLUX.2 patchifies by two:
    # 32*32 positions. Channel count is deliberately arbitrary here.
    latent = torch.zeros(1, 19, 32, 32)
    tokens = pack_latent_tokens(latent)
    ids = make_image_ids(latent)
    assert tokens.shape == (1, 1024, 19)
    assert ids.shape == (1, 1024, 4)
    assert torch.equal(ids[0, 0], torch.tensor([0, 0, 0, 0]))
    assert torch.equal(ids[0, -1], torch.tensor([0, 31, 31, 0]))

