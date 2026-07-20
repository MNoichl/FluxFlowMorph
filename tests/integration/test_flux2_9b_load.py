from __future__ import annotations

import os

import pytest
import torch

from flowmorph_klein import MODEL_ID, MODEL_REVISION
from flowmorph_klein.conditioning import ConditioningPackage
from flowmorph_klein.config import ProjectTemplateConfig
from flowmorph_klein.flow_schedule import euler_flow_update
from flowmorph_klein.flux2_latents import (
    decode_packed_latent,
    encode_image_to_packed_latent,
    preprocess_endpoint_image,
)
from flowmorph_klein.flux2_model import predict_cfg_velocity


def test_offline_load_contract_is_exact_base_9b_and_never_a_fallback() -> None:
    config = ProjectTemplateConfig()
    assert config.model.id == MODEL_ID
    assert MODEL_ID == "black-forest-labs/FLUX.2-klein-base-9B"
    assert MODEL_REVISION == "32773329fbe7e81a90ef971740e8ba4b0364ecf3"
    assert "4B" not in config.model.id
    assert "distill" not in config.model.id.lower()
    assert config.input.width == config.input.height == 512
    assert config.memory.allow_degraded_run is False


@pytest.mark.integration
@pytest.mark.flux2_9b
@pytest.mark.requires_gated_access
def test_real_flux2_klein_base_9b_load_and_stock_512_generation(integration_harness) -> None:
    """Load and generate only when gated access is explicitly opted into."""

    integration_harness.require()
    diffusers = pytest.importorskip("diffusers")

    pipeline = None
    try:
        pipeline = diffusers.Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=integration_harness.hf_cache,
            torch_dtype=torch.bfloat16,
            local_files_only=os.environ.get("FLOWMORPH_TEST_LOCAL_FILES_ONLY", "").lower()
            in {"1", "true", "yes", "on"},
        ).to("cuda:0")

        assert type(pipeline).__name__ == "Flux2KleinPipeline"
        assert type(pipeline.transformer).__name__ == "Flux2Transformer2DModel"
        assert type(pipeline.vae).__name__ == "AutoencoderKLFlux2"
        assert type(pipeline.scheduler).__name__ == "FlowMatchEulerDiscreteScheduler"
        assert getattr(pipeline.config, "is_distilled", False) is False
        transformer_config = pipeline.transformer.config
        assert transformer_config.in_channels == 128
        assert transformer_config.num_layers == 8
        assert transformer_config.num_single_layers == 24
        assert transformer_config.guidance_embeds is False

        device = torch.device("cuda:0")
        prompt = "a small cobalt sphere on a neutral background"
        generator = torch.Generator(device=device).manual_seed(42)
        output = pipeline(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=4.0,
            generator=generator,
            output_type="pil",
        )
        assert len(output.images) == 1
        assert output.images[0].size == (512, 512)

        # Real custom-vs-pinned preprocessing, deterministic VAE encoding,
        # packing/IDs, inverse transform, VAE decode, and postprocess parity.
        custom_image = preprocess_endpoint_image(
            output.images[0],
            pipeline.image_processor,
            height=512,
            width=512,
            resize_mode="default",
        ).to(device=device, dtype=pipeline.vae.dtype)
        pinned_image = pipeline.image_processor.preprocess(
            output.images[0],
            height=512,
            width=512,
            resize_mode="default",
        ).to(device=device, dtype=pipeline.vae.dtype)
        torch.testing.assert_close(custom_image, pinned_image, atol=0.0, rtol=0.0)

        with torch.inference_mode():
            custom_tokens, custom_ids = encode_image_to_packed_latent(
                custom_image,
                pipeline.vae,
                preprocessed=True,
            )
            pinned_spatial = pipeline._encode_vae_image(
                pinned_image,
                generator=torch.Generator(device=device).manual_seed(17),
            )
            pinned_tokens = pipeline._pack_latents(pinned_spatial)
            pinned_ids = pipeline._prepare_latent_ids(pinned_spatial).to(device)
        vae_tolerance = 2e-2 if custom_tokens.dtype is torch.bfloat16 else 5e-3
        torch.testing.assert_close(
            custom_tokens,
            pinned_tokens,
            atol=vae_tolerance,
            rtol=vae_tolerance,
        )
        assert torch.equal(custom_ids, pinned_ids)

        with torch.inference_mode():
            custom_decoded = decode_packed_latent(
                custom_tokens,
                custom_ids,
                pipeline.vae,
                postprocess=False,
            )
            pinned_unpacked = pipeline._unpack_latents_with_ids(
                pinned_tokens,
                pinned_ids,
            )
            pinned_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(
                pinned_unpacked.device,
                pinned_unpacked.dtype,
            )
            pinned_std = torch.sqrt(
                pipeline.vae.bn.running_var.view(1, -1, 1, 1)
                + pipeline.vae.config.batch_norm_eps
            ).to(pinned_unpacked.device, pinned_unpacked.dtype)
            pinned_vae_latent = pipeline._unpatchify_latents(
                pinned_unpacked * pinned_std + pinned_mean
            )
            pinned_decoded = pipeline.vae.decode(
                pinned_vae_latent,
                return_dict=False,
            )[0]
        torch.testing.assert_close(
            custom_decoded,
            pinned_decoded,
            atol=2e-2,
            rtol=2e-2,
        )
        with torch.inference_mode():
            custom_pixels = decode_packed_latent(
                custom_tokens,
                custom_ids,
                pipeline.vae,
                image_processor=pipeline.image_processor,
                output_type="pt",
                postprocess=True,
            )
            pinned_pixels = pipeline.image_processor.postprocess(
                pinned_decoded,
                output_type="pt",
            )
        torch.testing.assert_close(custom_pixels, pinned_pixels, atol=2e-2, rtol=2e-2)

        # Recover the stock pipeline's post-step packed latents through its
        # callback and compare them with our differentiable conditional/CFG
        # velocity plus the independently tested Euler arithmetic.
        with torch.inference_mode():
            prompt_embeds, text_ids = pipeline.encode_prompt(prompt, device=device)
            negative_embeds, negative_ids = pipeline.encode_prompt("", device=device)
            initial_state, image_ids = pipeline.prepare_latents(
                batch_size=1,
                num_latents_channels=pipeline.transformer.config.in_channels // 4,
                height=512,
                width=512,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=torch.Generator(device=device).manual_seed(1234),
            )
        conditional = ConditioningPackage(prompt, prompt_embeds.detach(), text_ids.detach())
        unconditional = ConditioningPackage(
            "",
            negative_embeds.detach(),
            negative_ids.detach(),
        )

        for guidance_scale in (1.0, 4.0):
            captured: dict[str, torch.Tensor] = {}

            def capture_step(_pipe, _step, _timestep, callback_kwargs):
                captured["latents"] = callback_kwargs["latents"].detach().clone()
                return callback_kwargs

            pipeline(
                prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                height=512,
                width=512,
                num_inference_steps=1,
                guidance_scale=guidance_scale,
                latents=initial_state.detach().clone(),
                output_type="latent",
                callback_on_step_end=capture_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            assert "latents" in captured
            timestep = pipeline.scheduler.timesteps[0].to(device)
            with torch.inference_mode():
                custom_velocity = predict_cfg_velocity(
                    pipeline.transformer,
                    initial_state,
                    timestep,
                    conditional,
                    unconditional,
                    image_ids,
                    guidance_scale=guidance_scale,
                    cfg_enabled=True,
                    cfg_execution="sequential",
                )
                custom_next = euler_flow_update(
                    initial_state,
                    custom_velocity,
                    pipeline.scheduler.sigmas[0],
                    pipeline.scheduler.sigmas[1],
                )
            torch.testing.assert_close(
                custom_next,
                captured["latents"],
                atol=2e-2,
                rtol=2e-2,
            )
    finally:
        del pipeline
        torch.cuda.empty_cache()
