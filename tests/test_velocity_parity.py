from contextlib import contextmanager

import pytest
import torch
from torch import nn

from flowmorph_klein.conditioning import ConditioningPackage
from flowmorph_klein.flux2_latents import make_image_ids
from flowmorph_klein.flux2_model import (
    FlowMorphFlux2Model,
    predict_cfg_velocity,
    predict_conditional_velocity,
)


class TinyFlux2Transformer(nn.Module):
    def __init__(self, channels: int, feature_width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(channels, channels, bias=False)
        self.condition = nn.Linear(feature_width, channels, bias=False)
        self.last_timestep = None
        self.cache_names: list[str] = []

    @contextmanager
    def cache_context(self, name: str):
        self.cache_names.append(name)
        yield

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        guidance,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        joint_attention_kwargs,
        return_dict,
    ):
        assert guidance is None
        assert return_dict is False
        assert txt_ids.shape[-1] == img_ids.shape[-1] == 4
        self.last_timestep = timestep.detach().clone()
        pooled = encoder_hidden_states.mean(dim=1)
        output = self.projection(hidden_states) + self.condition(pooled).unsqueeze(1)
        output = output + timestep[:, None, None].to(output.dtype)
        return (output,)


def package(value: float, *, batch: int = 1, sequence: int = 4, width: int = 5):
    embeds = torch.full((batch, sequence, width), value)
    ids = torch.zeros(batch, sequence, 4, dtype=torch.int64)
    ids[..., 3] = torch.arange(sequence)
    return ConditioningPackage(str(value), embeds, ids)


def test_conditional_path_matches_explicit_transformer_invocation() -> None:
    torch.manual_seed(8)
    transformer = TinyFlux2Transformer(channels=3, feature_width=5)
    state = torch.randn(2, 6, 3)
    conditioning = package(0.25)
    image_ids = make_image_ids(batch_size=2, height=2, width=3)
    actual = predict_conditional_velocity(
        transformer,
        state,
        torch.tensor(500.0),
        conditioning,
        image_ids,
    )
    expected = transformer(
        hidden_states=state,
        timestep=torch.full((2,), 0.5),
        guidance=None,
        encoder_hidden_states=conditioning.prompt_embeds.expand(2, -1, -1),
        txt_ids=conditioning.text_ids.expand(2, -1, -1),
        img_ids=image_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(transformer.last_timestep, torch.full((2,), 0.5))


def test_timestep_is_cast_to_model_dtype_before_normalization() -> None:
    transformer = TinyFlux2Transformer(channels=3, feature_width=5).to(torch.bfloat16)
    state = torch.randn(1, 6, 3, dtype=torch.float32)
    conditioning = package(0.25).to(dtype=torch.bfloat16)
    image_ids = make_image_ids(batch_size=1, height=2, width=3)
    raw = torch.tensor(758.4072, dtype=torch.float32)

    predict_conditional_velocity(transformer, state, raw, conditioning, image_ids)

    expected = raw.to(torch.bfloat16).expand(1) / 1000
    assert transformer.last_timestep.dtype is torch.bfloat16
    torch.testing.assert_close(transformer.last_timestep, expected)


@pytest.mark.parametrize("execution", ("sequential", "batched"))
def test_cfg_matches_reference_and_preserves_input_gradient(execution: str) -> None:
    torch.manual_seed(11)
    transformer = TinyFlux2Transformer(channels=4, feature_width=5)
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    state = torch.randn(2, 6, 4, requires_grad=True)
    conditional = package(0.75)
    unconditional = package(-0.25)
    image_ids = make_image_ids(batch_size=2, height=2, width=3)

    cfg = predict_cfg_velocity(
        transformer,
        state,
        800.0,
        conditional,
        unconditional,
        image_ids,
        guidance_scale=4.0,
        execution=execution,
    )
    cond = predict_conditional_velocity(transformer, state, 800.0, conditional, image_ids)
    uncond = predict_conditional_velocity(transformer, state, 800.0, unconditional, image_ids)
    expected = uncond + 4.0 * (cond - uncond)
    torch.testing.assert_close(cfg, expected, atol=1e-6, rtol=1e-6)

    cfg.square().mean().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in transformer.parameters())


def test_guidance_scale_one_uses_effective_no_cfg_path() -> None:
    transformer = TinyFlux2Transformer(channels=2, feature_width=5)
    state = torch.randn(1, 4, 2)
    conditional = package(0.5)
    unconditional = package(-1.0)
    ids = make_image_ids(batch_size=1, height=2, width=2)
    result = predict_cfg_velocity(
        transformer,
        state,
        250,
        conditional,
        unconditional,
        ids,
        guidance_scale=1.0,
    )
    expected = predict_conditional_velocity(transformer, state, 250, conditional, ids)
    torch.testing.assert_close(result, expected)


def test_cfg_residual_callback_observes_unscaled_conditional_difference() -> None:
    transformer = TinyFlux2Transformer(channels=2, feature_width=5)
    state = torch.randn(2, 4, 2)
    conditional = package(0.5)
    unconditional = package(-1.0)
    ids = make_image_ids(batch_size=2, height=2, width=2)
    observed: list[torch.Tensor] = []

    predict_cfg_velocity(
        transformer,
        state,
        250,
        conditional,
        unconditional,
        ids,
        guidance_scale=7.0,
        execution="batched",
        cfg_residual_callback=observed.append,
    )
    cond = predict_conditional_velocity(transformer, state, 250, conditional, ids)
    uncond = predict_conditional_velocity(transformer, state, 250, unconditional, ids)

    assert len(observed) == 1
    torch.testing.assert_close(observed[0], cond - uncond)


def test_conditioning_prompt_hash_is_stable_and_text_sensitive() -> None:
    first = package(0.5)
    repeated = package(0.5)
    changed = ConditioningPackage("different", first.prompt_embeds, first.text_ids)
    assert first.prompt_sha256 == repeated.prompt_sha256
    assert first.prompt_hash == first.prompt_sha256
    assert changed.prompt_sha256 != first.prompt_sha256


def test_wrapper_freezes_model_parameters() -> None:
    transformer = TinyFlux2Transformer(channels=2, feature_width=5)
    wrapper = FlowMorphFlux2Model(transformer)
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())
