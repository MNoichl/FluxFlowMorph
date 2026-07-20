"""Differentiable low-level FLUX.2 Klein velocity prediction.

The high-level Diffusers pipeline is decorated with ``torch.no_grad`` and is
therefore unsuitable for FlowMorph fitting.  These helpers reproduce only its
transformer invocation while preserving PEFT/LoRA hooks and gradients with
respect to the packed latent state.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch

from .conditioning import ConditioningPackage


VELOCITY_PARITY_ATOL: dict[torch.dtype, float] = {
    torch.float32: 1e-5,
    torch.bfloat16: 2e-2,
    torch.float16: 5e-3,
}
VELOCITY_PARITY_RTOL: dict[torch.dtype, float] = {
    torch.float32: 1e-5,
    torch.bfloat16: 2e-2,
    torch.float16: 5e-3,
}


def _unwrap_transformer(value: Any) -> Any:
    transformer = getattr(value, "transformer", value)
    if not callable(transformer):
        raise TypeError("expected a transformer module or pipeline with .transformer")
    return transformer


def _module_dtype(module: Any, fallback: torch.dtype) -> torch.dtype:
    flowmorph_compute_dtype = getattr(module, "_flowmorph_compute_dtype", None)
    if isinstance(flowmorph_compute_dtype, torch.dtype):
        return flowmorph_compute_dtype
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    return fallback


def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise ValueError(
        f"{name} batch size must be 1 or {batch_size}, got {tensor.shape[0]}"
    )


def _raw_timestep_batch(
    timestep: torch.Tensor | float | int,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(timestep, device=device, dtype=dtype)
    if value.ndim == 0:
        return value.expand(batch_size)
    if value.ndim == 1 and value.shape[0] in {1, batch_size}:
        return value.expand(batch_size)
    raise ValueError(
        f"timestep must be scalar or have shape (1,) / ({batch_size},), got {tuple(value.shape)}"
    )


def _cache_context(transformer: Any, name: str):
    factory = getattr(transformer, "cache_context", None)
    return factory(name) if callable(factory) else nullcontext()


def _extract_velocity(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    elif hasattr(output, "sample"):
        output = output.sample
    if not isinstance(output, torch.Tensor):
        raise TypeError("transformer output must contain a tensor velocity")
    if output.ndim != 3:
        raise ValueError(f"transformer velocity must have shape (B, N, C), got {tuple(output.shape)}")
    return output


def _predict_velocity(
    transformer_or_pipeline: Any,
    state: torch.Tensor,
    timestep: torch.Tensor | float | int,
    conditioning: ConditioningPackage,
    image_ids: torch.Tensor,
    *,
    joint_attention_kwargs: dict[str, Any] | None,
    cache_name: str,
) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError(f"state must have shape (B, N, C), got {tuple(state.shape)}")
    if not state.is_floating_point():
        raise TypeError("state must be a floating-point tensor")
    if image_ids.ndim != 3 or image_ids.shape[-1] != 4:
        raise ValueError("image_ids must have shape (B, N, 4)")
    if image_ids.shape[1] != state.shape[1]:
        raise ValueError("image_ids token count must equal state token count")

    transformer = _unwrap_transformer(transformer_or_pipeline)
    batch_size = state.shape[0]
    prompt_embeds = _expand_batch(conditioning.prompt_embeds, batch_size, "prompt embeddings")
    text_ids = _expand_batch(conditioning.text_ids, batch_size, "text IDs")
    latent_ids = _expand_batch(image_ids, batch_size, "image IDs")

    prompt_embeds = prompt_embeds.to(device=state.device)
    text_ids = text_ids.to(device=state.device)
    latent_ids = latent_ids.to(device=state.device)
    transformer_dtype = _module_dtype(transformer, state.dtype)
    raw_timestep = _raw_timestep_batch(
        timestep,
        batch_size=batch_size,
        device=state.device,
        dtype=transformer_dtype,
    )
    latent_model_input = state.to(dtype=transformer_dtype)
    autocast_context = (
        torch.autocast(
            device_type="cuda",
            dtype=transformer_dtype,
            enabled=True,
        )
        if state.device.type == "cuda"
        and transformer_dtype in {torch.bfloat16, torch.float16}
        else nullcontext()
    )

    # Flux2KleinPipeline supplies scheduler timesteps in [0, 1000] and divides
    # by 1000. Flux2Transformer2DModel multiplies them internally. Do not
    # normalize at any other layer.
    with _cache_context(transformer, cache_name), autocast_context:
        output = transformer(
            hidden_states=latent_model_input,
            timestep=raw_timestep / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )
    velocity = _extract_velocity(output)
    if velocity.shape[0] != batch_size or velocity.shape[2] != state.shape[2]:
        raise ValueError(
            "transformer velocity batch/channel dimensions do not match state: "
            f"{tuple(velocity.shape)} versus {tuple(state.shape)}"
        )
    if velocity.shape[1] < state.shape[1]:
        raise ValueError("transformer returned fewer image tokens than the FlowMorph state")
    return velocity[:, : state.shape[1] :]


def predict_conditional_velocity(
    transformer: Any,
    state: torch.Tensor,
    timestep: torch.Tensor | float | int,
    conditioning: ConditioningPackage,
    image_ids: torch.Tensor,
    *,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Predict the conditional velocity with the pinned transformer call."""

    return _predict_velocity(
        transformer,
        state,
        timestep,
        conditioning,
        image_ids,
        joint_attention_kwargs=joint_attention_kwargs,
        cache_name="cond",
    )


def predict_unconditional_velocity(
    transformer: Any,
    state: torch.Tensor,
    timestep: torch.Tensor | float | int,
    conditioning: ConditioningPackage,
    image_ids: torch.Tensor,
    *,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Predict the negative/unconditional branch without detaching it."""

    return _predict_velocity(
        transformer,
        state,
        timestep,
        conditioning,
        image_ids,
        joint_attention_kwargs=joint_attention_kwargs,
        cache_name="uncond",
    )


def _predict_batched_cfg_branches(
    transformer: Any,
    state: torch.Tensor,
    timestep: torch.Tensor | float | int,
    conditional: ConditioningPackage,
    unconditional: ConditioningPackage,
    image_ids: torch.Tensor,
    *,
    joint_attention_kwargs: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = state.shape[0]
    cond_embeds = _expand_batch(conditional.prompt_embeds, batch_size, "conditional embeddings")
    uncond_embeds = _expand_batch(unconditional.prompt_embeds, batch_size, "unconditional embeddings")
    cond_ids = _expand_batch(conditional.text_ids, batch_size, "conditional text IDs")
    uncond_ids = _expand_batch(unconditional.text_ids, batch_size, "unconditional text IDs")
    if cond_embeds.shape[1:] != uncond_embeds.shape[1:]:
        raise ValueError("batched CFG requires equal conditional/unconditional embedding shapes")
    if cond_ids.shape[1:] != uncond_ids.shape[1:]:
        raise ValueError("batched CFG requires equal conditional/unconditional text-ID shapes")

    raw_timestep = _raw_timestep_batch(
        timestep,
        batch_size=batch_size,
        device=state.device,
        dtype=state.dtype,
    )
    ids = _expand_batch(image_ids, batch_size, "image IDs")
    batched_conditioning = ConditioningPackage(
        prompt=("conditional", "unconditional"),
        prompt_embeds=torch.cat((cond_embeds, uncond_embeds), dim=0).detach(),
        text_ids=torch.cat((cond_ids, uncond_ids), dim=0).detach(),
    )
    batched_velocity = _predict_velocity(
        transformer,
        torch.cat((state, state), dim=0),
        torch.cat((raw_timestep, raw_timestep), dim=0),
        batched_conditioning,
        torch.cat((ids, ids), dim=0),
        joint_attention_kwargs=joint_attention_kwargs,
        cache_name="cfg",
    )
    conditional_velocity, unconditional_velocity = batched_velocity.split(batch_size, dim=0)
    return conditional_velocity, unconditional_velocity


def predict_cfg_velocity(
    transformer: Any,
    state: torch.Tensor,
    timestep: torch.Tensor | float | int,
    conditional: ConditioningPackage,
    unconditional: ConditioningPackage,
    image_ids: torch.Tensor,
    *,
    guidance_scale: float = 4.0,
    cfg_enabled: bool = True,
    cfg_execution: str = "sequential",
    execution: str | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply exact external CFG using sequential or batched evaluation."""

    if not isinstance(guidance_scale, (int, float)) or not math.isfinite(float(guidance_scale)):
        raise ValueError("guidance_scale must be finite")
    mode = execution if execution is not None else cfg_execution
    mode = str(mode).lower()
    if mode not in {"sequential", "batched"}:
        raise ValueError("CFG execution must be 'sequential' or 'batched'")

    # This mirrors the Base pipeline's `guidance_scale > 1` gate. Scale 1 is
    # the explicit diagnostic no-CFG mode and still returns the conditional
    # branch, not the unconditional branch.
    if not cfg_enabled or float(guidance_scale) <= 1.0:
        return predict_conditional_velocity(
            transformer,
            state,
            timestep,
            conditional,
            image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    if mode == "sequential":
        conditional_velocity = predict_conditional_velocity(
            transformer,
            state,
            timestep,
            conditional,
            image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        unconditional_velocity = predict_unconditional_velocity(
            transformer,
            state,
            timestep,
            unconditional,
            image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
        )
    else:
        conditional_velocity, unconditional_velocity = _predict_batched_cfg_branches(
            transformer,
            state,
            timestep,
            conditional,
            unconditional,
            image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
        )
    return unconditional_velocity + float(guidance_scale) * (
        conditional_velocity - unconditional_velocity
    )


class FlowMorphFlux2Model:
    """Frozen transformer facade used by fitting and rendering code."""

    def __init__(self, pipeline_or_transformer: Any, *, freeze: bool = True) -> None:
        self.pipeline = pipeline_or_transformer if hasattr(pipeline_or_transformer, "transformer") else None
        self.transformer = _unwrap_transformer(pipeline_or_transformer)
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        parameters = getattr(self.transformer, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        evaluation = getattr(self.transformer, "eval", None)
        if callable(evaluation):
            evaluation()

    def parameters(self):
        parameters = getattr(self.transformer, "parameters", None)
        return parameters() if callable(parameters) else iter(())

    def enable_gradient_checkpointing(self) -> None:
        enable = getattr(self.transformer, "enable_gradient_checkpointing", None)
        if not callable(enable):
            raise RuntimeError("loaded transformer does not support gradient checkpointing")
        enable()

    def disable_gradient_checkpointing(self) -> None:
        disable = getattr(self.transformer, "disable_gradient_checkpointing", None)
        if not callable(disable):
            raise RuntimeError("loaded transformer cannot disable gradient checkpointing")
        disable()

    def set_attention_backend(self, backend: str) -> None:
        """Map the project's ``sdpa`` label to Diffusers' ``native`` name."""

        normalized = str(backend).lower()
        if normalized == "sdpa":
            normalized = "native"
        setter = getattr(self.transformer, "set_attention_backend", None)
        if not callable(setter):
            raise RuntimeError("loaded transformer does not expose set_attention_backend")
        setter(normalized)

    def predict_conditional_velocity(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return predict_conditional_velocity(self.transformer, *args, **kwargs)

    def predict_unconditional_velocity(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return predict_unconditional_velocity(self.transformer, *args, **kwargs)

    def predict_cfg_velocity(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return predict_cfg_velocity(self.transformer, *args, **kwargs)
