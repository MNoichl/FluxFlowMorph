"""Prompt resolution and cached FLUX.2 Klein Qwen conditioning.

Production encoding delegates to the pinned pipeline's ``encode_prompt``
method.  Consequently chat templating, hidden-layer selection, sequence
length, and feature width remain properties of the loaded pipeline instead of
being duplicated as fragile constants here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch


NEUTRAL_PROMPT = "an image"


@dataclass(frozen=True, slots=True)
class ConditioningPackage:
    """All fixed text tensors required by a FLUX.2 transformer call."""

    prompt: str | tuple[str, ...]
    prompt_embeds: torch.Tensor
    text_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape (batch, sequence, features)")
        if self.text_ids.ndim != 3 or self.text_ids.shape[-1] != 4:
            raise ValueError("text_ids must have shape (batch, sequence, 4)")
        if self.prompt_embeds.shape[:2] != self.text_ids.shape[:2]:
            raise ValueError("prompt_embeds and text_ids batch/sequence dimensions must match")
        if self.prompt_embeds.requires_grad:
            raise ValueError("cached prompt embeddings must not require gradients")

    @property
    def batch_size(self) -> int:
        return int(self.prompt_embeds.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.prompt_embeds.shape[1])

    @property
    def feature_width(self) -> int:
        return int(self.prompt_embeds.shape[2])

    @property
    def device(self) -> torch.device:
        return self.prompt_embeds.device

    @property
    def prompt_sha256(self) -> str:
        """Stable checksum of the resolved prompt text for resume checks."""

        canonical = json.dumps(
            self.prompt,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def prompt_hash(self) -> str:
        """Alias retained for checkpoint/manifest call sites."""

        return self.prompt_sha256

    def to(
        self,
        device: torch.device | str | None = None,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "ConditioningPackage":
        """Return a moved package while preserving integer position IDs."""

        embeds = self.prompt_embeds.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )
        ids = self.text_ids.to(device=device, non_blocking=non_blocking)
        return ConditioningPackage(self.prompt, embeds.detach(), ids.detach())

    def cpu(self) -> "ConditioningPackage":
        return self.to("cpu")

    def as_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            "prompt_embeds": self.prompt_embeds,
            "text_ids": self.text_ids,
        }


def stack_conditioning_packages(
    packages: tuple[ConditioningPackage, ...] | list[ConditioningPackage],
) -> ConditioningPackage:
    """Stack independent prompt packages into one transformer batch."""

    if not packages:
        raise ValueError("at least one conditioning package is required")
    first = packages[0]
    expected_embed_shape = first.prompt_embeds.shape[1:]
    expected_id_shape = first.text_ids.shape[1:]
    if any(package.prompt_embeds.shape[1:] != expected_embed_shape for package in packages):
        raise ValueError("conditioning embedding sequence/feature shapes must match")
    if any(package.text_ids.shape[1:] != expected_id_shape for package in packages):
        raise ValueError("conditioning text-ID sequence shapes must match")
    prompts: list[str] = []
    for package in packages:
        if isinstance(package.prompt, tuple):
            prompts.extend(package.prompt)
        else:
            prompts.append(package.prompt)
    return ConditioningPackage(
        prompt=tuple(prompts),
        prompt_embeds=torch.cat([package.prompt_embeds for package in packages], dim=0).detach(),
        text_ids=torch.cat([package.text_ids for package in packages], dim=0).detach(),
    )


@dataclass(frozen=True, slots=True)
class ResolvedPrompts:
    source: str
    target: str
    negative: str
    bridge: str | None


@dataclass(frozen=True, slots=True)
class ConditioningCache:
    source: ConditioningPackage
    target: ConditioningPackage
    unconditional: ConditioningPackage
    bridge: ConditioningPackage | None = None
    prompt_schedule: tuple[ConditioningPackage, ...] = ()

    def cpu(self) -> "ConditioningCache":
        return ConditioningCache(
            source=self.source.cpu(),
            target=self.target.cpu(),
            unconditional=self.unconditional.cpu(),
            bridge=self.bridge.cpu() if self.bridge is not None else None,
            prompt_schedule=tuple(package.cpu() for package in self.prompt_schedule),
        )

    def as_dict(self) -> dict[str, ConditioningPackage]:
        result = {
            "source": self.source,
            "target": self.target,
            "unconditional": self.unconditional,
        }
        if self.bridge is not None:
            result["bridge"] = self.bridge
        result.update({f"schedule_{index:03d}": package for index, package in enumerate(self.prompt_schedule)})
        return result

    @property
    def prompt_hashes(self) -> dict[str, str]:
        return {name: package.prompt_sha256 for name, package in self.as_dict().items()}


def _clean_optional_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    if not isinstance(prompt, str):
        raise TypeError("prompts must be strings or None")
    stripped = prompt.strip()
    return stripped or None


def resolve_prompts(
    *,
    source_prompt: str | None,
    target_prompt: str | None,
    bridge_prompt: str | None,
    negative_prompt: str = "",
    neutral_prompt: str = NEUTRAL_PROMPT,
) -> ResolvedPrompts:
    """Apply the documented endpoint prompt fallback policy."""

    source = _clean_optional_prompt(source_prompt)
    target = _clean_optional_prompt(target_prompt)
    bridge = _clean_optional_prompt(bridge_prompt)
    neutral = _clean_optional_prompt(neutral_prompt)
    if neutral is None:
        raise ValueError("neutral_prompt must contain non-whitespace text")
    if not isinstance(negative_prompt, str):
        raise TypeError("negative_prompt must be a string")
    return ResolvedPrompts(
        source=source or bridge or neutral,
        target=target or bridge or neutral,
        negative=negative_prompt,
        bridge=bridge,
    )


def encode_prompt_conditioning(
    pipeline: Any,
    prompt: str | list[str],
    *,
    device: torch.device | str | None = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int | None = None,
    text_encoder_out_layers: tuple[int, ...] | None = None,
) -> ConditioningPackage:
    """Encode through ``Flux2KleinPipeline.encode_prompt`` without gradients.

    Optional parameters are omitted when unspecified, letting the pinned
    pipeline provide its own defaults rather than hardcoding Qwen internals in
    this package.
    """

    encode_prompt = getattr(pipeline, "encode_prompt", None)
    if not callable(encode_prompt):
        raise TypeError("pipeline must expose a callable encode_prompt method")
    if isinstance(prompt, str):
        prompt_value: str | list[str] = prompt
        recorded_prompt: str | tuple[str, ...] = prompt
    elif isinstance(prompt, list) and prompt and all(isinstance(item, str) for item in prompt):
        prompt_value = prompt
        recorded_prompt = tuple(prompt)
    else:
        raise TypeError("prompt must be a string or a non-empty list of strings")
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive")

    kwargs: dict[str, Any] = {
        "prompt": prompt_value,
        "num_images_per_prompt": num_images_per_prompt,
    }
    if device is not None:
        kwargs["device"] = device
    if max_sequence_length is not None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        kwargs["max_sequence_length"] = max_sequence_length
    if text_encoder_out_layers is not None:
        if not text_encoder_out_layers:
            raise ValueError("text_encoder_out_layers must not be empty")
        kwargs["text_encoder_out_layers"] = text_encoder_out_layers

    with torch.no_grad():
        output = encode_prompt(**kwargs)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("pipeline.encode_prompt() must return prompt embeddings and text IDs")
    prompt_embeds, text_ids = output[:2]
    if not isinstance(prompt_embeds, torch.Tensor) or not isinstance(text_ids, torch.Tensor):
        raise TypeError("pipeline.encode_prompt() outputs must be torch tensors")
    return ConditioningPackage(recorded_prompt, prompt_embeds.detach(), text_ids.detach())


def build_conditioning_cache(
    pipeline: Any,
    *,
    source_prompt: str | None,
    target_prompt: str | None,
    bridge_prompt: str | None,
    bridge_prompts: tuple[str, ...] | list[str] | None = None,
    negative_prompt: str = "",
    neutral_prompt: str = NEUTRAL_PROMPT,
    device: torch.device | str | None = None,
    offload_to_cpu: bool = True,
    num_images_per_prompt: int = 1,
    max_sequence_length: int | None = None,
    text_encoder_out_layers: tuple[int, ...] | None = None,
) -> ConditioningCache:
    """Resolve and encode unique prompts into reusable fixed packages."""

    prompts = resolve_prompts(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        bridge_prompt=bridge_prompt,
        negative_prompt=negative_prompt,
        neutral_prompt=neutral_prompt,
    )
    encoded: dict[str, ConditioningPackage] = {}

    def get(prompt: str) -> ConditioningPackage:
        package = encoded.get(prompt)
        if package is None:
            package = encode_prompt_conditioning(
                pipeline,
                prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
            if offload_to_cpu:
                package = package.cpu()
            encoded[prompt] = package
        return package

    return ConditioningCache(
        source=get(prompts.source),
        target=get(prompts.target),
        unconditional=get(prompts.negative),
        bridge=get(prompts.bridge) if prompts.bridge is not None else None,
        prompt_schedule=tuple(get(prompt) for prompt in (bridge_prompts or ())),
    )


def interpolate_conditioning(
    source: ConditioningPackage,
    target: ConditioningPackage,
    alpha: float,
) -> ConditioningPackage:
    """Interpolate embeddings while retaining, never interpolating, token IDs."""

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if source.prompt_embeds.shape != target.prompt_embeds.shape:
        raise ValueError("conditioning embeddings must have identical shapes")
    if source.text_ids.shape != target.text_ids.shape or not torch.equal(
        source.text_ids.to("cpu"), target.text_ids.to("cpu")
    ):
        raise ValueError("text IDs must be identical; token IDs are never interpolated")
    target_embeds = target.prompt_embeds.to(
        device=source.prompt_embeds.device,
        dtype=source.prompt_embeds.dtype,
    )
    embeds = torch.lerp(source.prompt_embeds, target_embeds, alpha).detach()
    return ConditioningPackage(
        prompt=(f"interpolated:{alpha:.8g}",),
        prompt_embeds=embeds,
        text_ids=source.text_ids.detach(),
    )


def interpolate_conditioning_through_midpoint(
    source: ConditioningPackage,
    midpoint: ConditioningPackage,
    target: ConditioningPackage,
    alpha: float,
) -> ConditioningPackage:
    """Interpolate source→midpoint→target embeddings over the unit interval."""

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if alpha <= 0.5:
        return interpolate_conditioning(source, midpoint, alpha * 2.0)
    return interpolate_conditioning(midpoint, target, (alpha - 0.5) * 2.0)


def select_render_conditioning(
    cache: ConditioningCache,
    mode: str,
    alpha: float,
) -> ConditioningPackage:
    """Select one of the documented render-conditioning policies."""

    normalized_mode = str(mode).lower()
    if normalized_mode == "source":
        return cache.source
    if normalized_mode == "target":
        return cache.target
    if normalized_mode == "shared_bridge":
        if cache.bridge is None:
            raise ValueError("shared_bridge mode requires bridge conditioning")
        return cache.bridge
    if normalized_mode == "prompt_schedule":
        raise ValueError("prompt_schedule conditioning must be selected by frame index")
    if normalized_mode == "interpolated_embeddings":
        return interpolate_conditioning(cache.source, cache.target, alpha)
    if normalized_mode == "nearest_endpoint":
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and lie in [0, 1]")
        return cache.source if alpha < 0.5 else cache.target
    raise ValueError(f"unsupported render conditioning mode {mode!r}")
