"""FLUX.2 Klein schedule construction and sparse Euler chains.

Diffusers is intentionally not imported at module import time.  Production
callers normally pass the scheduler loaded with ``Flux2KleinPipeline``;
CPU-only tests can instead supply already materialized timestep and sigma
arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


Tensor = torch.Tensor


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Reproduce the pinned ``Flux2KleinPipeline`` empirical shift formula."""

    if image_seq_len <= 0:
        raise ValueError("image_seq_len must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def klein_custom_sigmas(num_inference_steps: int) -> tuple[float, ...]:
    """Return Klein's pre-shift custom sigma inputs.

    The pipeline supplies ``linspace(1, 1 / N, N)`` to the scheduler, which
    then performs its configured shift and appends the terminal sigma.
    """

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    return tuple(
        float(value)
        for value in np.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
        )
    )


def _to_1d_float_tensor(values: Sequence[float] | Tensor, name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().clone()
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def validate_sigma_order(
    sigmas: Sequence[float] | Tensor,
    *,
    strictly_decreasing: bool = True,
) -> bool:
    """Validate the denoising order and return ``True`` on success."""

    values = _to_1d_float_tensor(sigmas, "sigmas")
    if values.numel() < 2:
        raise ValueError("a sigma schedule needs at least two values")
    differences = values[1:] - values[:-1]
    valid = differences.lt(0).all() if strictly_decreasing else differences.le(0).all()
    if not bool(valid.item()):
        relation = "strictly decreasing" if strictly_decreasing else "non-increasing"
        raise ValueError(f"sigma schedule must be {relation}")
    return True


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _serializable_scheduler_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        source = dict(config)
    elif hasattr(config, "to_dict"):
        source = dict(config.to_dict())
    else:
        source = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    serializable: dict[str, Any] = {}
    for key, value in source.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            serializable[str(key)] = value
        elif isinstance(value, (tuple, list)):
            serializable[str(key)] = list(value)
        else:
            serializable[str(key)] = str(value)
    return serializable


@dataclass(frozen=True, slots=True)
class FlowSchedule:
    """Materialized scheduler points including the appended terminal sigma."""

    timesteps: Tensor
    sigmas: Tensor
    image_seq_len: int | None = None
    mu: float | None = None
    scheduler_configuration: Mapping[str, Any] | None = None
    used_klein_custom_sigmas: bool = False

    def __post_init__(self) -> None:
        timesteps = _to_1d_float_tensor(self.timesteps, "timesteps")
        sigmas = _to_1d_float_tensor(self.sigmas, "sigmas")
        if sigmas.numel() != timesteps.numel() + 1:
            raise ValueError("sigmas must contain one terminal value beyond timesteps")
        validate_sigma_order(sigmas)
        if self.image_seq_len is not None and self.image_seq_len <= 0:
            raise ValueError("image_seq_len must be positive")
        if self.mu is not None and not np.isfinite(self.mu):
            raise ValueError("mu must be finite")
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(
            self,
            "scheduler_configuration",
            dict(self.scheduler_configuration or {}),
        )

    @property
    def num_inference_steps(self) -> int:
        return int(self.timesteps.numel())

    @property
    def sigma_last(self) -> float:
        return float(self.sigmas[-1].item())


def create_flow_match_scheduler(config: Mapping[str, Any]) -> Any:
    """Create a Diffusers scheduler without importing Diffusers eagerly."""

    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except ImportError as error:  # pragma: no cover - exercised only in production environments
        raise RuntimeError("Diffusers is required to construct a scheduler from config") from error
    return FlowMatchEulerDiscreteScheduler.from_config(dict(config))


def _resolve_image_seq_len(
    image_seq_len: int | None,
    packed_latents: Tensor | None,
) -> int | None:
    if packed_latents is not None:
        if packed_latents.ndim < 2:
            raise ValueError("packed_latents must expose token count at dimension 1")
        inferred = int(packed_latents.shape[1])
        if image_seq_len is not None and image_seq_len != inferred:
            raise ValueError("image_seq_len disagrees with packed_latents.shape[1]")
        image_seq_len = inferred
    if image_seq_len is not None and image_seq_len <= 0:
        raise ValueError("image_seq_len must be positive")
    return image_seq_len


def build_flowmorph_schedule(
    scheduler: Any | None = None,
    *,
    scheduler_points: int | None = None,
    image_seq_len: int | None = None,
    packed_latents: Tensor | None = None,
    device: str | torch.device | None = None,
    timesteps: Sequence[float] | Tensor | None = None,
    sigmas: Sequence[float] | Tensor | None = None,
) -> FlowSchedule:
    """Build the exact Klein schedule or wrap supplied arrays.

    With a real scheduler, token count must come from ``image_seq_len`` or
    ``packed_latents.shape[1]``.  The pinned pipeline computes empirical
    ``mu`` from that count and supplies Klein's custom sigma linspace unless
    the scheduler advertises ``use_flow_sigmas``.  When ``scheduler`` is
    omitted, both materialized arrays must be supplied; this path has no
    Diffusers dependency and is intended for numerical tests and replay.
    """

    image_seq_len = _resolve_image_seq_len(image_seq_len, packed_latents)
    if scheduler is None:
        if timesteps is None or sigmas is None:
            raise ValueError("supply a scheduler or both materialized timesteps and sigmas")
        timestep_tensor = _to_1d_float_tensor(timesteps, "timesteps")
        sigma_tensor = _to_1d_float_tensor(sigmas, "sigmas")
        inferred_steps = int(timestep_tensor.numel())
        if scheduler_points is not None and scheduler_points != inferred_steps:
            raise ValueError("scheduler_points disagrees with supplied timesteps")
        mu = (
            compute_empirical_mu(image_seq_len, inferred_steps)
            if image_seq_len is not None
            else None
        )
        return FlowSchedule(
            timesteps=timestep_tensor,
            sigmas=sigma_tensor,
            image_seq_len=image_seq_len,
            mu=mu,
            scheduler_configuration={},
            used_klein_custom_sigmas=False,
        )

    if timesteps is not None or sigmas is not None:
        raise ValueError("materialized arrays cannot be combined with a scheduler")
    points = 100 if scheduler_points is None else scheduler_points
    if points <= 0:
        raise ValueError("scheduler_points must be positive")
    if image_seq_len is None:
        raise ValueError("a real Klein scheduler requires the actual packed image token count")

    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=points)
    scheduler_config = getattr(scheduler, "config", None)
    uses_internal_flow_sigmas = bool(
        _config_value(scheduler_config, "use_flow_sigmas", False)
    )
    if uses_internal_flow_sigmas:
        scheduler.set_timesteps(
            num_inference_steps=points,
            device=device,
            mu=mu,
        )
    else:
        scheduler.set_timesteps(
            sigmas=klein_custom_sigmas(points),
            device=device,
            mu=mu,
        )

    return FlowSchedule(
        timesteps=torch.as_tensor(scheduler.timesteps),
        sigmas=torch.as_tensor(scheduler.sigmas),
        image_seq_len=image_seq_len,
        mu=mu,
        scheduler_configuration=_serializable_scheduler_config(scheduler_config),
        used_klein_custom_sigmas=not uses_internal_flow_sigmas,
    )


@dataclass(frozen=True, slots=True)
class StartStateMetadata:
    start_index: int
    timestep_i: float
    sigma_i: float
    sigma_last: float
    delta_sigma: float
    image_seq_len: int | None
    mu: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "start_index": self.start_index,
            "timestep_i": self.timestep_i,
            "sigma_i": self.sigma_i,
            "sigma_last": self.sigma_last,
            "delta_sigma": self.delta_sigma,
            "image_seq_len": self.image_seq_len,
            "mu": self.mu,
        }


def get_start_state_metadata(schedule: FlowSchedule, start_index: int) -> StartStateMetadata:
    if start_index < 0 or start_index >= schedule.num_inference_steps:
        raise IndexError("start_index is outside the scheduler points")
    sigma_i = float(schedule.sigmas[start_index].item())
    sigma_last = schedule.sigma_last
    return StartStateMetadata(
        start_index=start_index,
        timestep_i=float(schedule.timesteps[start_index].item()),
        sigma_i=sigma_i,
        sigma_last=sigma_last,
        delta_sigma=sigma_last - sigma_i,
        image_seq_len=schedule.image_seq_len,
        mu=schedule.mu,
    )


@dataclass(frozen=True, slots=True)
class RenderStep:
    chain_position: int
    current_index: int
    next_index: int | None
    timestep: Tensor
    current_sigma: float
    next_sigma: float

    @property
    def sigma_delta(self) -> float:
        return self.next_sigma - self.current_sigma

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "chain_position": self.chain_position,
            "current_index": self.current_index,
            "next_index": self.next_index,
            "timestep": float(self.timestep.item()),
            "current_sigma": self.current_sigma,
            "next_sigma": self.next_sigma,
            "sigma_delta": self.sigma_delta,
        }


def get_render_chain(
    schedule: FlowSchedule,
    render_indices: Sequence[int] = (35, 55, 75, 95),
) -> tuple[RenderStep, ...]:
    """Resolve sparse scheduler indices, including the final terminal step."""

    indices = tuple(int(index) for index in render_indices)
    if not indices:
        raise ValueError("render_indices must not be empty")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("render_indices must be strictly increasing")
    if indices[0] < 0 or indices[-1] >= schedule.num_inference_steps:
        raise IndexError("a render index is outside the scheduler points")

    chain: list[RenderStep] = []
    for position, current_index in enumerate(indices):
        next_index = indices[position + 1] if position + 1 < len(indices) else None
        next_sigma = (
            float(schedule.sigmas[next_index].item())
            if next_index is not None
            else schedule.sigma_last
        )
        chain.append(
            RenderStep(
                chain_position=position,
                current_index=current_index,
                next_index=next_index,
                timestep=schedule.timesteps[current_index].detach().clone(),
                current_sigma=float(schedule.sigmas[current_index].item()),
                next_sigma=next_sigma,
            )
        )
    return tuple(chain)


def euler_flow_update(
    state: Tensor,
    velocity: Tensor,
    current_sigma: float | Tensor,
    next_sigma: float | Tensor,
) -> Tensor:
    """Apply Diffusers' deterministic flow Euler update with exact sign.

    Like the pinned scheduler, the sample is upcast to float32 for the update
    and the result is cast to the model-output dtype.
    """

    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have identical shapes")
    if not state.is_floating_point() or not velocity.is_floating_point():
        raise TypeError("Euler updates require floating-point tensors")
    current = torch.as_tensor(current_sigma, device=state.device, dtype=torch.float32)
    following = torch.as_tensor(next_sigma, device=state.device, dtype=torch.float32)
    if current.numel() != 1 or following.numel() != 1:
        raise ValueError("Euler sigmas must be scalar")
    if not bool(torch.isfinite(current).item()) or not bool(torch.isfinite(following).item()):
        raise ValueError("Euler sigmas must be finite")
    if bool((following > current).item()):
        raise ValueError("denoising Euler updates require next_sigma <= current_sigma")
    updated = state.to(torch.float32) + (following - current) * velocity
    return updated.to(velocity.dtype)
