"""Numerically safe interpolation of fitted FlowMorph endpoints."""

from __future__ import annotations

import math

import torch

from .flow_state import FlowMorphEndpoint


Tensor = torch.Tensor


def _coerce_alpha(alpha: float | Tensor) -> float:
    value = torch.as_tensor(alpha, dtype=torch.float64)
    if value.numel() != 1:
        raise ValueError("alpha must be scalar")
    result = float(value.item())
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    return result


def _validate_pair(a: Tensor, b: Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError("interpolated tensors must have identical shapes")
    if a.device != b.device:
        raise ValueError("interpolated tensors must be on the same device")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("SLERP requires floating-point tensors")


def _deterministic_orthogonal(unit: Tensor, eps: float) -> Tensor | None:
    """Choose a reproducible unit vector orthogonal to ``unit``."""

    flat = unit.reshape(-1)
    if flat.numel() < 2:
        return None
    basis = torch.zeros_like(flat)
    basis[int(torch.argmin(flat.abs()).item())] = 1.0
    orthogonal = basis - torch.dot(basis, flat) * flat
    norm = torch.linalg.vector_norm(orthogonal)
    if float(norm.item()) <= eps:
        return None
    return (orthogonal / norm).reshape_as(unit)


def slerp(
    a: Tensor,
    b: Tensor,
    alpha: float | Tensor,
    *,
    eps: float = 1e-7,
) -> Tensor:
    """Spherically interpolate two tensors with deterministic edge cases.

    Angles and norms are always calculated in float32.  Near-parallel inputs
    use linear interpolation.  Antipodal inputs use a deterministic
    orthogonal great-circle direction instead of dividing by ``sin(pi)``.
    If either input has near-zero norm, linear interpolation is the only
    well-defined continuous fallback.
    """

    _validate_pair(a, b)
    amount = _coerce_alpha(alpha)
    if amount == 0.0:
        return a.clone()
    if amount == 1.0:
        return b.clone()
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    output_dtype = a.dtype
    a32 = a.float()
    b32 = b.float()
    norm_a = torch.linalg.vector_norm(a32)
    norm_b = torch.linalg.vector_norm(b32)
    if float(norm_a.item()) <= eps or float(norm_b.item()) <= eps:
        return torch.lerp(a32, b32, amount).to(output_dtype)

    direction_a = a32 / norm_a
    direction_b = b32 / norm_b
    cosine = torch.clamp(
        torch.sum(direction_a.reshape(-1) * direction_b.reshape(-1)),
        -1.0,
        1.0,
    )
    cosine_value = float(cosine.item())

    if cosine_value >= 1.0 - eps:
        return torch.lerp(a32, b32, amount).to(output_dtype)

    if cosine_value <= -1.0 + eps:
        orthogonal = _deterministic_orthogonal(direction_a, eps)
        if orthogonal is None:
            return torch.lerp(a32, b32, amount).to(output_dtype)
        angle = math.pi * amount
        magnitude = (1.0 - amount) * norm_a + amount * norm_b
        direction = math.cos(angle) * direction_a + math.sin(angle) * orthogonal
        return (magnitude * direction).to(output_dtype)

    omega = torch.arccos(cosine)
    sin_omega = torch.sin(omega)
    if float(sin_omega.abs().item()) <= eps:
        return torch.lerp(a32, b32, amount).to(output_dtype)
    weight_a = torch.sin((1.0 - amount) * omega) / sin_omega
    weight_b = torch.sin(amount * omega) / sin_omega
    return (weight_a * a32 + weight_b * b32).to(output_dtype)


def slerp_direction_and_magnitude(
    u_source: Tensor,
    u_target: Tensor,
    alpha: float | Tensor,
    *,
    eps: float = 1e-7,
) -> Tensor:
    """SLERP ``u`` direction and linearly interpolate its global magnitude."""

    _validate_pair(u_source, u_target)
    amount = _coerce_alpha(alpha)
    if amount == 0.0:
        return u_source.clone()
    if amount == 1.0:
        return u_target.clone()

    output_dtype = u_source.dtype
    source = u_source.float()
    target = u_target.float()
    source_norm = torch.linalg.vector_norm(source)
    target_norm = torch.linalg.vector_norm(target)
    source_is_zero = float(source_norm.item()) <= eps
    target_is_zero = float(target_norm.item()) <= eps

    if source_is_zero and target_is_zero:
        return torch.zeros_like(u_source)
    if source_is_zero:
        direction = target / target_norm
    elif target_is_zero:
        direction = source / source_norm
    else:
        direction = slerp(source / source_norm, target / target_norm, amount, eps=eps).float()
        direction_norm = torch.linalg.vector_norm(direction)
        if float(direction_norm.item()) <= eps:
            # This is reachable only in a one-dimensional antipodal space.
            return torch.lerp(source, target, amount).to(output_dtype)
        direction = direction / direction_norm

    magnitude = (1.0 - amount) * source_norm + amount * target_norm
    return (magnitude * direction).to(output_dtype)


def _same_schedule_value(a: float | Tensor, b: float | Tensor) -> bool:
    a_value = torch.as_tensor(a, dtype=torch.float64)
    b_value = torch.as_tensor(b, dtype=torch.float64)
    return a_value.numel() == b_value.numel() == 1 and bool(torch.equal(a_value, b_value))


def interpolate_endpoint(
    source: FlowMorphEndpoint,
    target: FlowMorphEndpoint,
    alpha: float | Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> FlowMorphEndpoint:
    """Interpolate ``z``/``delta`` linearly and ``u`` with decoupled SLERP."""

    _validate_pair(source.z, target.z)
    _validate_pair(source.delta, target.delta)
    _validate_pair(source.u, target.u)
    if not _same_schedule_value(source.sigma_i, target.sigma_i):
        raise ValueError("source and target use different start sigmas")
    if not _same_schedule_value(source.sigma_last, target.sigma_last):
        raise ValueError("source and target use different terminal sigmas")

    amount = _coerce_alpha(alpha)
    dtype = output_dtype or source.z.dtype
    if amount == 0.0:
        return FlowMorphEndpoint(
            z=source.z.detach().clone().to(dtype),
            delta=source.delta.detach().clone().to(dtype),
            u=source.u.detach().clone().to(dtype),
            sigma_i=source.sigma_i,
            sigma_last=source.sigma_last,
            timestep_i=source.timestep_i,
        )
    if amount == 1.0:
        return FlowMorphEndpoint(
            z=target.z.detach().clone().to(dtype),
            delta=target.delta.detach().clone().to(dtype),
            u=target.u.detach().clone().to(dtype),
            sigma_i=target.sigma_i,
            sigma_last=target.sigma_last,
            timestep_i=target.timestep_i,
        )

    z = torch.lerp(source.z.float(), target.z.float(), amount).to(dtype)
    delta = torch.lerp(source.delta.float(), target.delta.float(), amount).to(dtype)
    u = slerp_direction_and_magnitude(source.u, target.u, amount).to(dtype)
    return FlowMorphEndpoint(
        z=z,
        delta=delta,
        u=u,
        sigma_i=source.sigma_i,
        sigma_last=source.sigma_last,
        timestep_i=source.timestep_i,
    )


def interpolate_flowmorph_state(
    source: FlowMorphEndpoint,
    target: FlowMorphEndpoint,
    alpha: float | Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Construct the interpolated frame's short-trajectory starting state."""

    return interpolate_endpoint(source, target, alpha, output_dtype=output_dtype).state
