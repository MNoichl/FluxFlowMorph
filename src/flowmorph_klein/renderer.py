"""Sparse FlowMorph rendering against a fakeable velocity-predictor protocol."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Protocol

import torch

from .flow_schedule import FlowSchedule, RenderStep, euler_flow_update, get_render_chain
from .flow_state import FlowMorphEndpoint
from .interpolation import interpolate_endpoint
from .types import RenderConditioningMode


Tensor = torch.Tensor


class VelocityPredictor(Protocol):
    """The only model operation required by the latent renderer."""

    def predict_velocity(
        self,
        state: Tensor,
        timestep: Any,
        conditioning: Any,
    ) -> Tensor: ...


ConditioningInterpolator = Callable[[Any, Any, float], Any]


@dataclass(frozen=True, slots=True)
class RenderedLatentFrame:
    index: int
    alpha: float
    start_state: Tensor
    final_latent: Tensor
    conditioning_mode: RenderConditioningMode


def _alpha_value(alpha: float | Tensor) -> float:
    value = torch.as_tensor(alpha, dtype=torch.float64)
    if value.numel() != 1:
        raise ValueError("alpha must be scalar")
    result = float(value.item())
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    return result


def linear_alphas(frame_count: int = 20) -> tuple[float, ...]:
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    return tuple(float(value) for value in torch.linspace(0.0, 1.0, frame_count, dtype=torch.float64))


def select_render_conditioning(
    mode: RenderConditioningMode | str,
    alpha: float | Tensor,
    *,
    source_conditioning: Any,
    target_conditioning: Any,
    bridge_conditioning: Any = None,
    conditioning_interpolator: ConditioningInterpolator | None = None,
) -> Any:
    """Select conditioning without making assumptions about package layout."""

    try:
        selected_mode = mode if isinstance(mode, RenderConditioningMode) else RenderConditioningMode(mode)
    except ValueError as error:
        raise ValueError(f"unsupported rendering conditioning mode: {mode!r}") from error
    amount = _alpha_value(alpha)

    if selected_mode is RenderConditioningMode.SOURCE:
        return source_conditioning
    if selected_mode is RenderConditioningMode.TARGET:
        return target_conditioning
    if selected_mode is RenderConditioningMode.SHARED_BRIDGE:
        if bridge_conditioning is None:
            raise ValueError("shared_bridge rendering requires bridge_conditioning")
        return bridge_conditioning
    if selected_mode is RenderConditioningMode.NEAREST_ENDPOINT:
        return source_conditioning if amount < 0.5 else target_conditioning
    if selected_mode is RenderConditioningMode.INTERPOLATED_EMBEDDINGS:
        if conditioning_interpolator is None:
            raise ValueError(
                "interpolated_embeddings requires an explicit compatible-embedding interpolator"
            )
        return conditioning_interpolator(source_conditioning, target_conditioning, amount)
    raise AssertionError(f"unhandled conditioning mode: {selected_mode}")


def _predict_velocity(
    predictor: VelocityPredictor,
    state: Tensor,
    timestep: Any,
    conditioning: Any,
) -> Tensor:
    method = getattr(predictor, "predict_velocity", None)
    if callable(method):
        output = method(state, timestep, conditioning)
    elif callable(predictor):
        output = predictor(state, timestep, conditioning)  # type: ignore[operator]
    else:
        raise TypeError("predictor must define predict_velocity or be callable")
    if not isinstance(output, Tensor):
        raise TypeError("predictor must return a torch.Tensor")
    if output.shape != state.shape:
        raise ValueError("velocity shape does not match rendering state")
    return output


def _require_finite_render_tensor(
    tensor: Tensor,
    quantity: str,
    *,
    frame_index: int | None = None,
    step: RenderStep | None = None,
) -> None:
    if bool(torch.isfinite(tensor).all().item()):
        return
    context: list[str] = []
    if frame_index is not None:
        context.append(f"frame {frame_index}")
    if step is not None:
        context.append(
            f"render step {step.chain_position} (scheduler index {step.current_index})"
        )
    location = " at " + ", ".join(context) if context else ""
    raise FloatingPointError(f"{quantity} contains non-finite values{location}")


def render_latent_trajectory(
    initial_state: Tensor,
    *,
    predictor: VelocityPredictor,
    conditioning: Any,
    render_chain: Sequence[RenderStep],
    frame_index: int | None = None,
) -> Tensor:
    """Advance one interpolated state through the sparse deterministic chain."""

    if not render_chain:
        raise ValueError("render_chain must not be empty")
    _require_finite_render_tensor(
        initial_state,
        "initial render state",
        frame_index=frame_index,
    )
    state = initial_state
    for step in render_chain:
        timestep = step.timestep.to(device=state.device)
        velocity = _predict_velocity(predictor, state, timestep, conditioning)
        _require_finite_render_tensor(
            velocity,
            "predicted velocity",
            frame_index=frame_index,
            step=step,
        )
        state = euler_flow_update(
            state,
            velocity,
            step.current_sigma,
            step.next_sigma,
        )
        _require_finite_render_tensor(
            state,
            "Euler-updated render state",
            frame_index=frame_index,
            step=step,
        )
    _require_finite_render_tensor(
        state,
        "final rendered latent",
        frame_index=frame_index,
    )
    return state


def render_morph(
    source: FlowMorphEndpoint,
    target: FlowMorphEndpoint,
    *,
    schedule: FlowSchedule,
    predictor: VelocityPredictor,
    source_conditioning: Any,
    target_conditioning: Any,
    alphas: Iterable[float] | None = None,
    frame_count: int = 20,
    render_indices: Sequence[int] = (35, 55, 75, 95),
    conditioning_mode: RenderConditioningMode | str = RenderConditioningMode.SOURCE,
    bridge_conditioning: Any = None,
    conditioning_interpolator: ConditioningInterpolator | None = None,
    output_dtype: torch.dtype | None = None,
    use_inference_mode: bool = True,
) -> tuple[RenderedLatentFrame, ...]:
    """Interpolate and render all raw algorithmic latent frames."""

    try:
        selected_mode = (
            conditioning_mode
            if isinstance(conditioning_mode, RenderConditioningMode)
            else RenderConditioningMode(conditioning_mode)
        )
    except ValueError as error:
        raise ValueError(f"unsupported rendering conditioning mode: {conditioning_mode!r}") from error
    coefficients = linear_alphas(frame_count) if alphas is None else tuple(_alpha_value(alpha) for alpha in alphas)
    if not coefficients:
        raise ValueError("at least one alpha is required")
    chain = get_render_chain(schedule, render_indices)

    context: ContextManager[Any]
    context = torch.inference_mode() if use_inference_mode else nullcontext()
    frames: list[RenderedLatentFrame] = []
    with context:
        for index, alpha in enumerate(coefficients):
            endpoint = interpolate_endpoint(
                source,
                target,
                alpha,
                output_dtype=output_dtype,
            )
            conditioning = select_render_conditioning(
                selected_mode,
                alpha,
                source_conditioning=source_conditioning,
                target_conditioning=target_conditioning,
                bridge_conditioning=bridge_conditioning,
                conditioning_interpolator=conditioning_interpolator,
            )
            start_state = endpoint.state
            final_latent = render_latent_trajectory(
                start_state,
                predictor=predictor,
                conditioning=conditioning,
                render_chain=chain,
                frame_index=index,
            )
            frames.append(
                RenderedLatentFrame(
                    index=index,
                    alpha=alpha,
                    start_state=start_state.detach().clone(),
                    final_latent=final_latent.detach().clone(),
                    conditioning_mode=selected_mode,
                )
            )
    return tuple(frames)
