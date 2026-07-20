"""Endpoint fitting with isolated float32 FlowMorph master parameters."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from .flow_state import (
    FlowMorphEndpoint,
    one_step_reconstruction,
    reconstruction_loss,
    state_from_pred_and_u,
)
from .types import LossMode


Tensor = torch.Tensor


class EndpointVelocityPredictor(Protocol):
    """Minimal model interface needed by endpoint optimization."""

    def predict_velocity(
        self,
        state: Tensor,
        timestep: Any,
        conditioning: Any,
    ) -> Tensor: ...


class EndpointCheckpointCallback(Protocol):
    """Callback used to persist periodic endpoint and optimizer state."""

    def __call__(
        self,
        step: int,
        endpoint: FlowMorphEndpoint,
        optimizer: torch.optim.Optimizer,
        diagnostics: "OptimizationStepDiagnostics",
    ) -> None: ...


class EndpointDiagnosticsCallback(Protocol):
    """Per-step hook used to durably record loss/gradient diagnostics."""

    def __call__(self, diagnostics: "OptimizationStepDiagnostics") -> None: ...


@dataclass(frozen=True, slots=True)
class EndpointOptimizerConfig:
    optimization_steps: int = 100
    pred_learning_rate: float = 0.04
    u_learning_rate: float = 0.01
    weight_decay: float | None = 0.01
    loss_mode: LossMode | str = LossMode.CODE_L2_NORM
    lambda_delta: float = 0.0
    lambda_u: float = 0.0
    checkpoint_every: int = 25

    def __post_init__(self) -> None:
        if self.optimization_steps <= 0:
            raise ValueError("optimization_steps must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        for name in ("pred_learning_rate", "u_learning_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.weight_decay is None:
            # ``FlowMorphConfig.weight_decay=None`` means inherit the released
            # torch.optim.AdamW default, which is 0.01.
            object.__setattr__(self, "weight_decay", 0.01)
        for name in ("weight_decay", "lambda_delta", "lambda_u"):
            value = getattr(self, name)
            assert value is not None
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        try:
            LossMode(self.loss_mode)
        except ValueError as error:
            raise ValueError(f"unsupported loss mode: {self.loss_mode!r}") from error


@dataclass(frozen=True, slots=True)
class OptimizationStepDiagnostics:
    step: int
    total_loss: float
    reconstruction_loss: float
    regularization_loss: float
    pred_gradient_norm: float
    u_gradient_norm: float
    pred_parameter_norm: float
    u_parameter_norm: float
    delta_norm: float
    peak_allocated_vram_bytes: int | None
    peak_reserved_vram_bytes: int | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "step": self.step,
            "total_loss": self.total_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "regularization_loss": self.regularization_loss,
            "pred_gradient_norm": self.pred_gradient_norm,
            "u_gradient_norm": self.u_gradient_norm,
            "pred_parameter_norm": self.pred_parameter_norm,
            "u_parameter_norm": self.u_parameter_norm,
            "delta_norm": self.delta_norm,
            "peak_allocated_vram_bytes": self.peak_allocated_vram_bytes,
            "peak_reserved_vram_bytes": self.peak_reserved_vram_bytes,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class EndpointOptimizationResult:
    endpoint: FlowMorphEndpoint
    diagnostics: tuple[OptimizationStepDiagnostics, ...]
    completed_steps: int


def _prepare_master(
    value: Tensor,
    *,
    reference_shape: torch.Size,
    reference_device: torch.device,
    name: str,
) -> Tensor:
    if value.shape != reference_shape:
        raise ValueError(f"{name} shape does not match z")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    return value.detach().to(device=reference_device, dtype=torch.float32).clone()


def initialize_endpoint_parameters(
    z: Tensor,
    *,
    initial_delta: Tensor | None = None,
    initial_u: Tensor | None = None,
) -> tuple[Tensor, nn.Parameter, nn.Parameter]:
    """Create the detached target and released-code ``pred=z, u=0`` leaves."""

    if not z.is_floating_point():
        raise TypeError("z must use a floating-point dtype")
    z_master = z.detach().to(dtype=torch.float32).clone()
    delta_master = (
        torch.zeros_like(z_master)
        if initial_delta is None
        else _prepare_master(
            initial_delta,
            reference_shape=z.shape,
            reference_device=z.device,
            name="initial_delta",
        )
    )
    u_master = (
        torch.zeros_like(z_master)
        if initial_u is None
        else _prepare_master(
            initial_u,
            reference_shape=z.shape,
            reference_device=z.device,
            name="initial_u",
        )
    )
    pred = nn.Parameter(z_master + delta_master, requires_grad=True)
    u = nn.Parameter(u_master, requires_grad=True)
    return z_master, pred, u


def build_endpoint_optimizer(
    pred: nn.Parameter,
    u: nn.Parameter,
    config: EndpointOptimizerConfig,
) -> torch.optim.AdamW:
    """Build released-code AdamW groups with explicit decay for auditability."""

    if pred.dtype is not torch.float32 or u.dtype is not torch.float32:
        raise TypeError("pred and u master parameters must be float32")
    weight_decay = config.weight_decay
    assert weight_decay is not None
    return torch.optim.AdamW(
        [
            {
                "params": [pred],
                "lr": config.pred_learning_rate,
                "weight_decay": weight_decay,
                "group_name": "pred",
            },
            {
                "params": [u],
                "lr": config.u_learning_rate,
                "weight_decay": weight_decay,
                "group_name": "u",
            },
        ],
        weight_decay=weight_decay,
    )


def _predict_velocity(
    predictor: EndpointVelocityPredictor,
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
        raise ValueError("predicted velocity shape does not match FlowMorph state")
    if not output.is_floating_point():
        raise TypeError("predicted velocity must be floating point")
    if state.requires_grad and not output.requires_grad:
        raise RuntimeError(
            "predicted velocity is detached from the optimizable state; transformer input "
            "gradients are required"
        )
    return output


def _discover_predictor_parameters(
    predictor: EndpointVelocityPredictor,
    supplied: Iterable[nn.Parameter] | None,
) -> tuple[nn.Parameter, ...]:
    if supplied is not None:
        return tuple(supplied)
    parameters = getattr(predictor, "parameters", None)
    return tuple(parameters()) if callable(parameters) else ()


def _conditioning_tensors(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _conditioning_tensors(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _conditioning_tensors(item)


def _assert_frozen_inputs(
    parameters: tuple[nn.Parameter, ...],
    conditioning: Any,
) -> None:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if trainable:
        raise ValueError("all predictor/model/LoRA parameters must be frozen")
    for parameter in parameters:
        parameter.grad = None
    if any(tensor.requires_grad for tensor in _conditioning_tensors(conditioning)):
        raise ValueError("cached conditioning tensors must not require gradients")


def _finite_gradient(parameter: nn.Parameter, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"{name} did not receive a gradient")
    gradient = parameter.grad.detach().float()
    if not bool(torch.isfinite(gradient).all().item()):
        raise FloatingPointError(f"{name} received a non-finite gradient")
    return float(torch.linalg.vector_norm(gradient).item())


def _finite_loss(loss: Tensor) -> None:
    if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("endpoint loss must be one finite scalar")


def _vram_peaks(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _endpoint_snapshot(
    z: Tensor,
    pred: Tensor,
    u: Tensor,
    sigma_i: float | Tensor,
    sigma_last: float | Tensor,
    timestep_i: Any,
) -> FlowMorphEndpoint:
    return FlowMorphEndpoint(
        z=z.detach().clone(),
        delta=(pred.detach() - z).clone(),
        u=u.detach().clone(),
        sigma_i=sigma_i,
        sigma_last=sigma_last,
        timestep_i=timestep_i,
    )


def optimize_endpoint(
    z: Tensor,
    *,
    sigma_i: float | Tensor,
    sigma_last: float | Tensor,
    timestep_i: Any,
    predictor: EndpointVelocityPredictor,
    conditioning: Any,
    config: EndpointOptimizerConfig | None = None,
    initial_delta: Tensor | None = None,
    initial_u: Tensor | None = None,
    optimizer_state_dict: Mapping[str, Any] | None = None,
    start_step: int = 0,
    predictor_parameters: Iterable[nn.Parameter] | None = None,
    checkpoint_callback: EndpointCheckpointCallback | None = None,
    diagnostics_callback: EndpointDiagnosticsCallback | None = None,
) -> EndpointOptimizationResult:
    """Fit one endpoint while keeping every non-FlowMorph input frozen.

    ``pred`` and ``u`` are float32 leaf parameters.  The predictor may cast
    the state to its compute dtype internally; ordinary PyTorch casts retain
    the gradient path back to these masters.
    """

    if torch.is_inference_mode_enabled():
        raise RuntimeError("endpoint optimization cannot run inside torch.inference_mode()")
    settings = config or EndpointOptimizerConfig()
    if start_step < 0 or start_step >= settings.optimization_steps:
        raise ValueError("start_step must be in [0, optimization_steps)")

    z_master, pred, u = initialize_endpoint_parameters(
        z,
        initial_delta=initial_delta,
        initial_u=initial_u,
    )
    model_parameters = _discover_predictor_parameters(predictor, predictor_parameters)
    _assert_frozen_inputs(model_parameters, conditioning)

    optimizer = build_endpoint_optimizer(pred, u, settings)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(dict(optimizer_state_dict))
    optimizer.zero_grad(set_to_none=True)

    history: list[OptimizationStepDiagnostics] = []
    started = time.perf_counter()
    device = z_master.device
    loss_mode = LossMode(settings.loss_mode)

    with torch.set_grad_enabled(True):
        for step in range(start_step + 1, settings.optimization_steps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            state = state_from_pred_and_u(pred, u, sigma_i, sigma_last)
            velocity = _predict_velocity(predictor, state, timestep_i, conditioning)
            reconstructed = one_step_reconstruction(
                state,
                velocity,
                sigma_i,
                sigma_last,
            )
            reconstruction = reconstruction_loss(reconstructed, z_master, loss_mode)
            delta = pred - z_master
            regularization = (
                settings.lambda_delta * delta.float().square().sum()
                + settings.lambda_u * u.float().square().sum()
            )
            total = reconstruction + regularization
            _finite_loss(total)
            total.backward()

            pred_gradient_norm = _finite_gradient(pred, "pred")
            u_gradient_norm = _finite_gradient(u, "u")
            if any(parameter.grad is not None for parameter in model_parameters):
                raise RuntimeError("a frozen predictor/model/LoRA parameter received a gradient")

            optimizer.step()
            peak_allocated, peak_reserved = _vram_peaks(device)
            diagnostics = OptimizationStepDiagnostics(
                step=step,
                total_loss=float(total.detach().float().item()),
                reconstruction_loss=float(reconstruction.detach().float().item()),
                regularization_loss=float(regularization.detach().float().item()),
                pred_gradient_norm=pred_gradient_norm,
                u_gradient_norm=u_gradient_norm,
                pred_parameter_norm=float(torch.linalg.vector_norm(pred.detach().float()).item()),
                u_parameter_norm=float(torch.linalg.vector_norm(u.detach().float()).item()),
                delta_norm=float(torch.linalg.vector_norm((pred.detach() - z_master).float()).item()),
                peak_allocated_vram_bytes=peak_allocated,
                peak_reserved_vram_bytes=peak_reserved,
                elapsed_seconds=time.perf_counter() - started,
            )
            history.append(diagnostics)

            if diagnostics_callback is not None:
                diagnostics_callback(diagnostics)

            if checkpoint_callback is not None and (
                step % settings.checkpoint_every == 0
                or step == settings.optimization_steps
            ):
                checkpoint_callback(
                    step,
                    _endpoint_snapshot(
                        z_master,
                        pred,
                        u,
                        sigma_i,
                        sigma_last,
                        timestep_i,
                    ),
                    optimizer,
                    diagnostics,
                )
            optimizer.zero_grad(set_to_none=True)

    endpoint = _endpoint_snapshot(
        z_master,
        pred,
        u,
        sigma_i,
        sigma_last,
        timestep_i,
    )
    return EndpointOptimizationResult(
        endpoint=endpoint,
        diagnostics=tuple(history),
        completed_steps=settings.optimization_steps,
    )
