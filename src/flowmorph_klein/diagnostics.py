"""Gradient, CUDA memory, and production-backward diagnostics."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch


class DiagnosticFailure(RuntimeError):
    """Raised when a mandatory numerical diagnostic does not pass."""


@dataclass(frozen=True)
class BackwardProbeReport:
    passed: bool
    loss: float
    velocity_input_gradient_norm: float
    pred_gradient_norm: float
    u_gradient_norm: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    elapsed_seconds: float
    latent_shape: tuple[int, ...]
    model_parameter_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "loss": self.loss,
            "velocity_input_gradient_norm": self.velocity_input_gradient_norm,
            "pred_gradient_norm": self.pred_gradient_norm,
            "u_gradient_norm": self.u_gradient_norm,
            "peak_allocated_vram_bytes": self.peak_allocated_vram_bytes,
            "peak_reserved_vram_bytes": self.peak_reserved_vram_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "latent_shape": list(self.latent_shape),
            "model_parameter_count": self.model_parameter_count,
        }


def cuda_memory_snapshot(device: torch.device | str = "cuda:0") -> dict[str, int | None]:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "free_bytes": None,
            "total_bytes": None,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(selected)),
        "reserved_bytes": int(torch.cuda.memory_reserved(selected)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def gradient_norm(tensor: torch.Tensor) -> float:
    if tensor.grad is None:
        raise DiagnosticFailure("Expected a gradient but found None")
    gradient = tensor.grad.detach().float()
    if not torch.isfinite(gradient).all():
        raise DiagnosticFailure("Gradient contains non-finite values")
    return float(torch.linalg.vector_norm(gradient).cpu())


def assert_frozen_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    count = 0
    for parameter in parameters:
        count += parameter.numel()
        if parameter.requires_grad:
            raise DiagnosticFailure("A model or adapter parameter still has requires_grad=True")
        if parameter.grad is not None:
            raise DiagnosticFailure("A frozen model or adapter parameter received a gradient")
    return count


def run_backward_probe(
    *,
    z: torch.Tensor,
    sigma_i: torch.Tensor | float,
    sigma_last: torch.Tensor | float,
    timestep: torch.Tensor,
    predict_velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    frozen_parameters: Iterable[torch.nn.Parameter],
) -> BackwardProbeReport:
    """Run the exact one-step FlowMorph backward arithmetic on supplied inputs."""

    device = z.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    pred = z.detach().float().clone().requires_grad_(True)
    u = torch.zeros_like(pred, dtype=torch.float32, requires_grad=True)
    delta_sigma = torch.as_tensor(sigma_last, device=device, dtype=torch.float32) - torch.as_tensor(
        sigma_i, device=device, dtype=torch.float32
    )
    with torch.set_grad_enabled(True):
        state = pred - delta_sigma * u
        state.retain_grad()
        velocity = predict_velocity(state, timestep)
        if not isinstance(velocity, torch.Tensor) or velocity.shape != state.shape:
            raise DiagnosticFailure(
                "Velocity predictor must return a tensor with the FlowMorph state shape"
            )
        if not velocity.requires_grad:
            raise DiagnosticFailure(
                "Transformer velocity is detached from autograd; input gradients are unsupported"
            )

        # The reconstruction includes a direct `state` term, so pred/u
        # gradients alone cannot prove that the transformer is differentiable
        # with respect to its input. Run a separate deterministic velocity-only
        # VJP before the exact reconstruction backward pass. `backward()` is
        # used rather than `autograd.grad()` for compatibility with activation
        # checkpointing implementations that use reentrant autograd.
        generator = torch.Generator(device=velocity.device)
        generator.manual_seed(0xF10A2)
        projection_weights = torch.empty_like(velocity, dtype=torch.float32)
        projection_weights.bernoulli_(0.5, generator=generator)
        projection_weights.mul_(2.0).sub_(1.0)
        velocity_projection = (
            velocity.float() * projection_weights
        ).sum() / max(1, velocity.numel())
        velocity_projection.backward()
        if state.grad is None:
            raise DiagnosticFailure(
                "Transformer velocity produced no gradient with respect to its input state"
            )
        velocity_input_gradient = state.grad.detach().float()
        if not torch.isfinite(velocity_input_gradient).all():
            raise DiagnosticFailure(
                "Transformer velocity input gradient contains non-finite values"
            )
        velocity_input_gradient_norm = float(
            torch.linalg.vector_norm(velocity_input_gradient).cpu()
        )
        if velocity_input_gradient_norm <= 0.0:
            raise DiagnosticFailure(
                "Transformer velocity input gradient is zero; differentiable input support "
                "was not demonstrated"
            )
        pred.grad = None
        u.grad = None
        state.grad = None
        del (
            velocity_projection,
            projection_weights,
            velocity_input_gradient,
            velocity,
            state,
        )

        # Recompute after the VJP so the mandatory probe is an ordinary exact
        # reconstruction backward, matching endpoint fitting semantics.
        state = pred - delta_sigma * u
        velocity = predict_velocity(state, timestep)
        reconstruction = state + delta_sigma * velocity.float()
        loss = torch.linalg.vector_norm(reconstruction - z.detach().float())
        loss.backward()

    pred_norm = gradient_norm(pred)
    u_norm = gradient_norm(u)
    parameter_count = assert_frozen_parameters(frozen_parameters)
    memory = cuda_memory_snapshot(device)
    passed = bool(torch.isfinite(loss).item() and pred_norm >= 0 and u_norm >= 0)
    if not passed:
        raise DiagnosticFailure("Production backward probe produced non-finite values")
    return BackwardProbeReport(
        passed=True,
        loss=float(loss.detach().cpu()),
        velocity_input_gradient_norm=velocity_input_gradient_norm,
        pred_gradient_norm=pred_norm,
        u_gradient_norm=u_norm,
        peak_allocated_vram_bytes=int(memory["peak_allocated_bytes"] or 0),
        peak_reserved_vram_bytes=int(memory["peak_reserved_bytes"] or 0),
        elapsed_seconds=time.perf_counter() - start,
        latent_shape=tuple(z.shape),
        model_parameter_count=parameter_count,
    )


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
