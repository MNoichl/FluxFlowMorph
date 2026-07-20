"""Pure tensor equations for the FlowMorph endpoint parameterization.

The released FlowMorph code uses ``pred = z + delta`` and constructs the
optimizable state as ``pred - (sigma_last - sigma_i) * u``.  Keeping these
small operations in one module makes the sign convention independently
testable and avoids coupling the numerical core to Diffusers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .types import LossMode


Tensor = torch.Tensor


def _validate_same_shape(*tensors: Tensor) -> None:
    if not tensors:
        raise ValueError("at least one tensor is required")
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors[1:]):
        raise ValueError("FlowMorph tensors must have identical shapes")


def _scalar_like(value: float | Tensor, reference: Tensor, name: str) -> Tensor:
    scalar = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    if not bool(torch.isfinite(scalar).item()):
        raise ValueError(f"{name} must be finite")
    return scalar.reshape(())


def sigma_delta(
    sigma_i: float | Tensor,
    sigma_last: float | Tensor,
    *,
    like: Tensor,
) -> Tensor:
    """Return the signed FlowMorph interval ``sigma_last - sigma_i``."""

    current = _scalar_like(sigma_i, like, "sigma_i")
    terminal = _scalar_like(sigma_last, like, "sigma_last")
    return terminal - current


def state_from_pred_and_u(
    pred: Tensor,
    u: Tensor,
    sigma_i: float | Tensor,
    sigma_last: float | Tensor,
) -> Tensor:
    """Construct ``state = pred - (sigma_last - sigma_i) * u``."""

    _validate_same_shape(pred, u)
    return pred - sigma_delta(sigma_i, sigma_last, like=pred) * u


def construct_flow_state(
    z: Tensor,
    delta: Tensor,
    u: Tensor,
    sigma_i: float | Tensor,
    sigma_last: float | Tensor,
) -> Tensor:
    """Construct ``(z + delta) - (sigma_last - sigma_i) * u``."""

    _validate_same_shape(z, delta, u)
    return state_from_pred_and_u(z + delta, u, sigma_i, sigma_last)


def one_step_reconstruction(
    state: Tensor,
    velocity: Tensor,
    sigma_i: float | Tensor,
    sigma_last: float | Tensor,
) -> Tensor:
    """Apply the released one-step reconstruction sign convention.

    ``z_hat = state + (sigma_last - sigma_i) * velocity``
    """

    _validate_same_shape(state, velocity)
    return state + sigma_delta(sigma_i, sigma_last, like=state) * velocity


def reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    loss_mode: LossMode | str = LossMode.CODE_L2_NORM,
) -> Tensor:
    """Calculate a float32 endpoint reconstruction loss.

    ``code_l2_norm`` is the released-code objective: one unsquared vector
    norm over the complete residual.  The project contract defines
    ``paper_l2_squared`` as mean squared error so its scale does not depend on
    latent token count.
    """

    _validate_same_shape(reconstruction, target)
    try:
        mode = loss_mode if isinstance(loss_mode, LossMode) else LossMode(loss_mode)
    except ValueError as error:
        raise ValueError(f"unsupported FlowMorph loss mode: {loss_mode!r}") from error

    residual = reconstruction.float() - target.float()
    if mode is LossMode.CODE_L2_NORM:
        return torch.linalg.vector_norm(residual)
    if mode is LossMode.PAPER_L2_SQUARED:
        return residual.square().mean()
    raise AssertionError(f"unhandled loss mode: {mode}")


@dataclass(frozen=True, slots=True)
class FlowMorphEndpoint:
    """Explicit fitted endpoint tensors and their schedule location."""

    z: Tensor
    delta: Tensor
    u: Tensor
    sigma_i: float | Tensor
    sigma_last: float | Tensor
    timestep_i: Any = None

    def __post_init__(self) -> None:
        _validate_same_shape(self.z, self.delta, self.u)
        if not all(tensor.is_floating_point() for tensor in (self.z, self.delta, self.u)):
            raise TypeError("FlowMorph endpoint tensors must use floating-point dtypes")
        if len({tensor.device for tensor in (self.z, self.delta, self.u)}) != 1:
            raise ValueError("FlowMorph endpoint tensors must be on one device")
        if len({tensor.dtype for tensor in (self.z, self.delta, self.u)}) != 1:
            raise ValueError("FlowMorph endpoint tensors must use one dtype")
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (self.z, self.delta, self.u)
        ):
            raise ValueError("FlowMorph endpoint tensors must contain only finite values")
        _scalar_like(self.sigma_i, self.z, "sigma_i")
        _scalar_like(self.sigma_last, self.z, "sigma_last")

    @property
    def pred(self) -> Tensor:
        return self.z + self.delta

    @property
    def state(self) -> Tensor:
        return construct_flow_state(
            self.z,
            self.delta,
            self.u,
            self.sigma_i,
            self.sigma_last,
        )

    @property
    def delta_sigma(self) -> Tensor:
        return sigma_delta(self.sigma_i, self.sigma_last, like=self.z)

    def detached(
        self,
        *,
        device: str | torch.device | None = None,
        clone: bool = True,
    ) -> "FlowMorphEndpoint":
        """Return a graph-free copy suitable for checkpointing or rendering."""

        def prepare(tensor: Tensor) -> Tensor:
            value = tensor.detach()
            if device is not None:
                value = value.to(device)
            return value.clone() if clone else value

        return FlowMorphEndpoint(
            z=prepare(self.z),
            delta=prepare(self.delta),
            u=prepare(self.u),
            sigma_i=self.sigma_i,
            sigma_last=self.sigma_last,
            timestep_i=self.timestep_i,
        )

    def to(
        self,
        device: str | torch.device | None = None,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "FlowMorphEndpoint":
        """Move/cast all endpoint tensors while preserving schedule metadata."""

        def convert(tensor: Tensor) -> Tensor:
            return tensor.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )

        return FlowMorphEndpoint(
            z=convert(self.z),
            delta=convert(self.delta),
            u=convert(self.u),
            sigma_i=self.sigma_i,
            sigma_last=self.sigma_last,
            timestep_i=self.timestep_i,
        )

    def tensor_dict(self) -> dict[str, Tensor]:
        """Return the transparent ``z``/``delta``/``u`` checkpoint payload."""

        return {"z": self.z, "delta": self.delta, "u": self.u}
