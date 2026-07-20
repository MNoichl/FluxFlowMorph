import math

import pytest
import torch
from torch import nn

from flowmorph_klein.endpoint_optimizer import (
    EndpointOptimizerConfig,
    optimize_endpoint,
)


class FrozenCastPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.saw_differentiable_input = []

    def predict_velocity(self, state, timestep, conditioning):
        self.saw_differentiable_input.append(state.requires_grad)
        compute_state = state.to(torch.bfloat16)
        bias = conditioning["bias"].to(device=state.device, dtype=torch.bfloat16)
        return compute_state * self.gain.to(torch.bfloat16) + bias


def test_only_float32_pred_and_u_receive_gradients() -> None:
    predictor = FrozenCastPredictor()
    z = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.bfloat16)
    conditioning = {"bias": torch.zeros_like(z)}
    callback_steps = []
    callback_groups = []

    def checkpoint_callback(step, endpoint, optimizer, diagnostics):
        callback_steps.append(step)
        callback_groups.append(
            [
                (group["group_name"], group["lr"], group["weight_decay"])
                for group in optimizer.param_groups
            ]
        )
        assert endpoint.z.dtype is torch.float32
        assert not endpoint.delta.requires_grad
        assert diagnostics.step == step

    result = optimize_endpoint(
        z,
        sigma_i=0.8,
        sigma_last=0.0,
        timestep_i=torch.tensor([800.0]),
        predictor=predictor,
        conditioning=conditioning,
        config=EndpointOptimizerConfig(
            optimization_steps=3,
            checkpoint_every=2,
        ),
        checkpoint_callback=checkpoint_callback,
    )

    assert predictor.saw_differentiable_input == [True, True, True]
    assert predictor.gain.grad is None
    assert result.endpoint.z.dtype is torch.float32
    assert result.endpoint.delta.dtype is torch.float32
    assert result.endpoint.u.dtype is torch.float32
    assert not result.endpoint.z.requires_grad
    assert callback_steps == [2, 3]
    assert callback_groups == [
        [("pred", 0.04, 0.01), ("u", 0.01, 0.01)],
        [("pred", 0.04, 0.01), ("u", 0.01, 0.01)],
    ]

    assert len(result.diagnostics) == 3
    for diagnostics in result.diagnostics:
        assert math.isfinite(diagnostics.total_loss)
        assert diagnostics.pred_gradient_norm > 0.0
        assert diagnostics.u_gradient_norm > 0.0
        assert diagnostics.peak_allocated_vram_bytes is None
        assert diagnostics.peak_reserved_vram_bytes is None


def test_optimizer_rejects_trainable_model_parameters() -> None:
    predictor = FrozenCastPredictor()
    predictor.gain.requires_grad_(True)
    with pytest.raises(ValueError, match="must be frozen"):
        optimize_endpoint(
            torch.ones(1, 2),
            sigma_i=0.5,
            sigma_last=0.0,
            timestep_i=500.0,
            predictor=predictor,
            conditioning={"bias": torch.zeros(1, 2)},
            config=EndpointOptimizerConfig(optimization_steps=1),
        )


def test_optimizer_rejects_velocity_detached_from_state() -> None:
    class DetachedPredictor(nn.Module):
        def predict_velocity(self, state, timestep, conditioning):
            del timestep, conditioning
            return state.detach()

    with pytest.raises(RuntimeError, match="detached from the optimizable state"):
        optimize_endpoint(
            torch.ones(1, 2),
            sigma_i=0.5,
            sigma_last=0.0,
            timestep_i=500.0,
            predictor=DetachedPredictor(),
            conditioning={},
            config=EndpointOptimizerConfig(optimization_steps=1),
        )


def test_optimizer_rejects_grad_enabled_cached_conditioning() -> None:
    predictor = FrozenCastPredictor()
    with pytest.raises(ValueError, match="conditioning"):
        optimize_endpoint(
            torch.ones(1, 2),
            sigma_i=0.5,
            sigma_last=0.0,
            timestep_i=500.0,
            predictor=predictor,
            conditioning={"bias": torch.zeros(1, 2, requires_grad=True)},
            config=EndpointOptimizerConfig(optimization_steps=1),
        )
