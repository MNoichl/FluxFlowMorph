import torch
import pytest

from flowmorph_klein.flow_schedule import (
    build_flowmorph_schedule,
    euler_flow_update,
    get_render_chain,
)
from flowmorph_klein.flow_state import FlowMorphEndpoint
from flowmorph_klein.renderer import render_latent_trajectory, render_morph, select_render_conditioning
from flowmorph_klein.types import RenderConditioningMode


class ConstantVelocityPredictor:
    def __init__(self, velocity: torch.Tensor) -> None:
        self.velocity = velocity
        self.timesteps = []

    def predict_velocity(self, state, timestep, conditioning):
        self.timesteps.append(float(timestep.item()))
        return self.velocity.to(device=state.device, dtype=state.dtype)


def test_euler_update_has_scheduler_sign_and_value() -> None:
    state = torch.tensor([1.0, 2.0])
    velocity = torch.tensor([0.5, -1.0])
    result = euler_flow_update(state, velocity, 0.8, 0.2)
    expected = state + (0.2 - 0.8) * velocity
    assert torch.allclose(result, expected)


def test_sparse_constant_field_matches_analytic_terminal_state() -> None:
    schedule = build_flowmorph_schedule(
        timesteps=[800.0, 550.0, 250.0, 50.0],
        sigmas=[0.8, 0.55, 0.25, 0.05, 0.0],
    )
    chain = get_render_chain(schedule, (0, 1, 3))
    initial = torch.tensor([[1.0, -2.0]])
    velocity = torch.tensor([[0.25, -0.5]])
    predictor = ConstantVelocityPredictor(velocity)

    result = render_latent_trajectory(
        initial,
        predictor=predictor,
        conditioning={"prompt": "ignored"},
        render_chain=chain,
    )
    expected = initial + (schedule.sigma_last - chain[0].current_sigma) * velocity
    assert torch.allclose(result, expected, atol=1e-7)
    assert predictor.timesteps == [800.0, 550.0, 50.0]


@pytest.mark.parametrize("non_finite", (float("nan"), float("inf")))
def test_renderer_rejects_non_finite_velocity_with_step_context(
    non_finite: float,
) -> None:
    schedule = build_flowmorph_schedule(
        timesteps=[800.0],
        sigmas=[0.8, 0.0],
    )
    chain = get_render_chain(schedule, (0,))
    predictor = ConstantVelocityPredictor(torch.tensor([non_finite]))

    with pytest.raises(
        FloatingPointError,
        match=r"predicted velocity.*frame 7.*scheduler index 0",
    ):
        render_latent_trajectory(
            torch.zeros(1),
            predictor=predictor,
            conditioning=None,
            render_chain=chain,
            frame_index=7,
        )


def test_renderer_rejects_non_finite_initial_state_before_model_call() -> None:
    schedule = build_flowmorph_schedule(
        timesteps=[800.0],
        sigmas=[0.8, 0.0],
    )
    chain = get_render_chain(schedule, (0,))
    predictor = ConstantVelocityPredictor(torch.zeros(1))

    with pytest.raises(FloatingPointError, match="initial render state.*frame 2"):
        render_latent_trajectory(
            torch.tensor([float("inf")]),
            predictor=predictor,
            conditioning=None,
            render_chain=chain,
            frame_index=2,
        )
    assert predictor.timesteps == []


def test_euler_update_returns_model_output_dtype() -> None:
    state = torch.ones(4, dtype=torch.float32)
    velocity = torch.ones(4, dtype=torch.bfloat16)
    result = euler_flow_update(state, velocity, 1.0, 0.0)
    assert result.dtype is torch.bfloat16


def test_renderer_conditioning_selection_defaults_to_released_source_behavior() -> None:
    source = object()
    target = object()
    bridge = object()
    assert (
        select_render_conditioning(
            RenderConditioningMode.SOURCE,
            0.9,
            source_conditioning=source,
            target_conditioning=target,
        )
        is source
    )
    assert (
        select_render_conditioning(
            "nearest_endpoint",
            0.5,
            source_conditioning=source,
            target_conditioning=target,
        )
        is target
    )
    assert (
        select_render_conditioning(
            "shared_bridge",
            0.5,
            source_conditioning=source,
            target_conditioning=target,
            bridge_conditioning=bridge,
        )
        is bridge
    )


def test_renderer_uses_one_scheduled_conditioning_per_frame() -> None:
    class RecordingPredictor(ConstantVelocityPredictor):
        def __init__(self) -> None:
            super().__init__(torch.zeros(1))
            self.conditionings = []

        def predict_velocity(self, state, timestep, conditioning):
            self.conditionings.append(conditioning)
            return super().predict_velocity(state, timestep, conditioning)

    endpoint = FlowMorphEndpoint(
        z=torch.zeros(1),
        delta=torch.zeros(1),
        u=torch.zeros(1),
        sigma_i=0.8,
        sigma_last=0.0,
    )
    schedule = build_flowmorph_schedule(timesteps=[800.0], sigmas=[0.8, 0.0])
    predictor = RecordingPredictor()
    scheduled = ("first", "middle", "last")

    frames = render_morph(
        endpoint,
        endpoint,
        schedule=schedule,
        predictor=predictor,
        source_conditioning="source",
        target_conditioning="target",
        alphas=(0.0, 0.5, 1.0),
        render_indices=(0,),
        conditioning_mode="prompt_schedule",
        frame_conditionings=scheduled,
    )

    assert predictor.conditionings == list(scheduled)
    assert len(frames) == 3

    with pytest.raises(ValueError, match="one package per rendered frame"):
        render_morph(
            endpoint,
            endpoint,
            schedule=schedule,
            predictor=predictor,
            source_conditioning="source",
            target_conditioning="target",
            alphas=(0.0, 1.0),
            render_indices=(0,),
            conditioning_mode="prompt_schedule",
            frame_conditionings=("only one",),
        )
