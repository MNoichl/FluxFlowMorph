from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from flowmorph_klein.conditioning import ConditioningPackage
from flowmorph_klein.flow_state import FlowMorphEndpoint
from flowmorph_klein.sequence import FlowMorphSequenceSession
from flowmorph_klein.spline_trajectory import (
    PeriodicConditioningSpline,
    PeriodicCubicBSplineBasis,
    PeriodicFlowMorphSpline,
    PeriodicSplineFlowMorphRenderer,
    allocate_periodic_segment_frames,
    periodic_thumbnail_distances,
    regularized_periodic_timing,
    sample_periodic_timeline,
)


def endpoint(value: float, direction: tuple[float, float]) -> FlowMorphEndpoint:
    z = torch.full((1, 2, 2), value, dtype=torch.float32)
    delta = torch.full((1, 2, 2), value * 0.1, dtype=torch.float32)
    u = torch.tensor(direction, dtype=torch.float32).repeat(2).reshape(1, 2, 2)
    return FlowMorphEndpoint(
        z=z,
        delta=delta,
        u=u,
        sigma_i=0.5,
        sigma_last=0.0,
        timestep_i=1.0,
    )


def conditioning(value: float) -> ConditioningPackage:
    return ConditioningPackage(
        prompt=f"prompt {value}",
        prompt_embeds=torch.full((1, 3, 2), value, dtype=torch.float32),
        text_ids=torch.zeros((1, 3, 4), dtype=torch.long),
    )


def test_regularized_timing_is_tempered_and_ratio_capped() -> None:
    timing = regularized_periodic_timing(
        (1.0, 100.0, 1.0, 0.01),
        distance_strength=0.6,
        distance_exponent=0.5,
        maximum_segment_ratio=1.5,
    )

    durations = np.asarray(timing.segment_durations)
    assert durations.sum() == pytest.approx(1.0)
    assert durations.max() / durations.min() <= 1.5 + 1e-12
    assert timing.knot_times[0] == 0.0
    assert timing.knot_times[-1] == 1.0
    assert np.all(np.diff(timing.knot_times) > 0)


def test_zero_timing_strength_is_uniform() -> None:
    timing = regularized_periodic_timing(
        (1.0, 9.0, 2.0, 4.0),
        distance_strength=0.0,
    )
    assert timing.segment_durations == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_frame_allocation_and_sampling_keep_every_anchor_without_duplicate() -> None:
    timing = regularized_periodic_timing((1.0, 2.0, 3.0, 4.0))
    counts = allocate_periodic_segment_frames(
        timing.segment_durations,
        total_frames=24,
        minimum_frames_per_segment=3,
    )
    samples = sample_periodic_timeline(timing, counts)

    assert sum(counts) == 24
    assert len(samples) == 24
    assert [item.anchor_index for item in samples if item.anchor_index is not None] == [
        0,
        1,
        2,
        3,
    ]
    assert len({item.time for item in samples}) == len(samples)
    assert all(0.0 <= item.time < 1.0 for item in samples)


def test_periodic_basis_interpolates_knots_and_is_c2_at_seam() -> None:
    timing = regularized_periodic_timing((1.0, 2.0, 1.5, 0.8, 1.2))
    basis = PeriodicCubicBSplineBasis(timing.knot_times)

    knot_weights = basis.weights(timing.knot_times[:-1])
    assert knot_weights == pytest.approx(np.eye(5), abs=1e-10)
    for derivative in (0, 1, 2):
        seam = basis.weights((0.0, 1.0), derivative=derivative)
        assert seam[0] == pytest.approx(seam[1], abs=1e-10)


def test_flowmorph_spline_hits_every_fitted_endpoint_state() -> None:
    timing = regularized_periodic_timing((1.0, 1.5, 0.9, 1.2))
    basis = PeriodicCubicBSplineBasis(timing.knot_times)
    endpoints = (
        endpoint(0.0, (1.0, 0.1)),
        endpoint(1.0, (1.0, 0.2)),
        endpoint(2.0, (0.9, 0.3)),
        endpoint(3.0, (0.8, 0.4)),
    )
    spline = PeriodicFlowMorphSpline(endpoints, basis)

    states = spline.state_batch(
        timing.knot_times[:-1],
        device="cpu",
    )
    expected = torch.cat([item.state for item in endpoints], dim=0)
    assert torch.allclose(states, expected, atol=2e-5, rtol=2e-5)
    seam = spline.state_batch((0.0, 1.0), device="cpu")
    assert torch.allclose(seam[0], seam[1], atol=1e-6, rtol=1e-6)


def test_flowmorph_spline_handles_degenerate_zero_u() -> None:
    timing = regularized_periodic_timing((1.0, 1.0, 1.0, 1.0))
    basis = PeriodicCubicBSplineBasis(timing.knot_times)
    endpoints = []
    for value in range(4):
        tensor = torch.full((1, 2, 2), float(value))
        endpoints.append(
            FlowMorphEndpoint(
                z=tensor,
                delta=torch.zeros_like(tensor),
                u=torch.zeros_like(tensor),
                sigma_i=0.5,
                sigma_last=0.0,
            )
        )
    spline = PeriodicFlowMorphSpline(endpoints, basis)
    assert torch.isfinite(spline.state_batch((0.125, 0.375), device="cpu")).all()


def test_conditioning_spline_hits_anchor_embeddings() -> None:
    timing = regularized_periodic_timing((1.0, 1.0, 1.0, 1.0))
    basis = PeriodicCubicBSplineBasis(timing.knot_times)
    packages = tuple(conditioning(float(value)) for value in range(4))
    spline = PeriodicConditioningSpline(packages, basis)

    result = spline.evaluate_batch(timing.knot_times[:-1], device="cpu")
    expected = torch.cat([item.prompt_embeds for item in packages], dim=0)
    assert torch.allclose(result.prompt_embeds, expected, atol=1e-6)
    assert result.batch_size == 4


def test_conditioning_spline_rejects_different_position_ids() -> None:
    timing = regularized_periodic_timing((1.0, 1.0, 1.0, 1.0))
    basis = PeriodicCubicBSplineBasis(timing.knot_times)
    packages = [conditioning(float(value)) for value in range(4)]
    packages[-1] = ConditioningPackage(
        packages[-1].prompt,
        packages[-1].prompt_embeds,
        torch.ones_like(packages[-1].text_ids),
    )
    with pytest.raises(ValueError, match="position IDs"):
        PeriodicConditioningSpline(packages, basis)


def test_thumbnail_distances_include_wrap_pair(tmp_path) -> None:
    paths = []
    for index, color in enumerate(((0, 0, 0), (64, 0, 0), (128, 0, 0), (255, 0, 0))):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (32, 32), color).save(path)
        paths.append(path)
    distances = periodic_thumbnail_distances(paths, analysis_size=32)
    assert len(distances) == 4
    assert distances[-1] > distances[0]


class FakeRunner:
    def __init__(self) -> None:
        self._prepared = True
        self.device = torch.device("cpu")
        self.schedule = object()
        self.config = SimpleNamespace(
            lora=SimpleNamespace(render_scale=1.2),
            flowmorph=SimpleNamespace(render_indices=(1, 2)),
        )
        self.scales = []

    def _require_prepared_values(self):
        return (object(), object(), object(), object(), object(), object(), self.schedule)

    def _set_lora_scale(self, value):
        self.scales.append(value)

    def _bound_predictor(self):
        return SimpleNamespace(cfg_execution="batched")


def test_alternative_renderer_batches_spline_states(monkeypatch) -> None:
    runner = FakeRunner()
    session = FlowMorphSequenceSession(runner, render_batch_size=2)
    timing = regularized_periodic_timing((1.0, 1.0, 1.0, 1.0))
    basis = PeriodicCubicBSplineBasis(timing.knot_times)
    state_spline = PeriodicFlowMorphSpline(
        (
            endpoint(0.0, (1.0, 0.1)),
            endpoint(1.0, (1.0, 0.2)),
            endpoint(2.0, (0.9, 0.3)),
            endpoint(3.0, (0.8, 0.4)),
        ),
        basis,
    )
    prompt_spline = PeriodicConditioningSpline(
        tuple(conditioning(float(value)) for value in range(4)),
        basis,
    )
    observed_batches = []
    monkeypatch.setattr(
        "flowmorph_klein.spline_trajectory.get_render_chain",
        lambda schedule, indices: ("chain", schedule, indices),
    )

    def fake_render(initial_state, **kwargs):
        observed_batches.append(
            (
                initial_state.shape[0],
                kwargs["conditioning"].batch_size,
                kwargs["frame_index"],
            )
        )
        return initial_state + 5.0

    monkeypatch.setattr(
        "flowmorph_klein.spline_trajectory.render_latent_trajectory",
        fake_render,
    )
    renderer = PeriodicSplineFlowMorphRenderer(
        session,
        state_spline,
        prompt_spline,
    )
    frames = renderer.render((0.0, 0.125, 0.25, 0.375, 0.5))

    assert observed_batches == [(2, 2, 0), (2, 2, 2), (1, 1, 4)]
    assert len(frames) == 5
    assert runner.scales == [1.2]
    assert all(frame.conditioning_mode == "prompt_schedule" for frame in frames)
    assert session.last_render_batch_size == 2
