from __future__ import annotations

import pytest
import torch
from torch import nn

from flowmorph_klein.endpoint_optimizer import EndpointOptimizerConfig, optimize_endpoint
from flowmorph_klein.flow_schedule import build_flowmorph_schedule
from flowmorph_klein.packaging import validate_archive
from flowmorph_klein.renderer import render_morph


class _RecordedConstantField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.labels: list[str] = []

    def predict_velocity(self, state, timestep, conditioning):
        del timestep
        self.labels.append(conditioning["label"])
        # Keep this fake explicitly differentiable with respect to state so it
        # exercises the same optimizer contract as the production transformer.
        return (
            torch.ones_like(state) * conditioning["velocity"]
            + state * 1e-6
            + self.anchor
        )


def test_offline_sequential_endpoint_fits_then_three_frame_morph() -> None:
    predictor = _RecordedConstantField()
    source_z = torch.tensor([[[0.0, 0.25, 0.5, 0.75]]])
    target_z = torch.tensor([[[1.0, 0.75, 0.5, 0.25]]])
    source_conditioning = {"label": "source", "velocity": 0.2}
    target_conditioning = {"label": "target", "velocity": -0.15}
    fit_config = EndpointOptimizerConfig(optimization_steps=4, checkpoint_every=4)

    source = optimize_endpoint(
        source_z,
        sigma_i=0.8,
        sigma_last=0.0,
        timestep_i=800.0,
        predictor=predictor,
        conditioning=source_conditioning,
        config=fit_config,
    ).endpoint
    target = optimize_endpoint(
        target_z,
        sigma_i=0.8,
        sigma_last=0.0,
        timestep_i=800.0,
        predictor=predictor,
        conditioning=target_conditioning,
        config=fit_config,
    ).endpoint
    assert predictor.labels == ["source"] * 4 + ["target"] * 4

    schedule = build_flowmorph_schedule(
        scheduler_points=5,
        timesteps=[800, 600, 400, 200, 100],
        sigmas=[0.8, 0.6, 0.4, 0.2, 0.1, 0.0],
    )
    frames = render_morph(
        source,
        target,
        schedule=schedule,
        predictor=predictor,
        source_conditioning=source_conditioning,
        target_conditioning=target_conditioning,
        frame_count=3,
        render_indices=(0, 2, 4),
        conditioning_mode="source",
    )
    assert [frame.alpha for frame in frames] == [0.0, 0.5, 1.0]
    assert len(frames) == 3
    torch.testing.assert_close(frames[0].start_state, source.state)
    torch.testing.assert_close(frames[-1].start_state, target.state)
    assert all(torch.isfinite(frame.final_latent).all() for frame in frames)
    assert predictor.labels[-9:] == ["source"] * 9


@pytest.mark.integration
@pytest.mark.flux2_9b
@pytest.mark.requires_gated_access
def test_real_three_frame_smoke_morph_and_archive(integration_harness) -> None:
    integration_harness.require("FLOWMORPH_RUN_THREE_FRAME_INTEGRATION")

    from flowmorph_klein.pipeline import FlowMorphRunner

    config = integration_harness.config(
        run_mode="smoke",
        frame_count=3,
        source_steps=1,
        target_steps=1,
    )
    runner = FlowMorphRunner.from_config(config)
    runner.prepare()
    runner.run()

    assert len(list((runner.run_directory / "raw_frames").glob("frame_*.png"))) == 3
    assert len(list((runner.run_directory / "display_frames").glob("frame_*.png"))) == 3
    assert (runner.run_directory / "checkpoints/source/tensors.safetensors").is_file()
    assert (runner.run_directory / "checkpoints/target/tensors.safetensors").is_file()
    assert (runner.run_directory / "metrics.json").is_file()
    archives = list((runner.run_directory / "artifacts").glob("*.flowmorph-klein.zip"))
    assert len(archives) == 1
    names = validate_archive(archives[0])
    assert "raw_frames/frame_000.png" in names
    assert "raw_frames/frame_002.png" in names
