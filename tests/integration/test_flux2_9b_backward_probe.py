from __future__ import annotations

import pytest
import torch
from torch import nn

from flowmorph_klein.diagnostics import run_backward_probe
from flowmorph_klein.flow_schedule import euler_flow_update


class _FrozenProductionShapeField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.125), requires_grad=False)

    def forward(self, state: torch.Tensor, _timestep: torch.Tensor) -> torch.Tensor:
        return state * self.gain + 0.03125


def test_offline_production_latent_shape_backward_contract() -> None:
    """Exercise the full 512px packed shape without claiming a 9B probe."""

    model = _FrozenProductionShapeField()
    z = torch.linspace(-1.0, 1.0, 1024 * 128, dtype=torch.float32).reshape(1, 1024, 128)
    report = run_backward_probe(
        z=z,
        sigma_i=0.75,
        sigma_last=0.0,
        timestep=torch.tensor(750.0),
        predict_velocity=model,
        frozen_parameters=model.parameters(),
    )
    assert report.passed
    assert report.latent_shape == (1, 1024, 128)
    assert report.velocity_input_gradient_norm > 0.0
    assert report.pred_gradient_norm > 0.0
    assert report.u_gradient_norm > 0.0
    assert model.gain.grad is None


@pytest.mark.integration
@pytest.mark.flux2_9b
@pytest.mark.requires_gated_access
@pytest.mark.production_backward
def test_real_production_backward_and_scheduler_sign_parity(integration_harness) -> None:
    integration_harness.require("FLOWMORPH_RUN_PRODUCTION_BACKWARD_INTEGRATION")

    from flowmorph_klein.pipeline import FlowMorphRunner

    config = integration_harness.config(
        run_mode="smoke",
        frame_count=3,
        source_steps=1,
        target_steps=1,
    )
    runner = FlowMorphRunner.from_config(config)
    runner.prepare()
    report = runner.run_production_backward_probe()
    passed = report["passed"] if isinstance(report, dict) else report.passed
    latent_shape = report["latent_shape"] if isinstance(report, dict) else report.latent_shape
    assert passed is True
    assert tuple(latent_shape) == (1, 1024, 128)
    velocity_input_gradient_norm = (
        report["velocity_input_gradient_norm"]
        if isinstance(report, dict)
        else report.velocity_input_gradient_norm
    )
    assert velocity_input_gradient_norm > 0.0

    # Same real latent and model output, compared against one stock scheduler
    # step.  This is the sign-convention integration required before fitting.
    transformer_device = next(runner.model.parameters()).device
    state = runner.source_latent.detach().to(transformer_device)
    ids = runner.image_ids.to(transformer_device)
    start_index = config.flowmorph.start_timestep_index
    timestep = runner.schedule.timesteps[start_index].to(transformer_device)
    conditional = runner.conditioning_cache.source.to(transformer_device)
    unconditional = runner.conditioning_cache.unconditional.to(transformer_device)
    with torch.no_grad():
        velocity = runner.model.predict_cfg_velocity(
            state,
            timestep,
            conditional,
            unconditional,
            ids,
            guidance_scale=config.guidance.scale,
            cfg_enabled=config.guidance.enabled,
            cfg_execution=config.guidance.execution.value,
        )
        runner.pipeline.scheduler.set_begin_index(start_index)
        stock_next = runner.pipeline.scheduler.step(
            velocity,
            timestep,
            state,
            return_dict=False,
        )[0]
        custom_next = euler_flow_update(
            state,
            velocity,
            runner.schedule.sigmas[start_index],
            runner.schedule.sigmas[start_index + 1],
        )
    torch.testing.assert_close(custom_next, stock_next, atol=2e-2, rtol=2e-2)
