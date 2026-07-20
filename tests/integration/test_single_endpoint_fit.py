from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from flowmorph_klein.checkpoints import load_endpoint_checkpoint, save_endpoint_checkpoint
from flowmorph_klein.endpoint_optimizer import EndpointOptimizerConfig, optimize_endpoint


class _FrozenAffineVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.8), requires_grad=False)

    def predict_velocity(self, state, timestep, conditioning):
        del timestep
        return state * self.gain + conditioning["bias"]


def test_offline_complete_endpoint_fit_reduces_loss_and_checkpoints(tmp_path) -> None:
    predictor = _FrozenAffineVelocity()
    z = torch.tensor([[[0.8, -0.6, 0.3, -0.2]]], dtype=torch.float32)
    saved_steps: list[int] = []

    def checkpoint(step, endpoint, optimizer, diagnostics) -> None:
        saved_steps.append(step)
        save_endpoint_checkpoint(
            tmp_path / "source",
            endpoint.tensor_dict(),
            {"completed_steps": step, "prompt_checksum": "offline-source"},
            optimizer=optimizer,
        )
        assert diagnostics.step == step

    result = optimize_endpoint(
        z,
        sigma_i=0.8,
        sigma_last=0.0,
        timestep_i=torch.tensor(800.0),
        predictor=predictor,
        conditioning={"bias": torch.full_like(z, 0.1)},
        config=EndpointOptimizerConfig(
            optimization_steps=12,
            checkpoint_every=4,
            weight_decay=0.01,
        ),
        checkpoint_callback=checkpoint,
    )

    assert result.completed_steps == 12
    assert saved_steps == [4, 8, 12]
    assert result.diagnostics[-1].total_loss < result.diagnostics[0].total_loss
    assert result.endpoint.delta.abs().sum() > 0
    assert result.endpoint.u.abs().sum() > 0
    assert predictor.gain.grad is None

    loaded = load_endpoint_checkpoint(tmp_path / "source")
    assert loaded.metadata["completed_steps"] == 12
    torch.testing.assert_close(loaded.tensors["delta"], result.endpoint.delta)
    torch.testing.assert_close(loaded.tensors["u"], result.endpoint.u)
    assert loaded.metadata["optimizer_state_saved"] is True


@pytest.mark.integration
@pytest.mark.flux2_9b
@pytest.mark.requires_gated_access
@pytest.mark.full_endpoint
def test_real_complete_source_endpoint_fit(integration_harness) -> None:
    integration_harness.require("FLOWMORPH_RUN_FULL_ENDPOINT_INTEGRATION")

    from flowmorph_klein.pipeline import FlowMorphRunner

    # Smoke mode changes only frame count and permits a one-step target after
    # the required complete 100-step source fit; it does not change 512px,
    # scheduler, model, CFG, or source-fit semantics.
    config = integration_harness.config(
        run_mode="smoke",
        frame_count=3,
        source_steps=100,
        target_steps=1,
    )
    runner = FlowMorphRunner.from_config(config)
    runner.prepare()
    runner.run()

    source_checkpoint = load_endpoint_checkpoint(runner.run_directory / "checkpoints" / "source")
    manifest = json.loads((runner.run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_completed_steps"] == 100
    assert source_checkpoint.metadata["completed_steps"] == 100
    assert source_checkpoint.tensors["z"].shape == (1, 1024, 128)
