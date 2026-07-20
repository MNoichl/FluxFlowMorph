from __future__ import annotations

import pytest
import torch
from torch import nn

from flowmorph_klein.lora import compare_lora_velocities, verify_active_adapter


class _AdapterAwareTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_gain = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.lora_A = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.lora_B = nn.Parameter(torch.tensor(0.75), requires_grad=False)
        self.adapter_enabled = True

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        update = self.lora_A * self.lora_B if self.adapter_enabled else 0.0
        return state * (self.base_gain + update)


class _AdapterAwarePipeline:
    def __init__(self) -> None:
        self.transformer = _AdapterAwareTransformer()

    def get_active_adapters(self):
        return ["flowmorph_adapter"] if self.transformer.adapter_enabled else []

    def get_list_adapters(self):
        return {"transformer": ["flowmorph_adapter"]}


def test_offline_active_adapter_changes_velocity_and_preserves_input_gradient() -> None:
    pipeline = _AdapterAwarePipeline()
    state = torch.randn(1, 8, 4, requires_grad=True)

    pipeline.transformer.adapter_enabled = False
    baseline = pipeline.transformer(state)
    pipeline.transformer.adapter_enabled = True
    activation = verify_active_adapter(pipeline, "flowmorph_adapter")
    adapted = pipeline.transformer(state)
    numerical = compare_lora_velocities(baseline, adapted)

    assert activation.active
    assert numerical.changed
    assert numerical.maximum_absolute_difference > 0.0
    adapted.square().mean().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in pipeline.transformer.parameters())
    assert all(parameter.grad is None for parameter in pipeline.transformer.parameters())


@pytest.mark.integration
@pytest.mark.flux2_9b
@pytest.mark.requires_gated_access
@pytest.mark.requires_lora
@pytest.mark.production_backward
def test_real_lora_is_active_numerical_and_frozen(integration_harness) -> None:
    integration_harness.require("FLOWMORPH_RUN_LORA_INTEGRATION", require_lora=True)

    from flowmorph_klein.pipeline import FlowMorphRunner

    config = integration_harness.config(
        run_mode="smoke",
        frame_count=3,
        source_steps=1,
        target_steps=1,
        with_lora=True,
    )
    runner = FlowMorphRunner.from_config(config)
    runner.prepare()
    activation = verify_active_adapter(runner.pipeline, config.lora.adapter_name)
    assert activation.active

    device = next(runner.model.parameters()).device
    state = runner.source_latent.detach().to(device=device, dtype=torch.float32).requires_grad_(True)
    ids = runner.image_ids.to(device)
    timestep = runner.schedule.timesteps[config.flowmorph.start_timestep_index].to(device)
    conditional = runner.conditioning_cache.source.to(device)
    unconditional = runner.conditioning_cache.unconditional.to(device)

    runner.pipeline.disable_lora()
    with torch.no_grad():
        baseline = runner.model.predict_cfg_velocity(
            state.detach(),
            timestep,
            conditional,
            unconditional,
            ids,
            guidance_scale=config.guidance.scale,
            cfg_execution=config.guidance.execution.value,
        )
    runner.pipeline.enable_lora()
    adapted = runner.model.predict_cfg_velocity(
        state,
        timestep,
        conditional,
        unconditional,
        ids,
        guidance_scale=config.guidance.scale,
        cfg_execution=config.guidance.execution.value,
    )
    numerical = compare_lora_velocities(baseline, adapted)
    assert numerical.changed

    adapted.float().square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    adapter_parameters = [
        parameter
        for name, parameter in runner.pipeline.transformer.named_parameters()
        if "lora" in name.lower()
    ]
    assert adapter_parameters
    assert all(not parameter.requires_grad for parameter in runner.pipeline.transformer.parameters())
    assert all(parameter.grad is None for parameter in runner.pipeline.transformer.parameters())


@pytest.mark.integration
@pytest.mark.flux2_9b
@pytest.mark.requires_gated_access
@pytest.mark.requires_lora
@pytest.mark.production_backward
def test_real_gradient_checkpointing_ab_preserves_output_lora_and_input_gradients(
    integration_harness,
) -> None:
    """Mandatory real-model A/B; never represented by an offline substitute."""

    integration_harness.require(
        "FLOWMORPH_RUN_CHECKPOINTING_AB_INTEGRATION",
        require_lora=True,
    )

    from flowmorph_klein.lora import verify_active_adapter
    from flowmorph_klein.pipeline import FlowMorphRunner

    config = integration_harness.config(
        run_mode="smoke",
        frame_count=3,
        source_steps=1,
        target_steps=1,
        with_lora=True,
    )
    runner = FlowMorphRunner.from_config(config)
    runner.prepare()
    device = torch.device("cuda:0")
    timestep = runner.schedule.timesteps[config.flowmorph.start_timestep_index].to(
        device
    )
    conditional = runner.conditioning_cache.source.to(device)
    unconditional = runner.conditioning_cache.unconditional.to(device)
    image_ids = runner.image_ids.to(device)

    def execute(*, checkpointed: bool):
        if checkpointed:
            runner.model.enable_gradient_checkpointing()
        else:
            runner.model.disable_gradient_checkpointing()
        assert verify_active_adapter(
            runner.pipeline, config.lora.adapter_name, strict=True
        ).active
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        state = (
            runner.source_latent.detach()
            .to(device=device, dtype=torch.float32)
            .requires_grad_(True)
        )
        velocity = runner.model.predict_cfg_velocity(
            state,
            timestep,
            conditional,
            unconditional,
            image_ids,
            guidance_scale=config.guidance.scale,
            cfg_enabled=config.guidance.enabled,
            cfg_execution=config.guidance.execution.value,
        )
        velocity.float().square().mean().backward()
        torch.cuda.synchronize(device)
        assert state.grad is not None and torch.isfinite(state.grad).all()
        output = velocity.detach().float().cpu()
        input_gradient_norm = float(
            torch.linalg.vector_norm(state.grad.detach().float()).cpu()
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        del state, velocity
        return output, input_gradient_norm, peak

    plain_output, plain_gradient_norm, plain_peak = execute(checkpointed=False)
    checkpointed_output, checkpointed_gradient_norm, checkpointed_peak = execute(
        checkpointed=True
    )

    torch.testing.assert_close(
        checkpointed_output,
        plain_output,
        atol=2e-2,
        rtol=2e-2,
    )
    assert plain_gradient_norm > 0.0
    assert checkpointed_gradient_norm > 0.0
    assert checkpointed_peak < plain_peak
    assert all(
        parameter.grad is None
        for parameter in runner.pipeline.transformer.parameters()
    )
