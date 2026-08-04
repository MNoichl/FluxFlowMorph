from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from flowmorph_klein.chimera import (
    AdaptiveBatchSizer,
    ChimeraConfig,
    ChimeraEndpointCache,
    FluxFeatureController,
    LTM_CALIBRATION_VERSION,
    LTMCalibration,
    LTMPrototypeAccumulator,
    StoredFeature,
    append_anchor_conditioning,
    calibrate_flux_ltm,
    compute_glcs_from_similarities,
    conditioning_interpolation_report,
    estimate_safe_cuda_batch_size,
    flux_depth_ltm,
    invert_endpoint,
    interpolate_chimera_conditioning,
    ltm_mapping_report,
    map_denoising_to_inversion_step,
    match_ltm_prototypes,
    match_monotonic_ltm_prototypes,
    nearest_cached_step,
    prompt_anchor_reliability,
    radial_frequency_descriptor,
    render_chimera_morph,
    select_flux_feature_groups,
)
from flowmorph_klein.conditioning import ConditioningPackage
from flowmorph_klein.flow_schedule import FlowSchedule


class FakeDoubleBlock(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = float(offset)

    def forward(self, image: torch.Tensor):
        context = image.new_zeros(image.shape[0], 2, image.shape[-1])
        return context, image + self.offset


class FakeSingleBlock(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = float(offset)

    def forward(self, hidden: torch.Tensor):
        return hidden + self.offset


class FakeFluxTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [FakeDoubleBlock(index + 1) for index in range(3)]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [FakeSingleBlock(index + 4) for index in range(3)]
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=False,
    ):
        del timestep, img_ids, txt_ids, guidance, joint_attention_kwargs, return_dict
        image = hidden_states
        for block in self.transformer_blocks:
            _, image = block(image)
        text = image.new_zeros(
            image.shape[0], encoder_hidden_states.shape[1], image.shape[-1]
        )
        joint = torch.cat((text, image), dim=1)
        for block in self.single_transformer_blocks:
            joint = block(joint)
        return (joint[:, -image.shape[1] :, :] * 0.001,)


def conditioning(value: float, *, tokens: int = 3) -> ConditioningPackage:
    return ConditioningPackage(
        prompt=f"prompt-{value}",
        prompt_embeds=torch.full((1, tokens, 4), value),
        text_ids=torch.arange(tokens * 4).reshape(1, tokens, 4),
    )


def endpoint_cache(
    key: str,
    group: str,
    value: float,
    *,
    tokens: int = 4,
) -> ChimeraEndpointCache:
    stored = StoredFeature.from_tensor(
        torch.full((1, tokens, 3), value),
        "float32",
    )
    return ChimeraEndpointCache(
        key=key,
        inverted_latent=torch.zeros(1, tokens, 3),
        features={group: {0: stored}},
        inversion_steps=1,
        image_token_count=tokens,
        group_modules={group: "fake"},
    )


def test_chimera_config_exposes_paper_defaults_and_memory_controls() -> None:
    config = ChimeraConfig()
    assert config.inversion_steps == 50
    assert config.denoising_steps == 50
    assert config.aci_weight == pytest.approx(0.4)
    assert config.sap_active_ratio == pytest.approx(0.2)
    assert config.anchor_reliability_threshold == pytest.approx(0.45)
    assert config.ltm_mode == "fft"
    assert config.ltm_bands == 16
    assert config.auto_render_batch_size is True
    assert config.render_batch_max == 10
    assert config.cache_storage == "int8"
    assert config.cache_stride == 2
    assert config.conditioning_interpolation == "slerp"


def test_chimera_slerp_prevents_midpoint_conditioning_norm_collapse() -> None:
    text_ids = torch.zeros(1, 1, 4, dtype=torch.long)
    source = ConditioningPackage(
        "source",
        torch.tensor([[[1.0, 0.0]]]),
        text_ids,
    )
    target = ConditioningPackage(
        "target",
        torch.tensor([[[0.0, 2.0]]]),
        text_ids,
    )

    midpoint = interpolate_chimera_conditioning(source, target, 0.5)
    report = conditioning_interpolation_report(source, target, 0.5)

    assert torch.linalg.vector_norm(midpoint.prompt_embeds) == pytest.approx(1.5)
    assert report["linear_norm_retention"] == pytest.approx(math.sqrt(1.25) / 1.5)
    assert report["active_norm_retention"] == pytest.approx(1.0)
    assert report["mode"] == "slerp"


def test_cuda_batch_estimate_keeps_reserve_and_overhead_margin() -> None:
    gib = 1024**3
    estimate = estimate_safe_cuda_batch_size(
        current_batch_size=2,
        baseline_allocated_bytes=10 * gib,
        peak_allocated_bytes=14 * gib,
        free_before_bytes=20 * gib,
        total_bytes=40 * gib,
        maximum_batch_size=10,
        reserve_fraction=0.10,
        reserve_bytes=2 * gib,
        overhead_factor=1.25,
    )
    assert estimate == 6


def test_adaptive_batch_sizer_grows_then_binary_searches_after_oom() -> None:
    sizer = AdaptiveBatchSizer(initial_batch_size=2, maximum_batch_size=10)
    assert sizer.next_batch_size(10) == 2
    assert sizer.record_success(2, safe_ceiling_hint=10) == 10
    assert sizer.record_oom(10) == 6
    assert sizer.record_oom(6) == 4
    assert sizer.record_success(4, safe_ceiling_hint=10) == 5
    assert sizer.record_success(5, safe_ceiling_hint=10) == 5
    assert sizer.report() == {
        "candidate": 5,
        "largest_success": 5,
        "smallest_failure": 6,
        "maximum": 10,
    }


def test_flux_depth_groups_are_distinct_and_ordered() -> None:
    transformer = FakeFluxTransformer()
    groups = select_flux_feature_groups(transformer)

    assert [group.name for group in groups] == ["early", "middle", "late"]
    assert [group.combined_depth for group in groups] == sorted(
        group.combined_depth for group in groups
    )
    assert len({id(group.module) for group in groups}) == 3
    assert groups[-1].stream == "single"


def test_ltm_and_idm_cover_endpoints_and_stride_gaps() -> None:
    assert flux_depth_ltm(0, 9) == "early"
    assert flux_depth_ltm(4, 9) == "middle"
    assert flux_depth_ltm(8, 9) == "late"
    assert map_denoising_to_inversion_step(
        0, denoising_steps=5, inversion_steps=9
    ) == 0
    assert map_denoising_to_inversion_step(
        2, denoising_steps=5, inversion_steps=9
    ) == 4
    assert map_denoising_to_inversion_step(
        4, denoising_steps=5, inversion_steps=9
    ) == 8
    assert nearest_cached_step(5, [0, 4, 6, 8]) == 4


def test_fft_ltm_prototype_matching_and_roundtrip_are_deterministic() -> None:
    layers = {
        "early": [3.0, 0.0, 0.0],
        "middle": [0.0, 3.0, 0.0],
        "late": [0.0, 0.0, 3.0],
    }
    timesteps = ([2.5, 0.1, 0.0], [0.0, 0.2, 2.8], [0.1, 2.7, 0.0])
    mapping = match_ltm_prototypes(layers, timesteps, bands=3)
    assert mapping == ("early", "late", "middle")

    calibration = LTMCalibration(
        bands=3,
        sample_count=4,
        group_modules=("block.early", "block.middle", "block.late"),
        layer_prototypes=tuple(tuple(layers[group]) for group in ("early", "middle", "late")),
        timestep_prototypes=timesteps,
        mapping=mapping,
    )
    restored = LTMCalibration.from_dict(calibration.to_dict())
    assert calibration.version == 1
    assert "mapping_strategy" not in calibration.to_dict()
    assert restored == calibration
    assert restored.fingerprint == calibration.fingerprint


def test_monotonic_ltm_matching_selects_contiguous_coarse_to_fine_regions() -> None:
    layers = {
        "early": [1.0, 0.0, 0.0],
        "middle": [0.0, 1.0, 0.0],
        "late": [0.0, 0.0, 1.0],
    }
    timesteps = (
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.7, 0.3],
        [0.0, 0.2, 0.8],
        [0.0, 0.1, 0.9],
    )

    mapping = match_monotonic_ltm_prototypes(layers, timesteps, bands=3)
    report = ltm_mapping_report(mapping)

    assert mapping == ("early", "early", "middle", "middle", "late", "late")
    assert report["group_counts"] == {"early": 2, "middle": 2, "late": 2}
    assert report["monotonic_coarse_to_fine"] is True


def test_ltm_accumulator_builds_dataset_level_layer_and_timestep_means() -> None:
    accumulator = LTMPrototypeAccumulator(
        step_count=2,
        group_modules={"early": "e", "middle": "m", "late": "l"},
        bands=3,
    )
    for sample_offset in (0.0, 0.2):
        for step in range(2):
            for group_index, group in enumerate(("early", "middle", "late")):
                descriptor = torch.tensor(
                    [1.0 + group_index, 1.0 + step, 1.0 + sample_offset]
                )
                accumulator.add(group, step, descriptor)
            accumulator.add_timestep(
                step,
                torch.tensor([1.0 + step, 2.0 - step, 1.0 + sample_offset]),
            )
    calibration = accumulator.finalize()

    assert calibration.version == LTM_CALIBRATION_VERSION
    assert calibration.descriptor_normalized is True
    assert calibration.sample_count == 2
    assert calibration.step_count == 2
    assert calibration.group_module_map == {"early": "e", "middle": "m", "late": "l"}
    assert len(calibration.mapping) == 2


def test_ltm_accumulator_rejects_collapsed_fft_mapping_with_fixed_fallback() -> None:
    accumulator = LTMPrototypeAccumulator(
        step_count=6,
        group_modules={"early": "e", "middle": "m", "late": "l"},
        bands=3,
    )
    layer_descriptors = {
        "early": torch.tensor([1.0, 0.0, 0.0]),
        "middle": torch.tensor([0.0, 1.0, 0.0]),
        "late": torch.tensor([0.0, 0.0, 1.0]),
    }
    for step in range(6):
        for group, descriptor in layer_descriptors.items():
            accumulator.add(group, step, descriptor)
        accumulator.add_timestep(step, layer_descriptors["middle"])

    calibration = accumulator.finalize()

    assert calibration.mapping_strategy == "fixed_coarse_to_fine_fallback"
    assert calibration.mapping == (
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "late",
    )
    assert calibration.independent_mapping == ("middle",) * 6
    assert calibration.fallback_reason is not None
    assert calibration.mapping_report["groups_used"] == 3
    restored = LTMCalibration.from_dict(calibration.to_dict())
    assert restored == calibration
    assert restored.fingerprint == calibration.fingerprint


def test_ltm_accumulator_uses_monotonic_fft_mapping_when_spectra_are_healthy() -> None:
    accumulator = LTMPrototypeAccumulator(
        step_count=6,
        group_modules={"early": "e", "middle": "m", "late": "l"},
        bands=3,
    )
    descriptors = {
        "early": torch.tensor([1.0, 0.0, 0.0]),
        "middle": torch.tensor([0.0, 1.0, 0.0]),
        "late": torch.tensor([0.0, 0.0, 1.0]),
    }
    expected = ("early", "early", "middle", "middle", "late", "late")
    for step, active_group in enumerate(expected):
        for group, descriptor in descriptors.items():
            accumulator.add(group, step, descriptor)
        accumulator.add_timestep(step, descriptors[active_group])

    calibration = accumulator.finalize()

    assert calibration.mapping_strategy == "fft_monotonic"
    assert calibration.fallback_reason is None
    assert calibration.mapping == expected
    assert calibration.mapping_report["monotonic_coarse_to_fine"] is True


def test_sparse_cache_always_captures_calibrated_group_transitions() -> None:
    transformer = FakeFluxTransformer()
    schedule = FlowSchedule(
        timesteps=torch.tensor([1000.0, 666.0, 333.0]),
        sigmas=torch.tensor([1.0, 2 / 3, 1 / 3, 0.0]),
    )
    calibration = LTMCalibration(
        bands=3,
        sample_count=1,
        group_modules=(
            "transformer_blocks.1",
            "transformer_blocks.2",
            "single_transformer_blocks.1",
        ),
        layer_prototypes=((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)),
        timestep_prototypes=((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)),
        mapping=("early", "middle", "late"),
    )
    controller = FluxFeatureController(
        transformer,
        image_token_count=4,
        storage="float32",
    )
    with controller:
        cache = invert_endpoint(
            key="transitions",
            clean_latent=torch.zeros(1, 4, 3),
            schedule=schedule,
            transformer=transformer,
            conditioning=conditioning(1.0),
            image_ids=torch.zeros(1, 4, 4),
            controller=controller,
            ltm_calibration=calibration,
            cache_stride=99,
        )

    assert set(cache.features) == {"early", "middle", "late"}
    assert all(cache.features[group] for group in ("early", "middle", "late"))


def test_int8_feature_storage_roundtrips_with_bounded_error() -> None:
    tensor = torch.linspace(-3.0, 3.0, 97).reshape(1, 97, 1)
    stored = StoredFeature.from_tensor(tensor, "int8")
    restored = stored.materialize(device="cpu", dtype=torch.float32)

    assert stored.values.dtype is torch.int8
    assert torch.max(torch.abs(restored - tensor)).item() <= float(stored.scale) / 2 + 1e-6
    assert stored.storage_bytes < tensor.numel() * tensor.element_size()


def test_feature_controller_captures_and_injects_only_image_tokens() -> None:
    transformer = FakeFluxTransformer()
    image_tokens = 4
    controller = FluxFeatureController(
        transformer,
        image_token_count=image_tokens,
        storage="float32",
    )
    early = select_flux_feature_groups(transformer)[0]
    image = torch.zeros(1, image_tokens, 3)

    with controller:
        with controller.capture(key="captured", step=0, group="early"):
            _, captured_output = early.module(image)
        captured = controller.endpoint_cache(
            key="captured",
            inverted_latent=torch.zeros_like(image),
            inversion_steps=1,
        )
        stored = captured.feature("early", 0).materialize(
            device="cpu", dtype=torch.float32
        )
        assert torch.equal(stored, captured_output)

        source = endpoint_cache("source", "early", 2.0)
        target = endpoint_cache("target", "early", 4.0)
        batch = torch.zeros(2, image_tokens, 3)
        with controller.inject(
            source=source,
            target=target,
            inversion_step=0,
            group="early",
            alphas=torch.tensor([0.0, 1.0]),
            weight=0.5,
        ):
            _, injected = early.module(batch)

    baseline = early.module.offset
    assert torch.allclose(injected[0], torch.full_like(injected[0], baseline + 1.0))
    assert torch.allclose(injected[1], torch.full_like(injected[1], baseline + 2.0))


def test_sap_appends_anchor_tokens_and_reliability_uses_both_endpoints() -> None:
    base = conditioning(1.0, tokens=3)
    anchor = conditioning(1.0, tokens=2)
    combined = append_anchor_conditioning(base, anchor, max_anchor_tokens=1)

    assert combined.prompt_embeds.shape == (1, 4, 4)
    assert combined.text_ids.shape == (1, 4, 4)
    similarity_a, similarity_b, reliability = prompt_anchor_reliability(
        anchor,
        conditioning(2.0),
        conditioning(-1.0),
    )
    assert similarity_a == pytest.approx(1.0)
    assert similarity_b == pytest.approx(-1.0)
    assert reliability == pytest.approx(-1.0)


def test_radial_frequency_descriptor_is_normalized() -> None:
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    checkerboard = ((xx + yy) % 2).float().reshape(64, 1).repeat(1, 3)
    descriptor = radial_frequency_descriptor(checkerboard, bands=8)

    assert descriptor.shape == (8,)
    assert bool(torch.isfinite(descriptor).all())
    assert float(descriptor.sum()) == pytest.approx(1.0)
    assert descriptor[-1] > descriptor[0]

    chunked = radial_frequency_descriptor(checkerboard, bands=8, channel_chunk_size=1)
    raw = radial_frequency_descriptor(checkerboard, bands=8, normalize=False)
    assert torch.allclose(chunked, descriptor)
    assert float(raw.sum()) > 1.0


def test_glcs_returns_geometric_mean_of_global_and_local_terms() -> None:
    result = compute_glcs_from_similarities(
        [0.75, 0.50, 0.25],
        [0.25, 0.50, 0.75],
        endpoint_similarity_matrix=[[1.0, 0.0], [0.0, 1.0]],
        gamma=2.0,
    )

    assert result["gcs"] == pytest.approx(1.0)
    assert 0.0 < result["lcs"] <= 1.0
    assert result["glcs"] == pytest.approx(
        math.sqrt(result["gcs"] * result["lcs"])
    )


def test_tiny_end_to_end_inversion_aci_and_sap_path_is_finite() -> None:
    transformer = FakeFluxTransformer()
    schedule = FlowSchedule(
        timesteps=torch.tensor([1000.0, 500.0]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    image_ids = torch.zeros(1, 4, 4)
    controller = FluxFeatureController(
        transformer,
        image_token_count=4,
        storage="float32",
    )
    source_prompt = conditioning(1.0)
    target_prompt = conditioning(2.0)
    anchor_prompt = conditioning(1.5, tokens=2)
    calibration = calibrate_flux_ltm(
        endpoint_samples=(
            (torch.zeros(1, 4, 3), source_prompt),
            (torch.ones(1, 4, 3), target_prompt),
        ),
        schedule=schedule,
        transformer=transformer,
        image_ids=image_ids,
        bands=4,
        channel_chunk_size=2,
    )
    with controller:
        source = invert_endpoint(
            key="source",
            clean_latent=torch.zeros(1, 4, 3),
            schedule=schedule,
            transformer=transformer,
            conditioning=source_prompt,
                image_ids=image_ids,
                controller=controller,
                ltm_calibration=calibration,
            )
        target = invert_endpoint(
            key="target",
            clean_latent=torch.ones(1, 4, 3),
            schedule=schedule,
            transformer=transformer,
            conditioning=target_prompt,
                image_ids=image_ids,
                controller=controller,
                ltm_calibration=calibration,
            )

    diagnostics: list[dict[str, float | str | None]] = []
    frames = render_chimera_morph(
        source,
        target,
        schedule=schedule,
        transformer=transformer,
        image_ids=image_ids,
        source_conditioning=source_prompt,
        target_conditioning=target_prompt,
        anchor_conditioning=anchor_prompt,
        unconditional_conditioning=conditioning(0.0),
        alphas=[0.25, 0.75],
        config=ChimeraConfig(
            inversion_steps=2,
            denoising_steps=2,
            aci_weight=0.1,
            sap_active_ratio=0.5,
            anchor_max_tokens=2,
            cache_stride=1,
            cache_storage="float32",
            render_batch_size=2,
            decode_batch_size=2,
            guidance_scale=1.0,
            ltm_bands=4,
        ),
        ltm_calibration=calibration,
        diagnostics=diagnostics,
    )

    assert [frame.alpha for frame in frames] == [0.25, 0.75]
    assert all(frame.final_latent.shape == (1, 4, 3) for frame in frames)
    assert all(bool(torch.isfinite(frame.final_latent).all()) for frame in frames)
    assert not torch.equal(frames[0].final_latent, frames[1].final_latent)
    assert [row["alpha"] for row in diagnostics] == [0.25, 0.75]
    assert all(row["active_norm_retention"] == pytest.approx(1.0) for row in diagnostics)
    assert all(row["cfg_residual_rms_mean"] is None for row in diagnostics)
