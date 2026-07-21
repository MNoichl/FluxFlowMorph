from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flowmorph_klein.config import (
    BASE_MODEL_ID,
    FP8_MODEL_ID,
    MIRROR_MODEL_ID,
    MIRROR_MODEL_REVISION,
    ProjectTemplateConfig,
    canonical_config_hash,
    cli_namespace_overrides,
    load_config,
    parse_cli_overrides,
    resolve_config,
)
from flowmorph_klein.errors import ConfigurationError
from flowmorph_klein.types import HardwareProfile, RenderConditioningMode, RunMode


def test_default_template_is_exact_reference_contract_and_allows_unset_inputs() -> None:
    config = ProjectTemplateConfig()

    assert config.run_mode is RunMode.REFERENCE
    assert config.model.id == BASE_MODEL_ID
    assert config.model.profile is HardwareProfile.AUTO
    assert config.input.source_image is None
    assert config.input.target_image is None
    assert (config.input.width, config.input.height) == (512, 512)
    assert config.flowmorph.scheduler_points == 100
    assert config.flowmorph.start_timestep_index == 35
    assert config.flowmorph.optimization_steps_source == 100
    assert config.flowmorph.optimization_steps_target == 100
    assert config.flowmorph.render_indices == (35, 55, 75, 95)
    assert config.flowmorph.frame_count == 20
    assert config.guidance.enabled is True
    assert config.guidance.scale == 4.0
    assert config.memory.allow_degraded_run is False


@pytest.mark.parametrize(
    "model_id",
    [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-9B",
    ],
)
def test_forbidden_model_substitutions_fail(model_id: str) -> None:
    with pytest.raises(ValidationError, match="not supported"):
        ProjectTemplateConfig.model_validate({"model": {"id": model_id}})


def test_runware_mirror_requires_explicit_art_mode_and_pinned_revision(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="requires run_mode 'experimental'"):
        ProjectTemplateConfig.model_validate({"model": {"id": MIRROR_MODEL_ID}})

    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    template = ProjectTemplateConfig.model_validate({"run_mode": "experimental", "model": {"id": MIRROR_MODEL_ID}})
    resolved = resolve_config(
        template,
        selected_profile=HardwareProfile.A100_80GB_FULL,
        source_image=source,
        target_image=target,
    )

    assert resolved.model.id == MIRROR_MODEL_ID
    assert resolved.model.revision == MIRROR_MODEL_REVISION


def test_reference_profile_cannot_silently_reduce_semantics() -> None:
    with pytest.raises(ValidationError, match="cannot silently change semantics"):
        ProjectTemplateConfig.model_validate({"flowmorph": {"optimization_steps_source": 10}})

    with pytest.raises(ValidationError, match="cannot silently change semantics"):
        ProjectTemplateConfig.model_validate({"input": {"width": 256}})

    with pytest.raises(ValidationError, match="reference mode contract violation"):
        ProjectTemplateConfig.model_validate({"guidance": {"scale": 1.0}})


def test_smoke_mode_is_explicit_three_frame_base_9b_run() -> None:
    config = ProjectTemplateConfig.model_validate(
        {
            "run_mode": "smoke",
            "flowmorph": {
                "frame_count": 3,
                "optimization_steps_source": 2,
                "optimization_steps_target": 2,
            },
        }
    )
    assert config.run_mode is RunMode.SMOKE
    assert config.model.id == BASE_MODEL_ID
    assert config.flowmorph.frame_count == 3

    with pytest.raises(ValidationError, match="three frames"):
        ProjectTemplateConfig.model_validate({"run_mode": "smoke"})

    with pytest.raises(ValidationError, match="FP32 endpoint parameters"):
        ProjectTemplateConfig.model_validate(
            {
                "run_mode": "smoke",
                "model": {"optimization_parameter_dtype": "bfloat16"},
                "flowmorph": {"frame_count": 3},
            }
        )


def test_fp8_requires_explicit_matching_experimental_profile() -> None:
    with pytest.raises(ValidationError, match="requires profile"):
        ProjectTemplateConfig.model_validate(
            {
                "run_mode": "experimental",
                "model": {"id": FP8_MODEL_ID, "quantization": "fp8"},
            }
        )

    config = ProjectTemplateConfig.model_validate(
        {
            "run_mode": "experimental",
            "model": {
                "id": FP8_MODEL_ID,
                "profile": "fp8_9b_experimental",
                "quantization": "fp8",
            },
        }
    )
    assert config.model.profile is HardwareProfile.FP8_9B_EXPERIMENTAL


def test_quantization_label_must_match_the_exact_model_artifact() -> None:
    with pytest.raises(ValidationError, match="must be 'fp8'"):
        ProjectTemplateConfig.model_validate(
            {
                "run_mode": "experimental",
                "model": {
                    "id": FP8_MODEL_ID,
                    "profile": "fp8_9b_experimental",
                },
            }
        )
    with pytest.raises(ValidationError, match="must be 'none'"):
        ProjectTemplateConfig.model_validate({"model": {"id": BASE_MODEL_ID, "quantization": "fp8"}})


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("model", "optimization_parameter_dtype", "bfloat16"),
        ("memory", "run_production_backward_probe", False),
        ("memory", "save_intermediate_states", True),
    ),
)
def test_full_shape_profiles_reject_unimplemented_or_unsafe_runtime_changes(
    section: str,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="cannot silently change semantics"):
        ProjectTemplateConfig.model_validate(
            {
                "run_mode": "experimental",
                section: {field: value},
            }
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("model", "gradient_checkpointing", False),
        ("guidance", "execution", "batched"),
        ("memory", "text_encoder_offload", False),
        ("memory", "vae_offload", False),
    ),
)
def test_experimental_full_shape_allows_explicit_execution_controls(
    section: str,
    field: str,
    value: object,
) -> None:
    config = ProjectTemplateConfig.model_validate(
        {
            "run_mode": "experimental",
            section: {field: value},
        }
    )
    assert getattr(getattr(config, section), field) == value


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("model", "gradient_checkpointing", False),
        ("guidance", "execution", "batched"),
        ("memory", "text_encoder_offload", False),
        ("memory", "vae_offload", False),
    ),
)
def test_reference_mode_keeps_execution_controls_strict(
    section: str,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="reference mode contract violation"):
        ProjectTemplateConfig.model_validate({section: {field: value}})


def test_low_vram_diagnostic_still_requires_production_shape() -> None:
    with pytest.raises(ValidationError, match="cannot silently change semantics"):
        ProjectTemplateConfig.model_validate(
            {
                "run_mode": "diagnostic",
                "model": {"profile": "unsupported_low_vram"},
                "input": {"width": 256},
            }
        )


def test_invalid_render_chain_is_rejected() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        ProjectTemplateConfig.model_validate({"flowmorph": {"render_indices": [35, 75, 55, 95]}})
    with pytest.raises(ValidationError, match="smaller than"):
        ProjectTemplateConfig.model_validate({"flowmorph": {"render_indices": [35, 55, 75, 100]}})


def test_resolved_config_requires_paths_and_concrete_profile(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    template = ProjectTemplateConfig()

    with pytest.raises(ConfigurationError, match="profile='auto'"):
        resolve_config(template, source_image=source, target_image=target)

    resolved = resolve_config(
        template,
        selected_profile=HardwareProfile.A100_80GB_FULL,
        source_image=source,
        target_image=target,
    )
    assert resolved.input.source_image == source
    assert resolved.input.target_image == target


def test_resolved_config_validates_inputs_before_model_use(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="must exist before model download"):
        resolve_config(
            ProjectTemplateConfig(),
            selected_profile=HardwareProfile.A100_80GB_FULL,
            source_image=tmp_path / "missing-source.png",
            target_image=tmp_path / "missing-target.png",
        )


def test_yaml_loading_cli_overrides_and_aliases(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run_mode": "smoke",
                "flowmorph": {"frame_count": 3},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        path,
        overrides=[
            "--source=/content/source.png",
            "--target=/content/target.png",
            "--lora-scale=0.8",
            "reproducibility.seed=7",
        ],
    )
    assert config.input.source_image == Path("/content/source.png")
    assert config.input.target_image == Path("/content/target.png")
    assert config.lora.fit_scale == pytest.approx(0.8)
    assert config.lora.render_scale == pytest.approx(0.8)
    assert config.reproducibility.seed == 7

    parsed = parse_cli_overrides(["input.negative_prompt="])
    assert parsed["input"]["negative_prompt"] == ""
    documented = parse_cli_overrides(["--source", "/content/a.png", "--lora-scale", "0.7"])
    assert documented["input"]["source_image"] == "/content/a.png"
    assert documented["lora"]["fit_scale"] == pytest.approx(0.7)
    assert documented["lora"]["render_scale"] == pytest.approx(0.7)

    namespace = cli_namespace_overrides({"target_image": "/content/b.png", "lora_source": None})
    assert namespace == {"input": {"target_image": "/content/b.png"}}


def test_unknown_configuration_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProjectTemplateConfig.model_validate({"model": {"silent_4b_fallback": True}})


@pytest.mark.parametrize(
    "source",
    (
        "org/repo?token=secret-value",
        "https://huggingface.co/org/repo?access_token=secret-value",
        "hf_abcdefghijklmnop",
    ),
)
def test_lora_source_rejects_embedded_credentials(source: str) -> None:
    with pytest.raises(ValidationError, match="must not embed credentials"):
        ProjectTemplateConfig.model_validate({"lora": {"source": source}})


def test_configured_lora_requires_nonzero_scales() -> None:
    with pytest.raises(ValidationError, match="positive fit and render scales"):
        ProjectTemplateConfig.model_validate({"lora": {"source": "org/repo", "fit_scale": 0.0}})


@pytest.mark.parametrize(
    "field_name",
    (
        "save_raw_frames",
        "save_display_frames",
        "save_endpoint_states",
        "save_loss_history",
        "create_zip",
    ),
)
def test_mandatory_output_artifacts_cannot_be_disabled(field_name: str) -> None:
    with pytest.raises(ValidationError, match=rf"output\.{field_name} must be true"):
        ProjectTemplateConfig.model_validate({"output": {field_name: False}})


@pytest.mark.parametrize("field_name", ("record_environment", "record_checksums"))
def test_mandatory_provenance_records_cannot_be_disabled(field_name: str) -> None:
    with pytest.raises(
        ValidationError,
        match=rf"reproducibility\.{field_name} must be true",
    ):
        ProjectTemplateConfig.model_validate({"reproducibility": {field_name: False}})


@pytest.mark.parametrize(
    "config_data",
    (
        {},
        {"run_mode": "smoke", "flowmorph": {"frame_count": 3}},
        {
            "run_mode": "diagnostic",
            "model": {"profile": "unsupported_low_vram"},
        },
    ),
)
def test_interpolated_render_conditioning_requires_experimental_mode(
    config_data: dict[str, object],
) -> None:
    flowmorph = dict(config_data.get("flowmorph", {}))
    flowmorph["render_conditioning_mode"] = "interpolated_embeddings"
    data = {**config_data, "flowmorph": flowmorph}

    with pytest.raises(ValidationError, match="requires run_mode 'experimental'"):
        ProjectTemplateConfig.model_validate(data)


def test_interpolated_render_conditioning_is_allowed_in_experimental_mode() -> None:
    config = ProjectTemplateConfig.model_validate(
        {
            "run_mode": "experimental",
            "flowmorph": {"render_conditioning_mode": "interpolated_embeddings"},
        }
    )

    assert config.flowmorph.render_conditioning_mode is RenderConditioningMode.INTERPOLATED_EMBEDDINGS


def test_prompt_schedule_requires_experimental_mode_and_one_prompt_per_frame() -> None:
    prompts = [f"frame {index}" for index in range(20)]
    with pytest.raises(ValidationError, match="requires run_mode 'experimental'"):
        ProjectTemplateConfig.model_validate(
            {
                "input": {"bridge_prompts": prompts},
                "flowmorph": {"render_conditioning_mode": "prompt_schedule"},
            }
        )

    with pytest.raises(ValidationError, match="exactly flowmorph.frame_count"):
        ProjectTemplateConfig.model_validate(
            {
                "run_mode": "experimental",
                "input": {"bridge_prompts": prompts[:-1]},
                "flowmorph": {"render_conditioning_mode": "prompt_schedule"},
            }
        )

    config = ProjectTemplateConfig.model_validate(
        {
            "run_mode": "experimental",
            "input": {"bridge_prompts": prompts},
            "flowmorph": {"render_conditioning_mode": "prompt_schedule"},
        }
    )
    assert config.input.bridge_prompts == tuple(prompts)
    assert config.flowmorph.render_conditioning_mode is RenderConditioningMode.PROMPT_SCHEDULE


def test_config_hash_is_stable_and_sensitive() -> None:
    first = ProjectTemplateConfig()
    second = ProjectTemplateConfig()
    assert canonical_config_hash(first) == canonical_config_hash(second)

    smoke = ProjectTemplateConfig.model_validate({"run_mode": "smoke", "flowmorph": {"frame_count": 3}})
    assert canonical_config_hash(first) != canonical_config_hash(smoke)
