"""Validated configuration loading and resolution.

Configuration deliberately has two stages.  A distributable template may
leave endpoint paths unset and may request the ``auto`` hardware profile.  A
``ResolvedRunConfig`` is the contract accepted by production orchestration:
its endpoint paths and hardware profile are concrete and the fixed reference
semantics have survived every overlay.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .errors import ConfigurationError
from .types import (
    AlphaSchedule,
    AttentionBackend,
    CFGExecution,
    ComputeDType,
    HardwareProfile,
    InterpolationMode,
    LossMode,
    OptimizerName,
    QuantizationMode,
    RenderConditioningMode,
    ResizeMode,
    RunMode,
)


BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
FP8_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9b-fp8"
ALLOWED_MODEL_IDS = frozenset({BASE_MODEL_ID, FP8_MODEL_ID})
REFERENCE_RENDER_INDICES = (35, 55, 75, 95)


class StrictConfigModel(BaseModel):
    """Base model used by every serialized configuration section."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ProjectConfig(StrictConfigModel):
    name: str = Field(default="flowmorph_klein_full", min_length=1, max_length=128)
    version: int = Field(default=1, ge=1)
    repository_root: Path = Path("/content/FlowMorphKlein9B")


class PathsConfig(StrictConfigModel):
    input_root: Path = Path("/content/flowmorph_klein_images/max_v1")
    work_root: Path = Path("/content/flowmorph_klein_work/max_v1")
    result_root: Path = Path(
        "/content/flowmorph_klein_results/max_v1/full_lora_reproduction_v1"
    )
    hf_cache: Path = Path("/content/hf_cache")
    drive_root: Path | None = Path("/content/drive/MyDrive/FlowMorphKlein9B")


class ModelConfig(StrictConfigModel):
    id: str = BASE_MODEL_ID
    revision: str | None = None
    profile: HardwareProfile = HardwareProfile.AUTO
    transformer_compute_dtype: ComputeDType = ComputeDType.BFLOAT16
    optimization_parameter_dtype: ComputeDType = ComputeDType.FLOAT32
    quantization: QuantizationMode = QuantizationMode.NONE
    gradient_checkpointing: bool = True
    use_tf32: bool = True
    attention_backend: AttentionBackend = AttentionBackend.SDPA

    @field_validator("id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if value not in ALLOWED_MODEL_IDS:
            raise ValueError(
                "model.id must be the FLUX.2 Klein Base 9B model or its explicit "
                "experimental Base-9B FP8 variant; FLUX.1, 4B, and distilled "
                "models are not supported"
            )
        return value

    @model_validator(mode="after")
    def validate_quantization_matches_model_artifact(self) -> "ModelConfig":
        expected = (
            QuantizationMode.FP8
            if self.id == FP8_MODEL_ID
            else QuantizationMode.NONE
        )
        if self.quantization is not expected:
            raise ValueError(
                f"model.quantization must be {expected.value!r} for model.id "
                f"{self.id!r}; configuration may not mislabel loaded weights"
            )
        return self


class LoraConfig(StrictConfigModel):
    source: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    weight_name: str | None = None
    adapter_name: str = Field(
        default="flowmorph_adapter", pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$"
    )
    fit_scale: float = Field(default=1.0, ge=0.0)
    render_scale: float = Field(default=1.0, ge=0.0)
    require_base_9b_compatibility: bool = True
    allow_distilled_9b: bool = False
    include_in_output_archive: bool = False

    @field_validator("fit_scale", "render_scale")
    @classmethod
    def validate_finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("LoRA scales must be finite")
        return value

    @field_validator("source")
    @classmethod
    def reject_embedded_credentials(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if re.search(r"hf_[A-Za-z0-9]{8,}", value) or re.search(
            r"(?i)[?&](?:token|access_token|auth|authorization)=[^&\s]+",
            value,
        ):
            raise ValueError(
                "lora.source must not embed credentials; use HF_TOKEN or Colab secrets"
            )
        return value

    @model_validator(mode="after")
    def validate_archive_policy(self) -> "LoraConfig":
        if self.source is not None and (
            self.fit_scale <= 0.0 or self.render_scale <= 0.0
        ):
            raise ValueError(
                "a configured LoRA requires positive fit and render scales so activation "
                "can be demonstrated numerically"
            )
        if self.include_in_output_archive:
            raise ValueError(
                "version 1 never packages downloaded/user LoRA weights; keep "
                "lora.include_in_output_archive=false"
            )
        return self


class InputConfig(StrictConfigModel):
    source_image: Path | None = None
    target_image: Path | None = None
    source_prompt: str | None = None
    target_prompt: str | None = None
    bridge_prompt: str | None = "a smooth transformation between the two subjects"
    negative_prompt: str = ""
    resize_mode: ResizeMode = ResizeMode.STRETCH
    width: int = Field(default=512, gt=0)
    height: int = Field(default=512, gt=0)


class ResolvedInputConfig(InputConfig):
    source_image: Path
    target_image: Path


class FlowMorphConfig(StrictConfigModel):
    scheduler_points: int = Field(default=100, ge=2)
    start_timestep_index: int = Field(default=35, ge=0)
    optimization_steps_source: int = Field(default=100, ge=1)
    optimization_steps_target: int = Field(default=100, ge=1)
    pred_learning_rate: float = Field(default=0.04, gt=0.0)
    u_learning_rate: float = Field(default=0.01, gt=0.0)
    optimizer: OptimizerName = OptimizerName.ADAMW
    weight_decay: float | None = Field(default=None, ge=0.0)
    loss_mode: LossMode = LossMode.CODE_L2_NORM
    frame_count: int = Field(default=20, ge=2)
    interpolation_mode: InterpolationMode = InterpolationMode.DECOUPLED
    render_indices: tuple[int, ...] = REFERENCE_RENDER_INDICES
    alpha_schedule: AlphaSchedule = AlphaSchedule.LINEAR
    render_conditioning_mode: RenderConditioningMode = RenderConditioningMode.SOURCE
    checkpoint_every: int = Field(default=25, ge=1)

    @model_validator(mode="after")
    def validate_schedule_indices(self) -> "FlowMorphConfig":
        indices = self.render_indices
        if not indices:
            raise ValueError("flowmorph.render_indices must not be empty")
        if any(index < 0 for index in indices):
            raise ValueError("flowmorph.render_indices cannot contain negative indices")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("flowmorph.render_indices must be strictly increasing")
        if indices[0] != self.start_timestep_index:
            raise ValueError(
                "the first render index must equal flowmorph.start_timestep_index"
            )
        if indices[-1] >= self.scheduler_points:
            raise ValueError(
                "all render indices must be smaller than flowmorph.scheduler_points"
            )
        return self


class GuidanceConfig(StrictConfigModel):
    enabled: bool = True
    scale: float = Field(default=4.0, ge=0.0)
    execution: CFGExecution = CFGExecution.SEQUENTIAL

    @field_validator("scale")
    @classmethod
    def validate_finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("guidance.scale must be finite")
        return value


class MemoryConfig(StrictConfigModel):
    text_encoder_offload: bool = True
    vae_offload: bool = True
    model_cpu_offload: bool = False
    sequential_cpu_offload: bool = False
    allow_degraded_run: bool = False
    run_production_backward_probe: bool = True
    save_intermediate_states: bool = False


class OutputConfig(StrictConfigModel):
    save_raw_frames: bool = True
    save_display_frames: bool = True
    save_endpoint_states: bool = True
    save_optimizer_states: bool = False
    save_loss_history: bool = True
    save_difference_images: bool = True
    save_contact_sheet: bool = True
    save_webp: bool = True
    save_gif: bool = True
    save_mp4: bool = True
    fps: int = Field(default=12, ge=1)
    create_zip: bool = True
    archive_suffix: str = ".flowmorph-klein.zip"
    copy_archive_to_drive: bool = False

    @field_validator(
        "save_raw_frames",
        "save_display_frames",
        "save_endpoint_states",
        "save_loss_history",
        "create_zip",
    )
    @classmethod
    def require_mandatory_artifacts(
        cls, value: bool, info: ValidationInfo
    ) -> bool:
        if not value:
            raise ValueError(
                f"output.{info.field_name} must be true because the runner and "
                "reproduction contract require this artifact"
            )
        return value

    @field_validator("archive_suffix")
    @classmethod
    def validate_archive_suffix(cls, value: str) -> str:
        if value != ".flowmorph-klein.zip":
            raise ValueError("output.archive_suffix must be '.flowmorph-klein.zip'")
        return value


class ReproducibilityConfig(StrictConfigModel):
    seed: int = 42
    deterministic_algorithms: bool = False
    record_environment: bool = True
    record_checksums: bool = True

    @field_validator("record_environment", "record_checksums")
    @classmethod
    def require_provenance_records(
        cls, value: bool, info: ValidationInfo
    ) -> bool:
        if not value:
            raise ValueError(
                f"reproducibility.{info.field_name} must be true because the "
                "reproduction contract requires this provenance record"
            )
        return value


class ProjectTemplateConfig(StrictConfigModel):
    """Serializable configuration before paths/profile are materialized."""

    run_mode: RunMode = RunMode.REFERENCE
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    flowmorph: FlowMorphConfig = Field(default_factory=FlowMorphConfig)
    guidance: GuidanceConfig = Field(default_factory=GuidanceConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)

    @model_validator(mode="after")
    def validate_cross_section_contract(self) -> "ProjectTemplateConfig":
        profile = self.model.profile
        if self.model.id == FP8_MODEL_ID:
            if profile is not HardwareProfile.FP8_9B_EXPERIMENTAL:
                raise ValueError(
                    "the Base-9B FP8 model requires profile 'fp8_9b_experimental'"
                )
            if self.run_mode is not RunMode.EXPERIMENTAL:
                raise ValueError("the Base-9B FP8 model requires run_mode 'experimental'")
        elif profile is HardwareProfile.FP8_9B_EXPERIMENTAL:
            raise ValueError("the FP8 profile requires the Base-9B FP8 model ID")

        if profile is HardwareProfile.UNSUPPORTED_LOW_VRAM:
            if self.run_mode is not RunMode.DIAGNOSTIC:
                raise ValueError(
                    "unsupported_low_vram is a diagnostic profile and cannot run fitting"
                )
        elif self.run_mode is RunMode.DIAGNOSTIC:
            raise ValueError(
                "run_mode 'diagnostic' requires profile 'unsupported_low_vram'"
            )

        if self.lora.fit_scale != self.lora.render_scale:
            warnings.warn(
                "different LoRA fit/render scales are experimental because endpoint "
                "variables were fitted against a different vector field",
                UserWarning,
                stacklevel=2,
            )

        if self.lora.allow_distilled_9b and self.run_mode is not RunMode.EXPERIMENTAL:
            raise ValueError(
                "lora.allow_distilled_9b is an explicit mismatch override and requires "
                "run_mode 'experimental'"
            )

        if (
            self.flowmorph.render_conditioning_mode
            is RenderConditioningMode.INTERPOLATED_EMBEDDINGS
            and self.run_mode is not RunMode.EXPERIMENTAL
        ):
            raise ValueError(
                "flowmorph.render_conditioning_mode='interpolated_embeddings' "
                "requires run_mode 'experimental'"
            )

        if self.run_mode is RunMode.REFERENCE:
            self._validate_reference_contract()
        elif self.run_mode is RunMode.SMOKE:
            self._validate_smoke_contract()
        elif self.run_mode is RunMode.EXPERIMENTAL:
            self._validate_full_shape_contract()
        elif self.run_mode is RunMode.DIAGNOSTIC:
            self._validate_full_shape_contract()
        return self

    def _validate_full_shape_contract(self) -> None:
        """Prevent memory profiles from changing production semantics."""

        expected = {
            "input.width": (self.input.width, 512),
            "input.height": (self.input.height, 512),
            "flowmorph.scheduler_points": (self.flowmorph.scheduler_points, 100),
            "flowmorph.start_timestep_index": (
                self.flowmorph.start_timestep_index,
                35,
            ),
            "flowmorph.optimization_steps_source": (
                self.flowmorph.optimization_steps_source,
                100,
            ),
            "flowmorph.optimization_steps_target": (
                self.flowmorph.optimization_steps_target,
                100,
            ),
            "flowmorph.frame_count": (self.flowmorph.frame_count, 20),
            "flowmorph.render_indices": (
                self.flowmorph.render_indices,
                REFERENCE_RENDER_INDICES,
            ),
            "guidance.enabled": (self.guidance.enabled, True),
            "model.optimization_parameter_dtype": (
                self.model.optimization_parameter_dtype,
                ComputeDType.FLOAT32,
            ),
            "memory.model_cpu_offload": (self.memory.model_cpu_offload, False),
            "memory.sequential_cpu_offload": (
                self.memory.sequential_cpu_offload,
                False,
            ),
            "memory.allow_degraded_run": (
                self.memory.allow_degraded_run,
                False,
            ),
            "memory.run_production_backward_probe": (
                self.memory.run_production_backward_probe,
                True,
            ),
            "memory.save_intermediate_states": (
                self.memory.save_intermediate_states,
                False,
            ),
        }
        mismatches = [
            f"{name}={actual!r} (required {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError(
                "the selected production profile cannot silently change semantics: "
                + ", ".join(mismatches)
            )

    def _validate_reference_contract(self) -> None:
        if self.model.id != BASE_MODEL_ID:
            raise ValueError("reference mode requires FLUX.2 Klein Base 9B")
        if self.model.profile not in {
            HardwareProfile.AUTO,
            HardwareProfile.A100_80GB_FULL,
            HardwareProfile.A100_40GB_CHECKPOINTED,
        }:
            raise ValueError("reference mode requires a supported Base-9B profile")
        self._validate_full_shape_contract()
        expected = {
            "model.transformer_compute_dtype": (
                self.model.transformer_compute_dtype,
                ComputeDType.BFLOAT16,
            ),
            "model.optimization_parameter_dtype": (
                self.model.optimization_parameter_dtype,
                ComputeDType.FLOAT32,
            ),
            "model.quantization": (self.model.quantization, QuantizationMode.NONE),
            "model.gradient_checkpointing": (self.model.gradient_checkpointing, True),
            "flowmorph.loss_mode": (
                self.flowmorph.loss_mode,
                LossMode.CODE_L2_NORM,
            ),
            "flowmorph.interpolation_mode": (
                self.flowmorph.interpolation_mode,
                InterpolationMode.DECOUPLED,
            ),
            "flowmorph.alpha_schedule": (
                self.flowmorph.alpha_schedule,
                AlphaSchedule.LINEAR,
            ),
            "flowmorph.render_conditioning_mode": (
                self.flowmorph.render_conditioning_mode,
                RenderConditioningMode.SOURCE,
            ),
            "guidance.scale": (self.guidance.scale, 4.0),
            "guidance.execution": (
                self.guidance.execution,
                CFGExecution.SEQUENTIAL,
            ),
            "memory.text_encoder_offload": (
                self.memory.text_encoder_offload,
                True,
            ),
            "memory.vae_offload": (self.memory.vae_offload, True),
            "memory.model_cpu_offload": (self.memory.model_cpu_offload, False),
            "memory.sequential_cpu_offload": (
                self.memory.sequential_cpu_offload,
                False,
            ),
            "memory.allow_degraded_run": (self.memory.allow_degraded_run, False),
            "output.save_raw_frames": (self.output.save_raw_frames, True),
            "output.save_display_frames": (self.output.save_display_frames, True),
            "output.create_zip": (self.output.create_zip, True),
        }
        mismatches = [
            f"{name}={actual!r} (required {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError(
                "reference mode contract violation: " + ", ".join(mismatches)
            )
        if self.lora.fit_scale != self.lora.render_scale:
            raise ValueError("reference mode requires identical LoRA fit/render scales")

    def _validate_smoke_contract(self) -> None:
        if self.model.id != BASE_MODEL_ID:
            raise ValueError("smoke mode still requires FLUX.2 Klein Base 9B")
        if self.model.optimization_parameter_dtype is not ComputeDType.FLOAT32:
            raise ValueError("smoke mode still requires FP32 endpoint parameters")
        if self.input.width != 512 or self.input.height != 512:
            raise ValueError("smoke mode must retain the production 512x512 shape")
        if self.flowmorph.frame_count != 3:
            raise ValueError("smoke mode must be explicitly configured for three frames")
        if not self.guidance.enabled:
            raise ValueError("smoke mode must not silently disable CFG")


class ResolvedRunConfig(ProjectTemplateConfig):
    """A runnable configuration with concrete endpoint paths and profile."""

    input: ResolvedInputConfig

    @model_validator(mode="after")
    def validate_resolved_values(self) -> "ResolvedRunConfig":
        if self.model.profile is HardwareProfile.AUTO:
            raise ValueError(
                "model.profile='auto' must be resolved by hardware detection before running"
            )
        required_revision = (
            "9ecf2143d71542449960c5584340269c6d401449"
            if self.model.id == FP8_MODEL_ID
            else "32773329fbe7e81a90ef971740e8ba4b0364ecf3"
        )
        if self.model.revision != required_revision:
            raise ValueError(
                f"resolved model revision must be pinned to {required_revision}; "
                f"received {self.model.revision!r}"
            )
        return self


ConfigT = TypeVar("ConfigT", bound=ProjectTemplateConfig)


def _validation_error(message: str, error: Exception) -> ConfigurationError:
    return ConfigurationError(f"{message}: {error}")


def validate_template(data: Mapping[str, Any]) -> ProjectTemplateConfig:
    """Validate an in-memory template and normalize all defaults."""

    try:
        return ProjectTemplateConfig.model_validate(data)
    except ValidationError as error:
        raise _validation_error("invalid FlowMorph Klein configuration", error) from error


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without constructing arbitrary Python objects."""

    config_path = Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise _validation_error(f"cannot load configuration {config_path}", error) from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"configuration {config_path} must contain a YAML mapping at its root"
        )
    return loaded


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings, replacing non-mapping leaves."""

    result: dict[str, Any] = deepcopy(dict(base))
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def _set_dotted_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ConfigurationError("CLI override keys cannot be empty")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            cursor[part] = {}
        elif not isinstance(existing, dict):
            raise ConfigurationError(
                f"cannot assign nested override {dotted_key!r}; {part!r} is not a mapping"
            )
        cursor = cursor[part]
    cursor[parts[-1]] = value


_CLI_ALIASES: dict[str, tuple[str, ...]] = {
    "source": ("input.source_image",),
    "source_image": ("input.source_image",),
    "target": ("input.target_image",),
    "target_image": ("input.target_image",),
    "lora-source": ("lora.source",),
    "lora_source": ("lora.source",),
    "lora-revision": ("lora.revision",),
    "lora_revision": ("lora.revision",),
    "lora-scale": ("lora.fit_scale", "lora.render_scale"),
    "lora_scale": ("lora.fit_scale", "lora.render_scale"),
    "profile": ("model.profile",),
    "frames": ("flowmorph.frame_count",),
    "seed": ("reproducibility.seed",),
}


def parse_cli_overrides(arguments: Sequence[str]) -> dict[str, Any]:
    """Parse dotted overrides and documented ``--key value`` arguments.

    Values use YAML scalar syntax, so booleans, nulls, numbers, and lists are
    typed without executing code.  The aliases mirror the documented CLI.
    """

    result: dict[str, Any] = {}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        item = argument[2:] if argument.startswith("--") else argument
        if "=" in item:
            raw_key, raw_value = item.split("=", 1)
        elif argument.startswith("--") and index + 1 < len(arguments):
            raw_key = item
            index += 1
            raw_value = arguments[index]
        else:
            raise ConfigurationError(
                f"CLI override {argument!r} must use key=value or --key value syntax"
            )
        key = raw_key.strip()
        if not key:
            raise ConfigurationError("CLI override keys cannot be empty")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise _validation_error(f"invalid value for CLI override {key!r}", error) from error
        if value is None and raw_value.strip().lower() not in {"null", "~"}:
            value = ""
        destinations = _CLI_ALIASES.get(key, (key,))
        for destination in destinations:
            _set_dotted_value(result, destination, value)
        index += 1
    return result


def cli_namespace_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ``vars(argparse_namespace)`` into overlays, ignoring unset values."""

    return normalize_overrides({key: value for key, value in values.items() if value is not None})


def normalize_overrides(
    overrides: Mapping[str, Any] | Sequence[str] | None,
) -> dict[str, Any]:
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        normalized: dict[str, Any] = {}
        for key, value in overrides.items():
            if "." in key or key in _CLI_ALIASES:
                destinations = _CLI_ALIASES.get(key, (key,))
                for destination in destinations:
                    _set_dotted_value(normalized, destination, value)
            elif isinstance(value, Mapping):
                normalized[key] = deepcopy(dict(value))
            else:
                normalized[key] = deepcopy(value)
        return normalized
    if isinstance(overrides, (str, bytes)):
        raise ConfigurationError("overrides must be a mapping or a sequence of key=value strings")
    return parse_cli_overrides(overrides)


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> ProjectTemplateConfig:
    """Load a YAML template, apply explicit overrides, and validate it."""

    data = deep_merge(load_yaml_mapping(path), normalize_overrides(overrides))
    return validate_template(data)


def resolve_config(
    template: ProjectTemplateConfig | Mapping[str, Any],
    *,
    selected_profile: HardwareProfile | str | None = None,
    source_image: str | Path | None = None,
    target_image: str | Path | None = None,
    check_input_files: bool = True,
) -> ResolvedRunConfig:
    """Materialize a runnable configuration and optionally verify input files."""

    if isinstance(template, ProjectTemplateConfig):
        data = template.model_dump(mode="python")
    else:
        data = validate_template(template).model_dump(mode="python")

    if selected_profile is not None:
        data["model"]["profile"] = selected_profile
    # Templates may leave the immutable revision unset for readability, but
    # runnable configurations always materialize the audited revision before
    # validation or any model access.
    if data["model"].get("revision") is None:
        data["model"]["revision"] = (
            "9ecf2143d71542449960c5584340269c6d401449"
            if data["model"].get("id") == FP8_MODEL_ID
            else "32773329fbe7e81a90ef971740e8ba4b0364ecf3"
        )
    if source_image is not None:
        data["input"]["source_image"] = Path(source_image)
    if target_image is not None:
        data["input"]["target_image"] = Path(target_image)

    try:
        resolved = ResolvedRunConfig.model_validate(data)
    except ValidationError as error:
        raise _validation_error("configuration is not runnable", error) from error

    if check_input_files:
        missing = [
            str(path)
            for path in (resolved.input.source_image, resolved.input.target_image)
            if not path.is_file()
        ]
        if missing:
            raise ConfigurationError(
                "endpoint images must exist before model download: " + ", ".join(missing)
            )
    return resolved


def canonical_config_hash(config: ProjectTemplateConfig) -> str:
    """Return a stable SHA-256 for checkpoint and provenance comparisons."""

    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
