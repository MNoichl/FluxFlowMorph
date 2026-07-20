"""End-to-end orchestration for the FLUX.2 Klein Base-9B reproduction.

The runner deliberately keeps orchestration separate from the numerical
modules.  It owns the expensive component lifecycle, immutable provenance,
phase transitions, compatible checkpoint resume, and output publication.  No
high-level Diffusers pipeline call is used for fitting or rendering.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from PIL import Image

from . import (
    DIFFUSERS_COMMIT,
    FLOWMORPH_COMMIT,
    FLUX2_COMMIT,
    FP8_MODEL_ID,
    MODEL_ID,
    MODEL_REVISION,
)
from .acceptance import (
    PHASE_ORDER,
    AcceptanceReport,
    RunPhase,
    require_completed_run,
    validate_phase_transition,
)
from .checkpoints import (
    CheckpointError,
    LoadedCheckpoint,
    load_endpoint_checkpoint,
    save_endpoint_checkpoint,
    unflatten_optimizer_state,
)
from .colab_io import create_run_id, sha256_file
from .conditioning import (
    ConditioningCache,
    ConditioningPackage,
    build_conditioning_cache,
    interpolate_conditioning,
)
from .config import ResolvedRunConfig, canonical_config_hash
from .diagnostics import (
    BackwardProbeReport,
    cuda_memory_snapshot,
    release_cuda_memory,
    run_backward_probe,
)
from .endpoint_optimizer import (
    EndpointOptimizationResult,
    EndpointOptimizerConfig,
    OptimizationStepDiagnostics,
    optimize_endpoint,
)
from .environment import (
    AuthenticationResult,
    collect_environment,
    redact_secrets,
    require_cuda_for_production,
    resolve_hf_token,
    verify_model_access,
    write_environment,
)
from .flow_schedule import (
    FlowSchedule,
    build_flowmorph_schedule,
    get_render_chain,
    get_start_state_metadata,
)
from .flow_state import FlowMorphEndpoint
from .flux2_latents import (
    decode_packed_latent,
    encode_image_to_packed_latent,
    preprocess_endpoint_image as flux2_preprocess_endpoint_image,
)
from .flux2_model import FlowMorphFlux2Model
from .image_io import preprocess_endpoint_pair
from .lora import (
    LoraLoadReport,
    compare_lora_velocities,
    load_flux2_lora,
    verify_active_adapter,
)
from .metrics import (
    endpoint_reconstruction_metrics,
    summarize_optimization,
    transition_metrics,
    write_csv,
    write_metrics,
)
from .packaging import ArchiveReport, create_run_archive
from .renderer import RenderedLatentFrame, render_morph
from .types import (
    ComputeDType,
    PreprocessedImage,
    RenderConditioningMode,
    RunMode,
)
from .video import save_gif, save_mp4, save_webp
from .visualization import (
    difference_image,
    make_contact_sheet,
    save_endpoint_comparison,
    save_loss_plot,
)


class PipelineError(RuntimeError):
    """The production workflow cannot safely continue."""


@dataclass(frozen=True)
class FlowMorphRunResult:
    run_id: str
    run_directory: Path
    phase: str
    metrics_path: Path
    archive: ArchiveReport | None
    acceptance: AcceptanceReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_directory),
            "phase": self.phase,
            "metrics_path": str(self.metrics_path),
            "archive": _jsonable(self.archive),
            "acceptance": self.acceptance.as_dict(),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> Path:
    serialized = json.dumps(
        _jsonable(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    _atomic_text(path, redact_secrets(serialized) + "\n")
    return path


def _torch_dtype(value: ComputeDType | str) -> torch.dtype:
    normalized = getattr(value, "value", value)
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[str(normalized)]
    except KeyError as error:
        raise PipelineError(f"unsupported compute dtype {normalized!r}") from error


def _module_device(module: Any) -> torch.device:
    device = getattr(module, "device", None)
    if isinstance(device, torch.device):
        return device
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.device
    return torch.device("cpu")


def _module_dtype(module: Any, default: torch.dtype = torch.float32) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    return default


def _move_module(module: Any, device: torch.device | str) -> None:
    mover = getattr(module, "to", None)
    if not callable(mover):
        raise PipelineError(f"{type(module).__name__} cannot be moved to {device}")
    mover(device)


def _freeze_module(module: Any) -> None:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    evaluation = getattr(module, "eval", None)
    if callable(evaluation):
        evaluation()


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    """Recognize PyTorch CUDA OOMs without classifying unrelated failures."""

    if isinstance(error, torch.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def _distribution_direct_url(name: str) -> dict[str, Any] | None:
    """Read PEP-610 provenance without importing private package internals."""

    try:
        distribution = importlib.metadata.distribution(name)
        text = distribution.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class _BoundCFGVelocityPredictor:
    """Bind fixed CFG and image-ID inputs to the small optimizer protocol."""

    def __init__(
        self,
        model: FlowMorphFlux2Model,
        unconditional: ConditioningPackage,
        image_ids: torch.Tensor,
        *,
        guidance_scale: float,
        cfg_enabled: bool,
        cfg_execution: str,
    ) -> None:
        self.model = model
        self.unconditional = unconditional
        self.image_ids = image_ids
        self.guidance_scale = guidance_scale
        self.cfg_enabled = cfg_enabled
        self.cfg_execution = cfg_execution

    def parameters(self):
        return self.model.parameters()

    def predict_velocity(
        self,
        state: torch.Tensor,
        timestep: Any,
        conditioning: ConditioningPackage,
    ) -> torch.Tensor:
        return self.model.predict_cfg_velocity(
            state,
            timestep,
            conditional=conditioning,
            unconditional=self.unconditional,
            image_ids=self.image_ids,
            guidance_scale=self.guidance_scale,
            cfg_enabled=self.cfg_enabled,
            cfg_execution=self.cfg_execution,
        )


class FlowMorphRunner:
    """One-model, sequential-endpoint FlowMorph production runner."""

    def __init__(
        self,
        config: ResolvedRunConfig,
        *,
        run_directory: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        if not isinstance(config, ResolvedRunConfig):
            raise TypeError("FlowMorphRunner requires a ResolvedRunConfig")
        self.config = config
        self.config_hash = canonical_config_hash(config)
        selected_id = run_id or create_run_id(config.project.name)
        selected_directory = (
            Path(run_directory).expanduser().resolve(strict=False)
            if run_directory is not None
            else (config.paths.result_root / selected_id).expanduser().resolve(strict=False)
        )
        self.run_directory = selected_directory
        self.run_id = selected_directory.name if run_directory is not None else selected_id
        self.run_directory.mkdir(parents=True, exist_ok=True)

        self.device: torch.device | None = None
        self.authentication: AuthenticationResult | None = None
        self.pipeline: Any | None = None
        self.model: FlowMorphFlux2Model | None = None
        self.conditioning_cache: ConditioningCache | None = None
        self.source_latent: torch.Tensor | None = None
        self.target_latent: torch.Tensor | None = None
        self.image_ids: torch.Tensor | None = None
        self.schedule: FlowSchedule | None = None
        self.source_preprocessed: PreprocessedImage | None = None
        self.target_preprocessed: PreprocessedImage | None = None
        self.source_endpoint: FlowMorphEndpoint | None = None
        self.target_endpoint: FlowMorphEndpoint | None = None
        self.lora_load_report: LoraLoadReport | None = None
        self.lora_report: dict[str, Any] = {"status": "not_configured"}
        self.model_report: dict[str, Any] = {}
        self.memory_report: dict[str, Any] = {}
        self.archive_report: ArchiveReport | None = None
        self.acceptance_report: AcceptanceReport | None = None
        # A durable probe report is provenance, not authorization for a newly
        # started CUDA process. Every prepared runner must prove the current
        # model/runtime path once before fitting can continue.
        self._session_backward_probe_report: BackwardProbeReport | None = None
        self._base_component_access: dict[str, Any] | None = None
        self._offload_report: dict[str, Any] = {}
        self._prepared = False

        self.manifest: dict[str, Any] = self._load_or_initialize_manifest()
        self.phase = RunPhase(self.manifest["phase"])
        self._write_config()
        self._write_manifest()
        self._log(f"run initialized: {self.run_id}")

    @classmethod
    def from_config(
        cls,
        config: ResolvedRunConfig,
        *,
        run_directory: str | Path | None = None,
        run_id: str | None = None,
    ) -> "FlowMorphRunner":
        return cls(config, run_directory=run_directory, run_id=run_id)

    def _load_or_initialize_manifest(self) -> dict[str, Any]:
        path = self.run_directory / "run_manifest.json"
        if path.is_file():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise PipelineError(f"cannot read existing run manifest {path}: {error}") from error
            existing_hash = manifest.get("config_hash")
            if existing_hash != self.config_hash:
                raise PipelineError(
                    "existing run directory was created from a different resolved configuration"
                )
            phase = manifest.get("phase")
            try:
                RunPhase(phase)
            except ValueError as error:
                raise PipelineError(f"existing manifest has invalid phase {phase!r}") from error
            return manifest

        now = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "project": self.config.project.name,
            "created_at": now,
            "updated_at": now,
            "phase": RunPhase.CREATED.value,
            "last_successful_phase": RunPhase.CREATED.value,
            "config_hash": self.config_hash,
            "model_id": self.config.model.id,
            "model_revision": self.config.model.revision,
            "profile": self.config.model.profile.value,
            "run_mode": self.config.run_mode.value,
            "allow_degraded_run": self.config.memory.allow_degraded_run,
            "backward_probe_status": "not_run",
            "lora_status": "not_configured" if self.config.lora.source is None else "not_verified",
            "source_completed_steps": 0,
            "target_completed_steps": 0,
            "raw_frame_count": 0,
            "display_frame_count": 0,
            "distribution_review_required": True,
            "generated_with_ai_model": True,
            "reference_reproduction_claimed": False,
        }

    def _write_config(self) -> None:
        data = self.config.model_dump(mode="json")
        serialized = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        _atomic_text(
            self.run_directory / "config.resolved.yaml",
            redact_secrets(serialized),
        )

    def _write_manifest(self) -> None:
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(self.run_directory / "run_manifest.json", self.manifest)

    def _log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} {redact_secrets(str(message))}\n"
        path = self.run_directory / "execution.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _at_least(self, phase: RunPhase) -> bool:
        if self.phase is RunPhase.FAILED:
            return False
        return PHASE_ORDER[self.phase] >= PHASE_ORDER[phase]

    def _advance(self, target: RunPhase) -> None:
        if self.phase is target:
            return
        validate_phase_transition(self.phase, target)
        self.phase = target
        self.manifest["phase"] = target.value
        self.manifest["last_successful_phase"] = target.value
        self._write_manifest()
        self._log(f"phase completed: {target.value}")

    def _restore_failed_phase_for_resume(self) -> None:
        if self.phase is not RunPhase.FAILED:
            return
        raw = self.manifest.get("last_successful_phase", RunPhase.CREATED.value)
        restored = RunPhase(raw)
        if restored is RunPhase.FAILED:
            restored = RunPhase.CREATED
        self.phase = restored
        self.manifest["phase"] = restored.value
        self.manifest["resumed_after_failure_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest()
        self._log(f"explicit resume restored phase {restored.value}")

    def _record_failure(self, error: BaseException, *, operation: str) -> None:
        message = redact_secrets(f"{type(error).__name__}: {error}")
        if self.phase is not RunPhase.FAILED:
            self.manifest["last_successful_phase"] = self.phase.value
        self.phase = RunPhase.FAILED
        self.manifest.update(
            {
                "phase": RunPhase.FAILED.value,
                "failed_operation": operation,
                "failure": message,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "reference_reproduction_claimed": False,
            }
        )
        self._write_manifest()
        self._log(f"FAILED during {operation}: {message}")

    def _require_prepared_values(self) -> tuple[
        Any,
        FlowMorphFlux2Model,
        ConditioningCache,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        FlowSchedule,
    ]:
        values = (
            self.pipeline,
            self.model,
            self.conditioning_cache,
            self.source_latent,
            self.target_latent,
            self.image_ids,
            self.schedule,
        )
        if any(value is None for value in values):
            raise PipelineError("runner is not prepared")
        return values  # type: ignore[return-value]

    def prepare(self, *, resume: bool = False) -> "FlowMorphRunner":
        """Validate inputs, load one exact model, cache inputs, and verify LoRA."""

        if self._prepared:
            return self
        reuse_persisted_inputs = False
        if "inputs" in self.manifest:
            if not resume:
                raise PipelineError(
                    "this run already contains staged inputs; use explicit resume to "
                    "avoid overwriting prior run evidence"
                )
            # This check deliberately precedes phase restoration, environment
            # collection, and every file copy. A mutable path must not turn an
            # old checkpoint directory into a mixed-input failure archive.
            self._validate_resume_artifacts_unchanged()
            reuse_persisted_inputs = True
        if self.phase is RunPhase.FAILED:
            if not resume:
                raise PipelineError(
                    "this run is marked failed; call prepare(resume=True) or resume() explicitly"
                )
            self._restore_failed_phase_for_resume()
        try:
            if (
                self.config.memory.model_cpu_offload
                or self.config.memory.sequential_cpu_offload
            ):
                raise PipelineError(
                    "Accelerate model/sequential CPU offload is not validated for the "
                    "differentiable transformer path; use the explicit VAE/text-encoder "
                    "offload profiles instead"
                )
            self.device = require_cuda_for_production()
            self._configure_reproducibility()
            self._prepare_inputs(reuse_persisted=reuse_persisted_inputs)
            if not self._at_least(RunPhase.INPUTS_VALIDATED):
                self._advance(RunPhase.INPUTS_VALIDATED)

            self.authentication = resolve_hf_token()
            access = verify_model_access(
                self.authentication,
                model_id=self.config.model.id,
                revision=self.config.model.revision,
            )
            self.pipeline = self._load_pipeline(self.authentication)
            self.model_report = self._verify_loaded_pipeline(self.pipeline, access)
            _write_json(self.run_directory / "model_report.json", self.model_report)
            if not self._at_least(RunPhase.MODEL_READY):
                self._advance(RunPhase.MODEL_READY)

            self._cache_conditioning_and_latents()
            self._load_and_test_lora()
            self._configure_transformer()
            if not self._at_least(RunPhase.ADAPTER_VERIFIED):
                self._advance(RunPhase.ADAPTER_VERIFIED)
            self._prepared = True
            return self
        except BaseException as error:
            self._record_failure(error, operation="prepare")
            raise

    def _configure_reproducibility(self) -> None:
        seed = self.config.reproducibility.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(
            self.config.reproducibility.deterministic_algorithms,
            warn_only=not self.config.reproducibility.deterministic_algorithms,
        )
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.config.model.use_tf32
            torch.backends.cudnn.allow_tf32 = self.config.model.use_tf32
        environment = collect_environment()
        environment["selected_profile"] = self.config.model.profile.value
        environment["resolved_config_hash"] = self.config_hash
        write_environment(self.run_directory / "environment.json", environment)

    def _validate_resume_artifacts_unchanged(self) -> None:
        """Verify mutable inputs before a resumed run can overwrite anything."""

        records = self.manifest.get("inputs")
        if not isinstance(records, Mapping):
            raise PipelineError("existing run manifest has malformed input provenance")
        input_directory = self.run_directory / "inputs"
        mismatches: list[str] = []
        for label, source in (
            ("source", self.config.input.source_image),
            ("target", self.config.input.target_image),
        ):
            record = records.get(label)
            if not isinstance(record, Mapping):
                mismatches.append(f"{label}: missing manifest record")
                continue
            expected_original = record.get("original_sha256")
            actual_original = sha256_file(source)
            if actual_original != expected_original:
                mismatches.append(
                    f"{label}: current source checksum differs from the staged run"
                )

            suffix = source.suffix.lower() or ".bin"
            staged_original = input_directory / f"{label}_original{suffix}"
            if not staged_original.is_file():
                mismatches.append(f"{label}: staged original is missing")
            elif sha256_file(staged_original) != expected_original:
                mismatches.append(f"{label}: staged original checksum is corrupt")

            processed = input_directory / f"{label}_preprocessed.png"
            expected_processed = record.get("processed_sha256")
            if not processed.is_file():
                mismatches.append(f"{label}: persisted preprocessed image is missing")
            elif sha256_file(processed) != expected_processed:
                mismatches.append(
                    f"{label}: persisted preprocessed image checksum is corrupt"
                )

        configured_lora = self.config.lora.source
        recorded_lora = self.manifest.get("lora")
        if configured_lora and isinstance(recorded_lora, Mapping):
            local_lora = Path(configured_lora).expanduser()
            if local_lora.is_file():
                expected_lora = recorded_lora.get("sha256")
                if sha256_file(local_lora) != expected_lora:
                    mismatches.append(
                        "LoRA: current local file checksum differs from the staged run"
                    )
        if mismatches:
            raise PipelineError(
                "resume provenance validation failed before staging: "
                + "; ".join(mismatches)
            )

    def _load_persisted_preprocessed_inputs(
        self,
    ) -> tuple[PreprocessedImage, PreprocessedImage]:
        records = self.manifest["inputs"]
        input_directory = self.run_directory / "inputs"

        def load(label: str, source: Path) -> PreprocessedImage:
            record = records[label]
            output_path = input_directory / f"{label}_preprocessed.png"
            with Image.open(output_path) as opened:
                opened.load()
                image = opened.convert("RGB").copy()
            original_size = tuple(int(value) for value in record["original_size"])
            processed_size = tuple(int(value) for value in record["processed_size"])
            if len(original_size) != 2 or len(processed_size) != 2:
                raise PipelineError(
                    f"persisted {label} input metadata contains invalid dimensions"
                )
            if image.size != processed_size:
                raise PipelineError(
                    f"persisted {label} image dimensions disagree with its manifest"
                )
            return PreprocessedImage(
                image=image,
                source_path=source,
                output_path=output_path,
                original_size=original_size,
                processed_size=processed_size,
                resize_mode=self.config.input.resize_mode,
                original_sha256=str(record["original_sha256"]),
                preprocessing_sha256=str(record["preprocessing_sha256"]),
            )

        return (
            load("source", self.config.input.source_image),
            load("target", self.config.input.target_image),
        )

    def _prepare_inputs(self, *, reuse_persisted: bool = False) -> None:
        if reuse_persisted:
            self.source_preprocessed, self.target_preprocessed = (
                self._load_persisted_preprocessed_inputs()
            )
            return
        input_directory = self.run_directory / "inputs"
        input_directory.mkdir(parents=True, exist_ok=True)
        for label, source in (
            ("source", self.config.input.source_image),
            ("target", self.config.input.target_image),
        ):
            suffix = source.suffix.lower() or ".bin"
            destination = input_directory / f"{label}_original{suffix}"
            if source.resolve() != destination.resolve(strict=False):
                shutil.copy2(source, destination)
        source, target = preprocess_endpoint_pair(
            self.config.input.source_image,
            self.config.input.target_image,
            width=self.config.input.width,
            height=self.config.input.height,
            resize_mode=self.config.input.resize_mode,
            output_directory=input_directory,
            divisibility=16,
        )
        self.source_preprocessed = source
        self.target_preprocessed = target
        self.manifest["inputs"] = {
            "source": self._preprocessed_record(source),
            "target": self._preprocessed_record(target),
            "width": self.config.input.width,
            "height": self.config.input.height,
            "resize_mode": self.config.input.resize_mode.value,
        }
        self._write_manifest()

    @staticmethod
    def _preprocessed_record(value: PreprocessedImage) -> dict[str, Any]:
        if value.output_path is None:
            raise PipelineError("preprocessed image was not persisted")
        return {
            "original_sha256": value.original_sha256,
            "processed_sha256": sha256_file(value.output_path),
            "preprocessing_sha256": value.preprocessing_sha256,
            "original_size": list(value.original_size),
            "processed_size": list(value.processed_size),
            "resize_mode": value.resize_mode.value,
        }

    def _load_pipeline(self, authentication: AuthenticationResult) -> Any:
        try:
            from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        except ImportError as error:
            raise PipelineError(
                "the pinned Diffusers installation with Flux2KleinPipeline is required"
            ) from error

        dtype = _torch_dtype(self.config.model.transformer_compute_dtype)
        common = {
            "token": authentication.token,
            "cache_dir": str(self.config.paths.hf_cache),
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            if self.config.model.id == FP8_MODEL_ID:
                from huggingface_hub import hf_hub_download

                self._base_component_access = verify_model_access(
                    authentication,
                    model_id=MODEL_ID,
                    revision=MODEL_REVISION,
                )
                fp8_file = hf_hub_download(
                    repo_id=FP8_MODEL_ID,
                    filename="flux-2-klein-base-9b-fp8.safetensors",
                    revision=self.config.model.revision,
                    token=authentication.token,
                    cache_dir=str(self.config.paths.hf_cache),
                )
                transformer = Flux2Transformer2DModel.from_single_file(
                    fp8_file,
                    config=MODEL_ID,
                    subfolder="transformer",
                    config_revision=MODEL_REVISION,
                    token=authentication.token,
                    cache_dir=str(self.config.paths.hf_cache),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                layerwise_casting = getattr(transformer, "enable_layerwise_casting", None)
                if not callable(layerwise_casting):
                    raise PipelineError(
                        "pinned Diffusers transformer lacks layerwise FP8 storage casting"
                    )
                layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=dtype,
                )
                # ModelMixin.dtype now reflects storage for some first
                # parameters; the low-level velocity path must still cast
                # packed states to the configured BF16 compute dtype.
                transformer._flowmorph_compute_dtype = dtype
                pipeline = Flux2KleinPipeline.from_pretrained(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    transformer=transformer,
                    **common,
                )
            else:
                pipeline = Flux2KleinPipeline.from_pretrained(
                    self.config.model.id,
                    revision=self.config.model.revision,
                    **common,
                )
        except Exception as error:
            profile = self.config.model.profile.value
            raise PipelineError(
                f"failed to load exact model {self.config.model.id!r} at revision "
                f"{self.config.model.revision!r} for profile {profile!r}: "
                f"{redact_secrets(str(error))}"
            ) from error
        return pipeline

    def _verify_loaded_pipeline(self, pipeline: Any, access: Mapping[str, Any]) -> dict[str, Any]:
        try:
            installed_diffusers_version = importlib.metadata.version("diffusers")
        except importlib.metadata.PackageNotFoundError as error:
            raise PipelineError("Diffusers distribution metadata is unavailable") from error
        if installed_diffusers_version != "0.39.0":
            raise PipelineError(
                "the inspected Diffusers 0.39.0 build is required; found "
                f"{installed_diffusers_version!r}"
            )
        direct_url = _distribution_direct_url("diffusers")
        vcs_info = direct_url.get("vcs_info", {}) if isinstance(direct_url, Mapping) else {}
        installed_commit = vcs_info.get("commit_id") if isinstance(vcs_info, Mapping) else None
        if installed_commit != DIFFUSERS_COMMIT:
            raise PipelineError(
                "Diffusers must be installed from the exact inspected Git commit; "
                f"found provenance {installed_commit!r}, expected {DIFFUSERS_COMMIT!r}. "
                "Install requirements-colab.txt."
            )
        if type(pipeline).__name__ != "Flux2KleinPipeline":
            raise PipelineError(
                f"loaded pipeline class is {type(pipeline).__name__}, expected Flux2KleinPipeline"
            )
        config = getattr(pipeline, "config", None)
        is_distilled = (
            config.get("is_distilled")
            if isinstance(config, Mapping)
            else getattr(config, "is_distilled", None)
        )
        if is_distilled is not False:
            raise PipelineError(
                "loaded pipeline does not explicitly identify itself as undistilled Klein Base 9B"
            )
        transformer = getattr(pipeline, "transformer", None)
        vae = getattr(pipeline, "vae", None)
        scheduler = getattr(pipeline, "scheduler", None)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if any(component is None for component in (transformer, vae, scheduler, text_encoder)):
            raise PipelineError("loaded pipeline is missing a required FLUX.2 component")
        required_component_classes = {
            "transformer": (transformer, "Flux2Transformer2DModel"),
            "vae": (vae, "AutoencoderKLFlux2"),
            "scheduler": (scheduler, "FlowMatchEulerDiscreteScheduler"),
            "text_encoder": (text_encoder, "Qwen3ForCausalLM"),
        }
        wrong_classes = [
            f"{name}={type(component).__name__} (expected {expected})"
            for name, (component, expected) in required_component_classes.items()
            if type(component).__name__ != expected
        ]
        if wrong_classes:
            raise PipelineError(
                "loaded pipeline component classes do not match the pinned contract: "
                + ", ".join(wrong_classes)
            )
        expected = {
            "in_channels": 128,
            "num_layers": 8,
            "num_single_layers": 24,
            "attention_head_dim": 128,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
            "guidance_embeds": False,
        }
        transformer_config = getattr(transformer, "config", None)
        observed: dict[str, Any] = {}
        mismatches: list[str] = []
        for key, required in expected.items():
            actual = (
                transformer_config.get(key)
                if isinstance(transformer_config, Mapping)
                else getattr(transformer_config, key, None)
            )
            observed[key] = actual
            if actual != required:
                mismatches.append(f"{key}={actual!r}, expected {required!r}")
        if mismatches:
            raise PipelineError(
                "loaded transformer is not the Klein 9B architecture: " + ", ".join(mismatches)
            )
        parameter_dtypes: dict[str, int] = {}
        parameter_count = 0
        for parameter in transformer.parameters():
            parameter_count += parameter.numel()
            label = str(parameter.dtype).removeprefix("torch.")
            parameter_dtypes[label] = parameter_dtypes.get(label, 0) + parameter.numel()
        if parameter_count <= 0:
            raise PipelineError("loaded transformer contains no parameters")
        if self.config.model.id == FP8_MODEL_ID and not any(
            label.startswith("float8") for label in parameter_dtypes
        ):
            raise PipelineError(
                "experimental FP8 repository loaded without any float8 transformer parameters"
            )
        resolved = access.get("resolved_revision")
        if resolved != self.config.model.revision:
            raise PipelineError(
                f"Hub resolved model revision {resolved!r}, expected {self.config.model.revision!r}"
            )
        report = {
            "status": "loaded_and_structurally_verified",
            "model_id": self.config.model.id,
            "requested_revision": self.config.model.revision,
            "resolved_revision": resolved,
            "reference_bf16_model": self.config.model.id == MODEL_ID,
            "experimental_fp8_model": self.config.model.id == FP8_MODEL_ID,
            "profile": self.config.model.profile.value,
            "pipeline_class": type(pipeline).__name__,
            "transformer_class": type(transformer).__name__,
            "vae_class": type(vae).__name__,
            "scheduler_class": type(scheduler).__name__,
            "text_encoder_class": type(text_encoder).__name__,
            "is_distilled": is_distilled,
            "transformer_configuration": observed,
            "transformer_parameter_count": parameter_count,
            "transformer_parameter_dtypes": parameter_dtypes,
            "configured_compute_dtype": self.config.model.transformer_compute_dtype.value,
            "quantization": self.config.model.quantization.value,
            "diffusers_commit_expected": DIFFUSERS_COMMIT,
            "diffusers_version_installed": installed_diffusers_version,
            "diffusers_commit_installed": installed_commit,
            "diffusers_direct_url": direct_url,
            "model_access": dict(access),
            "base_component_access": self._base_component_access,
            "production_backward_probe": "not_run",
        }
        if self.config.model.id == FP8_MODEL_ID:
            report["experimental_warning"] = (
                "FP8 Base-9B is not a reference reproduction and is unsupported until its "
                "production-shape backward probe passes on this runtime."
            )
        return report

    def _cache_conditioning_and_latents(self) -> None:
        assert self.pipeline is not None and self.device is not None
        assert self.source_preprocessed is not None and self.target_preprocessed is not None
        pipe = self.pipeline

        _freeze_module(pipe.text_encoder)
        _freeze_module(pipe.vae)
        _freeze_module(pipe.transformer)

        _move_module(pipe.text_encoder, self.device)
        text_encoder_before = cuda_memory_snapshot(self.device)
        self.conditioning_cache = build_conditioning_cache(
            pipe,
            source_prompt=self.config.input.source_prompt,
            target_prompt=self.config.input.target_prompt,
            bridge_prompt=self.config.input.bridge_prompt,
            negative_prompt=self.config.input.negative_prompt,
            device=self.device,
            offload_to_cpu=True,
        )
        if self.config.memory.text_encoder_offload:
            _move_module(pipe.text_encoder, "cpu")
            release_cuda_memory()
        text_encoder_after = cuda_memory_snapshot(self.device)
        self._offload_report["text_encoder"] = self._offload_delta(
            text_encoder_before, text_encoder_after
        )

        prompt_manifest = {
            "policy": "source/target prompt; bridge fallback; then neutral 'an image'",
            "negative_prompt": self.config.input.negative_prompt,
            "render_conditioning_mode": self.config.flowmorph.render_conditioning_mode.value,
            "prompt_hashes": self.conditioning_cache.prompt_hashes,
            "packages": {
                name: {
                    "prompt": package.prompt,
                    "shape": list(package.prompt_embeds.shape),
                    "text_ids_shape": list(package.text_ids.shape),
                    "dtype": str(package.prompt_embeds.dtype).removeprefix("torch."),
                }
                for name, package in self.conditioning_cache.as_dict().items()
            },
        }
        _write_json(
            self.run_directory / "conditioning" / "prompt_manifest.json",
            prompt_manifest,
        )

        _move_module(pipe.vae, self.device)
        vae_before = cuda_memory_snapshot(self.device)
        vae_dtype = _module_dtype(pipe.vae, _torch_dtype(self.config.model.transformer_compute_dtype))
        source_image_tensor = flux2_preprocess_endpoint_image(
            self.source_preprocessed.image,
            pipe.image_processor,
            height=self.config.input.height,
            width=self.config.input.width,
            resize_mode="default",
        ).to(device=self.device, dtype=vae_dtype)
        target_image_tensor = flux2_preprocess_endpoint_image(
            self.target_preprocessed.image,
            pipe.image_processor,
            height=self.config.input.height,
            width=self.config.input.width,
            resize_mode="default",
        ).to(device=self.device, dtype=vae_dtype)
        with torch.inference_mode():
            source_latent, source_ids = encode_image_to_packed_latent(
                source_image_tensor,
                pipe.vae,
                preprocessed=True,
            )
            target_latent, target_ids = encode_image_to_packed_latent(
                target_image_tensor,
                pipe.vae,
                preprocessed=True,
            )
        if source_latent.shape != target_latent.shape:
            raise PipelineError("source and target packed latent shapes differ")
        if not torch.equal(source_ids, target_ids):
            raise PipelineError("source and target image position IDs differ")
        transformer_config = getattr(pipe.transformer, "config", None)
        expected_feature_width = (
            transformer_config.get("in_channels")
            if isinstance(transformer_config, Mapping)
            else getattr(transformer_config, "in_channels", None)
        )
        if (
            source_latent.ndim != 3
            or expected_feature_width is None
            or source_latent.shape[-1] != int(expected_feature_width)
        ):
            raise PipelineError(
                "packed latent shape does not match the loaded transformer input width: "
                f"{tuple(source_latent.shape)} versus {expected_feature_width!r}"
            )
        vae_config = getattr(pipe.vae, "config", None)
        loaded_patch_size = (
            vae_config.get("patch_size")
            if isinstance(vae_config, Mapping)
            else getattr(vae_config, "patch_size", None)
        )
        if loaded_patch_size is None:
            raise PipelineError("loaded FLUX.2 VAE config does not expose patch_size")
        self.source_latent = source_latent.detach().to("cpu")
        self.target_latent = target_latent.detach().to("cpu")
        self.image_ids = source_ids.detach().to("cpu")
        del source_image_tensor, target_image_tensor, source_latent, target_latent
        if self.config.memory.vae_offload:
            _move_module(pipe.vae, "cpu")
            release_cuda_memory()
        vae_after = cuda_memory_snapshot(self.device)
        self._offload_report["vae"] = self._offload_delta(vae_before, vae_after)

        self.schedule = build_flowmorph_schedule(
            pipe.scheduler,
            scheduler_points=self.config.flowmorph.scheduler_points,
            packed_latents=self.source_latent,
            device=self.device,
        )
        self._write_schedule()
        self.manifest["latent_contract"] = {
            "packed_shape": list(self.source_latent.shape),
            "image_ids_shape": list(self.image_ids.shape),
            "packed_feature_width": int(self.source_latent.shape[-1]),
            "image_token_count": int(self.source_latent.shape[1]),
            "vae_encoding": "posterior_mode_argmax",
            "patch_size": _jsonable(loaded_patch_size),
            "normalization": "loaded_vae_batch_norm_statistics",
        }
        self._write_manifest()

    @staticmethod
    def _offload_delta(
        before: Mapping[str, int | None],
        after: Mapping[str, int | None],
    ) -> dict[str, Any]:
        allocated_before = int(before.get("allocated_bytes") or 0)
        allocated_after = int(after.get("allocated_bytes") or 0)
        reserved_before = int(before.get("reserved_bytes") or 0)
        reserved_after = int(after.get("reserved_bytes") or 0)
        return {
            "before": dict(before),
            "after": dict(after),
            "allocated_bytes_released": max(0, allocated_before - allocated_after),
            "reserved_bytes_released": max(0, reserved_before - reserved_after),
        }

    def _write_schedule(self) -> None:
        assert self.schedule is not None
        start = get_start_state_metadata(
            self.schedule, self.config.flowmorph.start_timestep_index
        )
        chain = get_render_chain(
            self.schedule, self.config.flowmorph.render_indices
        )
        payload = {
            "scheduler_points": self.schedule.num_inference_steps,
            "image_seq_len": self.schedule.image_seq_len,
            "empirical_mu": self.schedule.mu,
            "used_klein_custom_sigmas": self.schedule.used_klein_custom_sigmas,
            "scheduler_configuration": self.schedule.scheduler_configuration,
            "timesteps": self.schedule.timesteps.detach().cpu().float().tolist(),
            "sigmas": self.schedule.sigmas.detach().cpu().float().tolist(),
            "start_state": start.to_dict(),
            "render_indices": list(self.config.flowmorph.render_indices),
            "render_chain": [step.to_dict() for step in chain],
            "euler_update": "x_next = x_current + (sigma_next - sigma_current) * velocity",
            "attention_modification": "none",
            "attention_backend": self.config.model.attention_backend.value,
        }
        _write_json(self.run_directory / "schedule.json", payload)
        _write_json(self.run_directory / "attention_and_schedule.json", payload)

    def _load_and_test_lora(self) -> None:
        assert self.pipeline is not None and self.device is not None
        assert self.conditioning_cache is not None
        assert self.source_latent is not None and self.image_ids is not None
        if self.config.lora.source is None:
            self.lora_report = {
                "status": "not_configured",
                "adapter_optional": True,
                "fit_scale": None,
                "render_scale": None,
                "numerical_smoke_test": "not_applicable",
            }
            self.manifest["lora_status"] = "not_configured"
            _write_json(self.run_directory / "lora_report.json", self.lora_report)
            self._write_manifest()
            return

        token = self.authentication.token if self.authentication is not None else None
        self.lora_load_report = load_flux2_lora(
            self.pipeline,
            self.config.lora.source,
            adapter_name=self.config.lora.adapter_name,
            scale=self.config.lora.fit_scale,
            token=token,
            cache_dir=self.config.paths.hf_cache,
            revision=self.config.lora.revision,
            subfolder=self.config.lora.subfolder,
            weight_name=self.config.lora.weight_name,
            require_base_9b_provenance=self.config.lora.require_base_9b_compatibility,
            allow_distilled_9b=self.config.lora.allow_distilled_9b,
        )
        recorded_lora = self.manifest.get("lora")
        if isinstance(recorded_lora, Mapping):
            recorded_sha = recorded_lora.get("sha256")
            loaded_sha = self.lora_load_report.source.sha256
            if recorded_sha != loaded_sha:
                raise PipelineError(
                    "resolved LoRA checksum differs from the existing run; refusing to "
                    "overwrite adapter provenance before checkpoint validation"
                )
        # Adapter loading happens while weights are on CPU. Move the single
        # transformer before the mandatory numerical velocity comparison.
        _move_module(self.pipeline.transformer, self.device)
        model = FlowMorphFlux2Model(self.pipeline, freeze=True)
        state = self.source_latent.to(self.device, dtype=torch.float32)
        ids = self.image_ids.to(self.device)
        conditional = self.conditioning_cache.source.to(self.device)
        unconditional = self.conditioning_cache.unconditional.to(self.device)
        timestep = self.schedule.timesteps[self.config.flowmorph.start_timestep_index].to(
            self.device
        ) if self.schedule is not None else torch.tensor(0.0, device=self.device)

        disable = getattr(self.pipeline, "disable_lora", None)
        enable = getattr(self.pipeline, "enable_lora", None)
        setter = getattr(self.pipeline, "set_adapters", None)
        if not all(callable(method) for method in (disable, enable, setter)):
            raise PipelineError("pipeline lacks native LoRA enable/disable/scale APIs")
        with torch.inference_mode():
            disable()
            baseline = model.predict_cfg_velocity(
                state,
                timestep,
                conditional,
                unconditional,
                ids,
                guidance_scale=self.config.guidance.scale,
                cfg_enabled=self.config.guidance.enabled,
                cfg_execution=self.config.guidance.execution.value,
            )
            enable()
            setter(
                self.config.lora.adapter_name,
                adapter_weights=float(self.config.lora.fit_scale),
            )
            adapted = model.predict_cfg_velocity(
                state,
                timestep,
                conditional,
                unconditional,
                ids,
                guidance_scale=self.config.guidance.scale,
                cfg_enabled=self.config.guidance.enabled,
                cfg_execution=self.config.guidance.execution.value,
            )
        numerical = compare_lora_velocities(baseline, adapted)
        activation = verify_active_adapter(
            self.pipeline, self.config.lora.adapter_name, strict=True
        )
        if not numerical.changed:
            raise PipelineError(
                "LoRA is registered but did not change the deterministic velocity smoke test"
            )
        self.lora_report = {
            "status": "verified",
            "load": self.lora_load_report.as_dict(),
            "activation": activation.as_dict(),
            "numerical_smoke_test": numerical.as_dict(),
            "fit_scale": self.config.lora.fit_scale,
            "render_scale": self.config.lora.render_scale,
            "same_fit_and_render_scale": (
                self.config.lora.fit_scale == self.config.lora.render_scale
            ),
            "distilled_9b_override": self.config.lora.allow_distilled_9b,
            "fused": False,
            "included_in_output_archive": False,
        }
        self.manifest["lora_status"] = "verified"
        self.manifest["lora"] = {
            "repo_id": self.lora_load_report.source.repo_id,
            "requested_revision": self.lora_load_report.source.requested_revision,
            "resolved_revision": self.lora_load_report.source.resolved_revision,
            "weight_name": self.lora_load_report.source.weight_name,
            "sha256": self.lora_load_report.source.sha256,
            "fit_scale": self.config.lora.fit_scale,
            "render_scale": self.config.lora.render_scale,
        }
        _write_json(self.run_directory / "lora_report.json", self.lora_report)
        self._write_manifest()
        del state, ids, conditional, unconditional, baseline, adapted
        release_cuda_memory()

    def _configure_transformer(self) -> None:
        assert self.pipeline is not None and self.device is not None
        self.model = FlowMorphFlux2Model(self.pipeline, freeze=True)
        if self.config.model.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        self.model.set_attention_backend(self.config.model.attention_backend.value)
        _move_module(self.pipeline.transformer, self.device)
        self.model.freeze()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise PipelineError("one or more transformer/LoRA parameters remain trainable")
        self.model_report["gradient_checkpointing_enabled"] = bool(
            self.config.model.gradient_checkpointing
        )
        self.model_report["attention_backend_configured"] = (
            self.config.model.attention_backend.value
        )
        self.model_report["transformer_execution_device"] = str(
            _module_device(self.pipeline.transformer)
        )
        _write_json(self.run_directory / "model_report.json", self.model_report)

    def _bound_predictor(self) -> _BoundCFGVelocityPredictor:
        _, model, cache, _, _, image_ids, _ = self._require_prepared_values()
        assert self.device is not None
        _move_module(model.transformer, self.device)
        model.freeze()
        return _BoundCFGVelocityPredictor(
            model,
            cache.unconditional.to(self.device),
            image_ids.to(self.device),
            guidance_scale=self.config.guidance.scale,
            cfg_enabled=self.config.guidance.enabled,
            cfg_execution=self.config.guidance.execution.value,
        )

    def run_production_backward_probe(self) -> BackwardProbeReport:
        """Run and persist the mandatory real 512x512 differentiable probe."""

        if not self._prepared:
            self.prepare()
        if self._session_backward_probe_report is not None:
            return self._session_backward_probe_report

        # Retain previous-session evidence, but never use it to skip the
        # current-session input-Jacobian and VRAM proof.
        report_path = self.run_directory / "memory_report.json"
        previous_probe_history: list[dict[str, Any]] = []
        if report_path.is_file():
            try:
                previous_memory_report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                previous_memory_report = {}
            recorded_history = previous_memory_report.get("probe_history")
            if isinstance(recorded_history, list):
                previous_probe_history = [
                    dict(item) for item in recorded_history if isinstance(item, Mapping)
                ]
            else:
                legacy_probe = previous_memory_report.get("backward_probe")
                if isinstance(legacy_probe, Mapping):
                    previous_probe_history.append(
                        {
                            "recorded_at": previous_memory_report.get("recorded_at"),
                            "profile": previous_memory_report.get("profile"),
                            "backward_probe": dict(legacy_probe),
                            "legacy_record": True,
                        }
                    )
        probe_attempts: list[dict[str, Any]] = []
        retry_policy = {
            "allowed_profiles": [self.config.model.profile.value],
            "retries_per_allowed_profile": 1,
            "automatic_profile_switching": False,
            "semantic_controls_held_fixed": [
                "9B model and revision",
                (
                    f"{self.config.input.width}x{self.config.input.height} resolution"
                ),
                f"{self.config.flowmorph.scheduler_points} scheduler points",
                (
                    f"{self.config.flowmorph.optimization_steps_source}/"
                    f"{self.config.flowmorph.optimization_steps_target} source/target "
                    "optimization steps"
                ),
                "CFG enabled and configured execution",
            ],
            "reason": (
                "only the explicitly selected and validated profile is allowed; FP8 or "
                "another profile must be requested explicitly"
            ),
        }
        try:
            _, model, cache, source_latent, _, image_ids, schedule = (
                self._require_prepared_values()
            )
            assert self.device is not None
            start = get_start_state_metadata(
                schedule, self.config.flowmorph.start_timestep_index
            )

            def attempt_probe() -> tuple[BackwardProbeReport, tuple[int, ...]]:
                conditional = cache.source.to(self.device)
                unconditional = cache.unconditional.to(self.device)
                ids = image_ids.to(self.device)
                z = source_latent.to(self.device, dtype=torch.float32)

                def predict(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                    return model.predict_cfg_velocity(
                        state,
                        timestep,
                        conditional,
                        unconditional,
                        ids,
                        guidance_scale=self.config.guidance.scale,
                        cfg_enabled=self.config.guidance.enabled,
                        cfg_execution=self.config.guidance.execution.value,
                    )

                result = run_backward_probe(
                    z=z,
                    sigma_i=start.sigma_i,
                    sigma_last=start.sigma_last,
                    timestep=schedule.timesteps[
                        self.config.flowmorph.start_timestep_index
                    ].to(self.device),
                    predict_velocity=predict,
                    frozen_parameters=model.parameters(),
                )
                return result, tuple(z.shape)

            report: BackwardProbeReport | None = None
            production_shape: tuple[int, ...] | None = None
            for attempt_number in (1, 2):
                try:
                    report, production_shape = attempt_probe()
                    probe_attempts.append(
                        {
                            "attempt": attempt_number,
                            "profile": self.config.model.profile.value,
                            "status": "passed",
                            "oom_retry": attempt_number > 1,
                        }
                    )
                    break
                except BaseException as attempt_error:
                    oom = _is_cuda_out_of_memory(attempt_error)
                    probe_attempts.append(
                        {
                            "attempt": attempt_number,
                            "profile": self.config.model.profile.value,
                            "status": "out_of_memory" if oom else "failed",
                            "error": redact_secrets(
                                f"{type(attempt_error).__name__}: {attempt_error}"
                            ),
                            "memory": cuda_memory_snapshot(self.device),
                        }
                    )
                    if not oom or attempt_number == 2:
                        raise
                    # The retry must not retain the failed autograd graph via
                    # the exception traceback.
                    attempt_error.__traceback__ = None
                    release_cuda_memory()
                    self.memory_report = {
                        "profile": self.config.model.profile.value,
                        "backward_probe": {"passed": False, "status": "retrying_after_oom"},
                        "probe_attempts": probe_attempts,
                        "probe_history": previous_probe_history,
                        "oom_retry_policy": retry_policy,
                        "support_claim": "not_yet_established",
                    }
                    _write_json(
                        self.run_directory / "memory_report.json", self.memory_report
                    )
                    self._log(
                        "production backward probe OOM; released probe tensors and "
                        "retrying the explicitly selected profile once"
                    )

            if report is None or production_shape is None:
                raise PipelineError("production backward probe did not return a result")
            memory_after = cuda_memory_snapshot(self.device)
            total = memory_after.get("total_bytes")
            reserved_peak = report.peak_reserved_vram_bytes
            headroom = int(total - reserved_peak) if total is not None else None
            recorded_at = datetime.now(timezone.utc).isoformat()
            current_probe_record = {
                "recorded_at": recorded_at,
                "profile": self.config.model.profile.value,
                "backward_probe": report.as_dict(),
                "probe_attempts": probe_attempts,
            }
            self.memory_report = {
                "recorded_at": recorded_at,
                "profile": self.config.model.profile.value,
                "production_shape": list(production_shape),
                "image_resolution": [
                    self.config.input.width,
                    self.config.input.height,
                ],
                "cfg_execution": self.config.guidance.execution.value,
                "backward_probe": report.as_dict(),
                "probe_attempts": probe_attempts,
                "probe_history": [*previous_probe_history, current_probe_record],
                "oom_retry_policy": retry_policy,
                "phase_offload": self._offload_report,
                "memory_after_probe": memory_after,
                "estimated_peak_headroom_bytes": headroom,
                "support_claim": "probe_passed_on_this_runtime",
            }
            _write_json(self.run_directory / "memory_report.json", self.memory_report)
            self.manifest["backward_probe_status"] = "passed"
            self.manifest["last_backward_probe_at"] = recorded_at
            self.model_report["production_backward_probe"] = "passed"
            self.model_report["production_backward_probe_attempts"] = len(
                probe_attempts
            )
            _write_json(self.run_directory / "model_report.json", self.model_report)
            self._write_manifest()
            if not self._at_least(RunPhase.BACKWARD_PREFLIGHT_PASSED):
                self._advance(RunPhase.BACKWARD_PREFLIGHT_PASSED)
            self._session_backward_probe_report = report
            release_cuda_memory()
            return report
        except BaseException as error:
            snapshot = cuda_memory_snapshot(self.device or "cpu")
            recorded_at = datetime.now(timezone.utc).isoformat()
            failed_probe_record = {
                "recorded_at": recorded_at,
                "profile": self.config.model.profile.value,
                "backward_probe": {
                    "passed": False,
                    "error": redact_secrets(str(error)),
                },
                "probe_attempts": probe_attempts,
            }
            self.memory_report = {
                "recorded_at": recorded_at,
                "profile": self.config.model.profile.value,
                "backward_probe": {
                    "passed": False,
                    "error": redact_secrets(str(error)),
                },
                "probe_attempts": probe_attempts,
                "probe_history": [*previous_probe_history, failed_probe_record],
                "oom_retry_policy": retry_policy,
                "memory_at_failure": snapshot,
                "support_claim": "not_supported_on_this_runtime",
            }
            _write_json(self.run_directory / "memory_report.json", self.memory_report)
            self.manifest["backward_probe_status"] = "failed"
            self._record_failure(error, operation="production_backward_probe")
            error.__traceback__ = None
            release_cuda_memory()
            try:
                self.archive_report = create_run_archive(
                    self.run_directory, self.run_id
                )
            except Exception as packaging_error:
                self._log(
                    "diagnostic archive creation also failed: "
                    + redact_secrets(str(packaging_error))
                )
            raise

    def run(self, *, resume: bool = False) -> FlowMorphRunResult:
        """Fit both endpoints, render, evaluate, and publish one compact run."""

        if self.config.run_mode is RunMode.DIAGNOSTIC:
            raise PipelineError(
                "diagnostic/unsupported_low_vram runs stop after the backward preflight; "
                "use validate-colab rather than the fitting command"
            )
        if resume:
            self._restore_failed_phase_for_resume()
        try:
            if not self._prepared:
                self.prepare(resume=resume)
            if self._session_backward_probe_report is None:
                if not self.config.memory.run_production_backward_probe:
                    raise PipelineError(
                        "production backward probe is mandatory for this implementation"
                    )
                self.run_production_backward_probe()

            self._set_lora_scale(self.config.lora.fit_scale)
            self.source_endpoint, source_rows = self._fit_endpoint(
                "source", resume=resume
            )
            if not self._at_least(RunPhase.SOURCE_CHECKPOINTED):
                self._advance(RunPhase.SOURCE_CHECKPOINTED)
            release_cuda_memory()

            self.target_endpoint, target_rows = self._fit_endpoint(
                "target", resume=resume
            )
            if not self._at_least(RunPhase.TARGET_CHECKPOINTED):
                self._advance(RunPhase.TARGET_CHECKPOINTED)
            release_cuda_memory()

            self._set_lora_scale(self.config.lora.render_scale)
            latent_frames = self._render_latents()
            source_conditioning_frames = None
            if (
                self.config.flowmorph.render_conditioning_mode
                is RenderConditioningMode.INTERPOLATED_EMBEDDINGS
            ):
                source_conditioning_frames = self._render_latents(
                    conditioning_mode=RenderConditioningMode.SOURCE
                )
            raw_images, display_images = self._decode_and_save_frames(
                latent_frames,
                source_conditioning_frames=source_conditioning_frames,
            )
            if not self._at_least(RunPhase.FRAMES_RENDERED):
                self._advance(RunPhase.FRAMES_RENDERED)

            self._evaluate_and_visualize(
                latent_frames,
                raw_images,
                display_images,
                source_rows,
                target_rows,
            )
            self.manifest.update(
                {
                    "source_completed_steps": self.config.flowmorph.optimization_steps_source,
                    "target_completed_steps": self.config.flowmorph.optimization_steps_target,
                    "raw_frame_count": len(raw_images),
                    "display_frame_count": len(display_images),
                    "metrics_status": "complete",
                    "reference_reproduction_claimed": False,
                    "result_interpretation": (
                        "local single-pair reproduction run; not a reproduction of paper tables"
                    ),
                }
            )
            self._write_manifest()
            if not self._at_least(RunPhase.METRICS_COMPLETE):
                self._advance(RunPhase.METRICS_COMPLETE)

            self.acceptance_report = require_completed_run(
                self.run_directory,
                expected_frames=self.config.flowmorph.frame_count,
                expected_source_steps=self.config.flowmorph.optimization_steps_source,
                expected_target_steps=self.config.flowmorph.optimization_steps_target,
                expected_model_id=self.config.model.id,
                require_lora=self.config.lora.source is not None,
                require_conditioning_comparison=(
                    self.config.flowmorph.render_conditioning_mode
                    is RenderConditioningMode.INTERPOLATED_EMBEDDINGS
                ),
            )
            # Validate a preliminary archive before labeling the phase. The
            # final rebuild below includes the updated phase and checksum.
            if self.config.output.create_zip:
                self.archive_report = create_run_archive(
                    self.run_directory, self.run_id
                )
                if not self._at_least(RunPhase.ARCHIVE_VALIDATED):
                    self._advance(RunPhase.ARCHIVE_VALIDATED)
                self.archive_report = create_run_archive(
                    self.run_directory, self.run_id
                )
                self._maybe_copy_archive_to_drive()
            else:
                self.manifest["archive_status"] = "disabled_by_explicit_configuration"
                self._write_manifest()
            return FlowMorphRunResult(
                run_id=self.run_id,
                run_directory=self.run_directory,
                phase=self.phase.value,
                metrics_path=self.run_directory / "metrics.json",
                archive=self.archive_report,
                acceptance=self.acceptance_report,
            )
        except BaseException as error:
            if self.phase is not RunPhase.FAILED:
                self._record_failure(error, operation="run_resume" if resume else "run")
            raise

    def resume(self) -> FlowMorphRunResult:
        """Explicitly resume compatible source/target checkpoint state."""

        return self.run(resume=True)

    def _set_lora_scale(self, scale: float) -> None:
        if self.config.lora.source is None:
            return
        assert self.pipeline is not None
        setter = getattr(self.pipeline, "set_adapters", None)
        if not callable(setter):
            raise PipelineError("pipeline cannot set the active LoRA scale")
        setter(self.config.lora.adapter_name, adapter_weights=float(scale))
        verify_active_adapter(self.pipeline, self.config.lora.adapter_name, strict=True)

    def _endpoint_metadata(self, label: str) -> dict[str, Any]:
        _, _, cache, source_latent, target_latent, _, schedule = (
            self._require_prepared_values()
        )
        preprocessed = (
            self.source_preprocessed if label == "source" else self.target_preprocessed
        )
        latent = source_latent if label == "source" else target_latent
        conditioning = cache.source if label == "source" else cache.target
        if preprocessed is None or preprocessed.output_path is None:
            raise PipelineError("preprocessed endpoint metadata is unavailable")
        lora_source = None
        lora_revision = None
        lora_sha = None
        if self.lora_load_report is not None:
            lora_source = self.lora_load_report.source.repo_id or "local_safetensors"
            lora_revision = self.lora_load_report.source.resolved_revision
            lora_sha = self.lora_load_report.source.sha256
        start = get_start_state_metadata(
            schedule, self.config.flowmorph.start_timestep_index
        )
        return {
            "schema_version": 1,
            "endpoint": label,
            "model_id": self.config.model.id,
            "model_revision": self.config.model.revision,
            "lora_source": lora_source,
            "lora_revision": lora_revision,
            "lora_file_sha256": lora_sha,
            "lora_scale": self.config.lora.fit_scale if lora_source else None,
            "prompt_checksum": conditioning.prompt_sha256,
            "source_image_checksum": preprocessed.original_sha256,
            "processed_image_checksum": sha256_file(preprocessed.output_path),
            "preprocessing_hash": preprocessed.preprocessing_sha256,
            "resize_mode": preprocessed.resize_mode.value,
            "scheduler_configuration": {
                "config": schedule.scheduler_configuration,
                "timesteps": schedule.timesteps.detach().cpu().float().tolist(),
                "sigmas": schedule.sigmas.detach().cpu().float().tolist(),
                "mu": schedule.mu,
                "image_seq_len": schedule.image_seq_len,
            },
            "start_timestep_index": self.config.flowmorph.start_timestep_index,
            "timestep_i": start.timestep_i,
            "sigma_i": start.sigma_i,
            "sigma_last": start.sigma_last,
            "latent_shape": list(latent.shape),
            "optimizer_configuration": {
                "name": self.config.flowmorph.optimizer.value,
                "pred_learning_rate": self.config.flowmorph.pred_learning_rate,
                "u_learning_rate": self.config.flowmorph.u_learning_rate,
                "weight_decay": (
                    0.01
                    if self.config.flowmorph.weight_decay is None
                    else self.config.flowmorph.weight_decay
                ),
            },
            "loss_mode": self.config.flowmorph.loss_mode.value,
            "guidance_configuration": {
                "enabled": self.config.guidance.enabled,
                "scale": self.config.guidance.scale,
                "execution": self.config.guidance.execution.value,
            },
            "precision_configuration": {
                "transformer": self.config.model.transformer_compute_dtype.value,
                "endpoint_parameters": self.config.model.optimization_parameter_dtype.value,
                "quantization": self.config.model.quantization.value,
            },
            "diffusers_commit": DIFFUSERS_COMMIT,
            "flowmorph_commit": FLOWMORPH_COMMIT,
            "flux2_commit": FLUX2_COMMIT,
        }

    def _optimizer_config(self, label: str) -> EndpointOptimizerConfig:
        steps = (
            self.config.flowmorph.optimization_steps_source
            if label == "source"
            else self.config.flowmorph.optimization_steps_target
        )
        return EndpointOptimizerConfig(
            optimization_steps=steps,
            pred_learning_rate=self.config.flowmorph.pred_learning_rate,
            u_learning_rate=self.config.flowmorph.u_learning_rate,
            weight_decay=self.config.flowmorph.weight_decay,
            loss_mode=self.config.flowmorph.loss_mode,
            checkpoint_every=self.config.flowmorph.checkpoint_every,
        )

    def _fit_endpoint(
        self, label: str, *, resume: bool
    ) -> tuple[FlowMorphEndpoint, list[dict[str, Any]]]:
        if label not in {"source", "target"}:
            raise ValueError("endpoint label must be source or target")
        _, model, cache, source_latent, target_latent, _, schedule = (
            self._require_prepared_values()
        )
        assert self.device is not None
        z_cpu = source_latent if label == "source" else target_latent
        conditioning_cpu = cache.source if label == "source" else cache.target
        checkpoint_directory = self.run_directory / "checkpoints" / label
        metadata = self._endpoint_metadata(label)
        settings = self._optimizer_config(label)
        csv_path = self.run_directory / "optimization" / f"{label}_loss.csv"

        loaded: LoadedCheckpoint | None = None
        initial_delta = None
        initial_u = None
        optimizer_state = None
        start_step = 0
        previous_rows = self._read_csv_rows(csv_path)
        if checkpoint_directory.exists():
            if not resume:
                raise PipelineError(
                    f"{label} checkpoint already exists; use explicit resume to avoid overwriting it"
                )
            loaded = load_endpoint_checkpoint(
                checkpoint_directory,
                expected_metadata=metadata,
                device="cpu",
            )
            start_step = int(loaded.metadata.get("completed_steps", 0))
            # Resume from the exact encoded target stored in the checkpoint,
            # not from a newly encoded approximation on a different runtime.
            z_cpu = loaded.tensors["z"]
            initial_delta = loaded.tensors["delta"]
            initial_u = loaded.tensors["u"]
            if start_step >= settings.optimization_steps:
                endpoint = FlowMorphEndpoint(
                    z=loaded.tensors["z"].float(),
                    delta=loaded.tensors["delta"].float(),
                    u=loaded.tensors["u"].float(),
                    sigma_i=metadata["sigma_i"],
                    sigma_last=metadata["sigma_last"],
                    timestep_i=metadata["timestep_i"],
                )
                if not previous_rows:
                    raise CheckpointError(
                        f"completed {label} checkpoint lacks optimization history"
                    )
                return endpoint, previous_rows
            descriptor = loaded.metadata.get("optimizer_state")
            if not loaded.metadata.get("optimizer_state_saved") or not isinstance(
                descriptor, Mapping
            ):
                raise CheckpointError(
                    f"incomplete {label} checkpoint has no optimizer moments for exact resume"
                )
            optimizer_state = unflatten_optimizer_state(loaded.tensors, descriptor)

        z = z_cpu.to(self.device, dtype=torch.float32)
        conditioning = conditioning_cpu.to(self.device)
        predictor = self._bound_predictor()
        step_rows: list[dict[str, Any]] = []

        def diagnostics_callback(diagnostics: OptimizationStepDiagnostics) -> None:
            step_rows.append(diagnostics.to_dict())
            self._write_history_snapshot(csv_path, previous_rows + step_rows)

        def checkpoint_callback(
            step: int,
            endpoint: FlowMorphEndpoint,
            optimizer: torch.optim.Optimizer,
            diagnostics: OptimizationStepDiagnostics,
        ) -> None:
            checkpoint_metadata = dict(metadata)
            checkpoint_metadata["completed_steps"] = step
            checkpoint_metadata["optimization_steps_required"] = settings.optimization_steps
            save_optimizer = step < settings.optimization_steps or self.config.output.save_optimizer_states
            save_endpoint_checkpoint(
                checkpoint_directory,
                {"z": endpoint.z, "delta": endpoint.delta, "u": endpoint.u},
                checkpoint_metadata,
                optimizer=optimizer if save_optimizer else None,
            )
            self.manifest[f"{label}_completed_steps"] = step
            self._write_manifest()

        start = get_start_state_metadata(
            schedule, self.config.flowmorph.start_timestep_index
        )
        result: EndpointOptimizationResult = optimize_endpoint(
            z,
            sigma_i=start.sigma_i,
            sigma_last=start.sigma_last,
            timestep_i=schedule.timesteps[
                self.config.flowmorph.start_timestep_index
            ].to(self.device),
            predictor=predictor,
            conditioning=conditioning,
            config=settings,
            initial_delta=initial_delta,
            initial_u=initial_u,
            optimizer_state_dict=optimizer_state,
            start_step=start_step,
            predictor_parameters=model.parameters(),
            checkpoint_callback=checkpoint_callback,
            diagnostics_callback=diagnostics_callback,
        )
        all_rows = self._merge_history(
            previous_rows,
            [item.to_dict() for item in result.diagnostics],
        )
        write_csv(csv_path, all_rows)
        save_loss_plot(
            all_rows,
            self.run_directory / "optimization" / f"{label}_loss.png",
            title=f"{label.capitalize()} endpoint FlowMorph loss",
        )
        self.manifest[f"{label}_completed_steps"] = result.completed_steps
        self._write_manifest()
        endpoint_cpu = result.endpoint.to(device="cpu", dtype=torch.float32)
        del z, conditioning, predictor, result
        return endpoint_cpu, all_rows

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        numeric_fields = {
            "step",
            "total_loss",
            "reconstruction_loss",
            "regularization_loss",
            "pred_gradient_norm",
            "u_gradient_norm",
            "pred_parameter_norm",
            "u_parameter_norm",
            "delta_norm",
            "peak_allocated_vram_bytes",
            "peak_reserved_vram_bytes",
            "elapsed_seconds",
        }
        for row in rows:
            for field in numeric_fields.intersection(row):
                raw = row[field]
                if raw in {"", "None", None}:
                    row[field] = None
                elif field == "step":
                    row[field] = int(float(raw))
                else:
                    row[field] = float(raw)
        return rows

    @staticmethod
    def _merge_history(
        previous: Sequence[dict[str, Any]], new: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for row in (*previous, *new):
            by_step[int(row["step"])] = dict(row)
        return [by_step[step] for step in sorted(by_step)]

    def _write_history_snapshot(
        self, path: Path, rows: Sequence[dict[str, Any]]
    ) -> None:
        merged = self._merge_history((), rows)
        if merged:
            write_csv(path, merged)

    def _render_latents(
        self,
        *,
        conditioning_mode: RenderConditioningMode | None = None,
    ) -> tuple[RenderedLatentFrame, ...]:
        if self.source_endpoint is None or self.target_endpoint is None:
            raise PipelineError("both endpoint states are required before rendering")
        _, _, cache, _, _, _, schedule = self._require_prepared_values()
        assert self.device is not None
        predictor = self._bound_predictor()
        source = self.source_endpoint.to(self.device, dtype=torch.float32)
        target = self.target_endpoint.to(self.device, dtype=torch.float32)
        frames = render_morph(
            source,
            target,
            schedule=schedule,
            predictor=predictor,
            source_conditioning=cache.source.to(self.device),
            target_conditioning=cache.target.to(self.device),
            bridge_conditioning=(
                cache.bridge.to(self.device) if cache.bridge is not None else None
            ),
            frame_count=self.config.flowmorph.frame_count,
            render_indices=self.config.flowmorph.render_indices,
            conditioning_mode=(
                conditioning_mode
                or self.config.flowmorph.render_conditioning_mode
            ),
            conditioning_interpolator=interpolate_conditioning,
            output_dtype=torch.float32,
            use_inference_mode=True,
        )
        return tuple(
            RenderedLatentFrame(
                index=frame.index,
                alpha=frame.alpha,
                start_state=frame.start_state.to("cpu"),
                final_latent=frame.final_latent.to("cpu"),
                conditioning_mode=frame.conditioning_mode,
            )
            for frame in frames
        )

    def _decode_one(self, tokens: torch.Tensor) -> Image.Image:
        assert self.pipeline is not None and self.image_ids is not None and self.device is not None
        vae_dtype = _module_dtype(
            self.pipeline.vae, _torch_dtype(self.config.model.transformer_compute_dtype)
        )
        result = decode_packed_latent(
            tokens.to(self.device, dtype=vae_dtype),
            self.image_ids.to(self.device),
            self.pipeline.vae,
            image_processor=self.pipeline.image_processor,
            output_type="pil",
            postprocess=True,
        )
        if isinstance(result, (list, tuple)) and result:
            result = result[0]
        if not isinstance(result, Image.Image):
            raise PipelineError("VAE decode did not return a PIL image")
        return result.convert("RGB")

    def _decode_and_save_frames(
        self,
        latent_frames: Sequence[RenderedLatentFrame],
        *,
        source_conditioning_frames: Sequence[RenderedLatentFrame] | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        assert self.pipeline is not None and self.device is not None
        assert self.source_preprocessed is not None and self.target_preprocessed is not None
        # Rendering is complete. Offload the 9B transformer before returning
        # the VAE to GPU so the 40GB profile does not retain both at once.
        _move_module(self.pipeline.transformer, "cpu")
        release_cuda_memory()
        _move_module(self.pipeline.vae, self.device)
        raw_images: list[Image.Image] = []
        source_conditioning_images: list[Image.Image] = []
        with torch.inference_mode():
            for frame in latent_frames:
                raw_images.append(self._decode_one(frame.final_latent))
            if source_conditioning_frames is not None:
                for frame in source_conditioning_frames:
                    source_conditioning_images.append(
                        self._decode_one(frame.final_latent)
                    )
        if len(raw_images) != self.config.flowmorph.frame_count:
            raise PipelineError("decoded raw frame count does not match configuration")
        raw_directory = self.run_directory / "raw_frames"
        raw_directory.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(raw_images):
            image.save(raw_directory / f"frame_{index:03d}.png")

        display_images = [
            self.source_preprocessed.image.copy(),
            *[image.copy() for image in raw_images[1:-1]],
            self.target_preprocessed.image.copy(),
        ]
        display_directory = self.run_directory / "display_frames"
        display_directory.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(display_images):
            image.convert("RGB").save(display_directory / f"frame_{index:03d}.png")

        if source_conditioning_frames is not None:
            if len(source_conditioning_images) != len(raw_images):
                raise PipelineError(
                    "source-conditioning comparison frame count does not match the "
                    "interpolated-embedding render"
                )
            comparison_directory = self.run_directory / "conditioning_comparison"
            baseline_directory = comparison_directory / "source_conditioning_frames"
            baseline_directory.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(source_conditioning_images):
                image.save(baseline_directory / f"frame_{index:03d}.png")
            paired_images: list[Image.Image] = []
            paired_labels: list[str] = []
            for index, (primary, baseline) in enumerate(
                zip(raw_images, source_conditioning_images, strict=True)
            ):
                paired_images.extend((primary, baseline))
                paired_labels.extend(
                    (
                        f"{index:03d} interpolated embeddings",
                        f"{index:03d} source conditioning",
                    )
                )
            make_contact_sheet(
                paired_images,
                comparison_directory / "interpolated_vs_source.png",
                columns=4,
                labels=paired_labels,
            )
            comparison_report = {
                "status": "complete",
                "experimental_mode": "interpolated_embeddings",
                "baseline_mode": "source",
                "frame_count_per_mode": len(raw_images),
                "primary_frames": "raw_frames",
                "baseline_frames": "conditioning_comparison/source_conditioning_frames",
                "paired_contact_sheet": (
                    "conditioning_comparison/interpolated_vs_source.png"
                ),
            }
            _write_json(
                comparison_directory / "comparison.json", comparison_report
            )
            self.manifest["conditioning_comparison"] = comparison_report
            self._write_manifest()
        _move_module(self.pipeline.vae, "cpu")
        release_cuda_memory()
        return raw_images, display_images

    def _load_lpips_model(self) -> torch.nn.Module:
        try:
            import lpips
        except ImportError as error:
            raise PipelineError("lpips==0.1.4 is required for the mandated metrics") from error
        # LPIPS is explicit rather than hidden inside metric helpers. It may
        # download its published AlexNet calibration weights on first use.
        model = lpips.LPIPS(net="alex")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.to("cpu")

    def _evaluate_and_visualize(
        self,
        latent_frames: Sequence[RenderedLatentFrame],
        raw_images: Sequence[Image.Image],
        display_images: Sequence[Image.Image],
        source_rows: Sequence[dict[str, Any]],
        target_rows: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        assert self.source_preprocessed is not None and self.target_preprocessed is not None
        assert self.source_latent is not None and self.target_latent is not None
        lpips_model = self._load_lpips_model()
        endpoint_directory = self.run_directory / "endpoint_reconstruction"
        endpoint_directory.mkdir(parents=True, exist_ok=True)
        source_reference = self.source_preprocessed.image.convert("RGB")
        target_reference = self.target_preprocessed.image.convert("RGB")
        source_generated = raw_images[0].convert("RGB")
        target_generated = raw_images[-1].convert("RGB")
        saved = {
            "source_reference": source_reference,
            "source_generated": source_generated,
            "target_reference": target_reference,
            "target_generated": target_generated,
        }
        for name, image in saved.items():
            image.save(endpoint_directory / f"{name}.png")
        if self.config.output.save_difference_images:
            difference_image(source_reference, source_generated).save(
                endpoint_directory / "source_difference.png"
            )
            difference_image(target_reference, target_generated).save(
                endpoint_directory / "target_difference.png"
            )
        source_metrics = endpoint_reconstruction_metrics(
            source_reference,
            source_generated,
            self.source_latent,
            latent_frames[0].final_latent,
            lpips_model=lpips_model,
            lpips_device="cpu",
        )
        target_metrics = endpoint_reconstruction_metrics(
            target_reference,
            target_generated,
            self.target_latent,
            latent_frames[-1].final_latent,
            lpips_model=lpips_model,
            lpips_device="cpu",
        )
        transition_summary, transition_rows = transition_metrics(
            raw_images,
            lpips_model=lpips_model,
            lpips_device="cpu",
        )
        write_csv(
            self.run_directory / "optimization" / "transition_metrics.csv",
            transition_rows,
        )
        metrics = {
            "interpretation": (
                "local single-pair metrics; not a reproduction of FlowMorph paper tables"
            ),
            "metric_implementations": {
                "psnr": "project implementation, RGB [0,1]",
                "ssim": "skimage.metrics.structural_similarity, channel_axis=-1",
                "lpips": "lpips==0.1.4, AlexNet",
                "latent": "packed normalized FLUX.2 token space",
            },
            "endpoint_reconstruction": {
                "source": source_metrics,
                "target": target_metrics,
            },
            "transition": transition_summary,
            "optimization": {
                "source": summarize_optimization(source_rows),
                "target": summarize_optimization(target_rows),
            },
            "runtime_recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        write_metrics(self.run_directory / "metrics.json", metrics)

        previews = self.run_directory / "previews"
        if self.config.output.save_contact_sheet:
            make_contact_sheet(
                raw_images,
                previews / "raw_contact_sheet.png",
                columns=5,
                labels=[f"alpha={frame.alpha:.3f}" for frame in latent_frames],
            )
            make_contact_sheet(
                display_images,
                previews / "display_contact_sheet.png",
                columns=5,
                labels=[f"frame {index:03d}" for index in range(len(display_images))],
            )
            save_endpoint_comparison(
                source_reference,
                source_generated,
                target_reference,
                target_generated,
                previews / "endpoint_comparison.png",
            )
        if self.config.output.save_webp:
            save_webp(
                display_images,
                previews / "preview.webp",
                fps=self.config.output.fps,
            )
        if self.config.output.save_gif:
            save_gif(
                display_images,
                previews / "preview.gif",
                fps=self.config.output.fps,
            )
        if self.config.output.save_mp4:
            save_mp4(
                display_images,
                previews / "morph.mp4",
                fps=self.config.output.fps,
            )
        return metrics

    def _maybe_copy_archive_to_drive(self) -> None:
        if not self.config.output.copy_archive_to_drive:
            return
        if self.archive_report is None:
            raise PipelineError("cannot copy an archive before it exists")
        if self.config.paths.drive_root is None:
            raise PipelineError("copy_archive_to_drive requires paths.drive_root")
        destination_directory = self.config.paths.drive_root / "artifacts"
        destination_directory.mkdir(parents=True, exist_ok=True)
        copied_archive = destination_directory / self.archive_report.path.name
        shutil.copy2(
            self.archive_report.path,
            copied_archive,
        )
        copied_sha256 = sha256_file(copied_archive)
        if copied_sha256 != self.archive_report.sha256:
            raise PipelineError(
                "Drive archive checksum does not match the locally validated archive"
            )
        manifest_source = self.run_directory / "run_manifest.json"
        manifest_destination = self.config.paths.drive_root / "manifests" / (
            f"{self.run_id}.json"
        )
        manifest_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_source, manifest_destination)
        if sha256_file(manifest_source) != sha256_file(manifest_destination):
            raise PipelineError("Drive run-manifest checksum verification failed")


__all__ = [
    "FlowMorphRunResult",
    "FlowMorphRunner",
    "PipelineError",
]
