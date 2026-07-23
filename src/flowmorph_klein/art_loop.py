"""Validated prompt plans and helpers for sequential FlowMorph art loops."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .colab_io import stage_drive_inputs
from .config import MIRROR_MODEL_ID, MIRROR_MODEL_REVISION


class ArtLoopError(ValueError):
    """Raised when an art-loop specification or generated artifact is invalid."""


class ArtLoopModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ArtProject(ArtLoopModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    description: str = ""
    loop: bool = True


class ContinuityConfig(ArtLoopModel):
    enabled: bool = True
    reference_blend: float = Field(default=0.15, gt=0.0, le=1.0)
    blur_radius: float = Field(default=12.0, ge=0.0, le=64.0)
    grain_strength: float = Field(default=0.0, ge=0.0, le=0.25)
    background_rgb: tuple[int, int, int] = (127, 127, 127)

    @field_validator("background_rgb")
    @classmethod
    def validate_rgb(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("background_rgb channels must lie in [0, 255]")
        return value


class MainframeGenerationConfig(ArtLoopModel):
    model_id: str = MIRROR_MODEL_ID
    model_revision: str = MIRROR_MODEL_REVISION
    width: int = Field(default=512, ge=256, le=2048, multiple_of=16)
    height: int = Field(default=512, ge=256, le=2048, multiple_of=16)
    num_inference_steps: int = Field(default=28, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0.0, le=20.0)
    seed: int = Field(default=4200, ge=0)
    prompt_prefix: str = ""
    continuity: ContinuityConfig = Field(default_factory=ContinuityConfig)

    @model_validator(mode="after")
    def require_pinned_public_mirror(self) -> "MainframeGenerationConfig":
        if self.model_id != MIRROR_MODEL_ID or self.model_revision != MIRROR_MODEL_REVISION:
            raise ValueError("art-loop mainframes require the pinned public Runware FLUX.2 Klein Base 9B mirror")
        if (self.width, self.height) != (512, 512):
            raise ValueError("the current full-shape FlowMorph art loop requires 512x512 mainframes")
        return self


class ArtLoraConfig(ArtLoopModel):
    source: str = "MaxNoichl/RIJKSOIL_FLUX2_KLEIN9B_lora_01_000001650"
    revision: str = "042a31d6cd09bf55195f820461fac60b1a358409"
    weight_name: str = "RIJKSOIL_FLUX2_KLEIN9B_lora_01_000001650.safetensors"
    adapter_name: str = "rijks_oil"
    scale: float = Field(default=1.0, gt=0.0, le=4.0)
    allow_distilled_9b_provenance: bool = True


class FlowMorphLoopConfig(ArtLoopModel):
    frame_count: int = 20
    render_conditioning_mode: str = "prompt_schedule"
    fps: float = Field(default=12.0, gt=0.0, le=60.0)

    @model_validator(mode="after")
    def require_supported_schedule(self) -> "FlowMorphLoopConfig":
        if self.frame_count != 20:
            raise ValueError("the current full-shape FlowMorph contract requires exactly 20 frames")
        if self.render_conditioning_mode != "prompt_schedule":
            raise ValueError("art-loop transitions require render_conditioning_mode='prompt_schedule'")
        return self


class Mainframe(ArtLoopModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    prompt: str = Field(min_length=1)
    seed_offset: int = Field(default=0, ge=0)


class Transition(ArtLoopModel):
    from_id: str = Field(alias="from", min_length=1)
    to_id: str = Field(alias="to", min_length=1)
    bridge_prompts: tuple[str, ...]

    @field_validator("bridge_prompts")
    @classmethod
    def reject_blank_prompts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not prompt.strip() for prompt in value):
            raise ValueError("bridge_prompts cannot contain blank strings")
        return value


class ArtLoopSpec(ArtLoopModel):
    schema_version: int = 1
    project: ArtProject
    generation: MainframeGenerationConfig = Field(default_factory=MainframeGenerationConfig)
    lora: ArtLoraConfig = Field(default_factory=ArtLoraConfig)
    flowmorph: FlowMorphLoopConfig = Field(default_factory=FlowMorphLoopConfig)
    mainframes: tuple[Mainframe, ...] = Field(min_length=2)
    transitions: tuple[Transition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sequence(self) -> "ArtLoopSpec":
        if self.schema_version != 1:
            raise ValueError("only art-loop schema_version 1 is supported")

        identifiers = [mainframe.id for mainframe in self.mainframes]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("mainframe ids must be unique")

        expected_pairs = list(zip(identifiers, identifiers[1:]))
        if self.project.loop:
            expected_pairs.append((identifiers[-1], identifiers[0]))
        actual_pairs = [(transition.from_id, transition.to_id) for transition in self.transitions]
        if actual_pairs != expected_pairs:
            raise ValueError(f"transitions must follow mainframe order exactly: {expected_pairs!r}")

        wrong_counts = [
            f"{transition.from_id}->{transition.to_id} ({len(transition.bridge_prompts)})"
            for transition in self.transitions
            if len(transition.bridge_prompts) != self.flowmorph.frame_count
        ]
        if wrong_counts:
            raise ValueError(
                "each transition needs exactly "
                f"{self.flowmorph.frame_count} bridge prompts; invalid: {', '.join(wrong_counts)}"
            )
        return self

    def mainframe_by_id(self) -> dict[str, Mainframe]:
        return {mainframe.id: mainframe for mainframe in self.mainframes}


@dataclass(frozen=True, slots=True)
class GeneratedMainframe:
    id: str
    prompt: str
    seed: int
    path: Path
    soft_reference_path: Path | None


@dataclass(frozen=True, slots=True)
class ArtPersistenceReport:
    source: Path
    destination: Path
    project: str
    label: str
    sequence: int
    timestamp_utc: str
    file_count: int
    total_bytes: int


def load_art_loop_spec(path: str | Path) -> ArtLoopSpec:
    """Load and fully validate a JSON art-loop prompt plan."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtLoopError(f"art-loop specification does not exist: {source}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ArtLoopError(f"cannot read art-loop specification {source}: {error}") from error
    try:
        return ArtLoopSpec.model_validate(payload)
    except ValidationError as error:
        raise ArtLoopError(f"invalid art-loop specification {source}: {error}") from error


def make_soft_reference(
    previous: Image.Image,
    *,
    reference_blend: float = 0.15,
    blur_radius: float = 12.0,
    grain_strength: float = 0.0,
    grain_seed: int | None = None,
    background_rgb: tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Reduce a previous frame to a faint blurred reference with optional grain.

    ``grain_strength`` is the standard deviation of monochrome Gaussian grain
    on a normalized 0–1 intensity scale. Grain is applied after the background
    blend so its visible amplitude is not accidentally attenuated twice.
    """

    if not 0.0 < reference_blend <= 1.0:
        raise ValueError("reference_blend must lie in (0, 1]")
    if blur_radius < 0.0:
        raise ValueError("blur_radius cannot be negative")
    if not 0.0 <= grain_strength <= 0.25:
        raise ValueError("grain_strength must lie in [0, 0.25]")
    image = previous.convert("RGB")
    softened = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    background = Image.new("RGB", image.size, background_rgb)
    reference = Image.blend(background, softened, reference_blend)
    if grain_strength == 0.0:
        return reference
    array = np.asarray(reference, dtype=np.float32)
    rng = np.random.default_rng(grain_seed)
    grain = rng.normal(
        loc=0.0,
        scale=255.0 * grain_strength,
        size=(array.shape[0], array.shape[1], 1),
    )
    grained = np.clip(array + grain, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(grained, mode="RGB")


def apply_prompt_prefix(prefix: str, prompt: str) -> str:
    """Combine a project-wide LoRA trigger/style prefix with one scene prompt."""

    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise ValueError("prompt cannot be blank")
    clean_prefix = prefix.strip().rstrip(",")
    if not clean_prefix or clean_prompt.casefold().startswith(clean_prefix.casefold()):
        return clean_prompt
    return f"{clean_prefix}, {clean_prompt}"


def generate_mainframes(
    pipeline: Any,
    spec: ArtLoopSpec,
    output_directory: str | Path,
    *,
    generator_device: str = "cuda",
) -> tuple[GeneratedMainframe, ...]:
    """Generate ordered mainframes, softly conditioning each on its predecessor."""

    output = Path(output_directory).expanduser().resolve(strict=False)
    reference_directory = output / "soft_references"
    output.mkdir(parents=True, exist_ok=True)
    if spec.generation.continuity.enabled:
        reference_directory.mkdir(parents=True, exist_ok=True)

    records: list[GeneratedMainframe] = []
    previous: Image.Image | None = None
    for index, mainframe in enumerate(spec.mainframes):
        seed = spec.generation.seed + mainframe.seed_offset
        effective_prompt = apply_prompt_prefix(spec.generation.prompt_prefix, mainframe.prompt)
        kwargs: dict[str, Any] = {
            "prompt": effective_prompt,
            "height": spec.generation.height,
            "width": spec.generation.width,
            "num_inference_steps": spec.generation.num_inference_steps,
            "guidance_scale": spec.generation.guidance_scale,
            "generator": torch.Generator(device=generator_device).manual_seed(seed),
            "output_type": "pil",
        }
        soft_reference_path: Path | None = None
        if previous is not None and spec.generation.continuity.enabled:
            continuity = spec.generation.continuity
            soft_reference = make_soft_reference(
                previous,
                reference_blend=continuity.reference_blend,
                blur_radius=continuity.blur_radius,
                grain_strength=continuity.grain_strength,
                grain_seed=seed,
                background_rgb=continuity.background_rgb,
            )
            soft_reference_path = reference_directory / f"reference_{index:03d}_{mainframe.id}.png"
            soft_reference.save(soft_reference_path)
            kwargs["image"] = soft_reference

        result = pipeline(**kwargs)
        if not getattr(result, "images", None):
            raise ArtLoopError(f"mainframe generation returned no image for {mainframe.id!r}")
        image = result.images[0].convert("RGB")
        path = output / f"mainframe_{index:03d}_{mainframe.id}.png"
        image.save(path)
        records.append(
            GeneratedMainframe(
                id=mainframe.id,
                prompt=effective_prompt,
                seed=seed,
                path=path,
                soft_reference_path=soft_reference_path,
            )
        )
        previous = image
    return tuple(records)


def collect_loop_frames(run_directories: list[str | Path]) -> list[Image.Image]:
    """Concatenate transition display frames without duplicated mainframe boundaries."""

    loop_frames: list[Image.Image] = []
    for directory in run_directories:
        paths = sorted(Path(directory).joinpath("display_frames").glob("frame_*.png"))
        if len(paths) < 2:
            raise ArtLoopError(f"transition has fewer than two display frames: {directory}")
        loop_frames.extend(Image.open(path).convert("RGB") for path in paths[:-1])
    if not loop_frames:
        raise ArtLoopError("no transition frames were found")
    return loop_frames


def persist_artifact_tree(
    source: str | Path,
    persistent_root: str | Path,
    *,
    project_name: str,
    label: str,
    now: datetime | None = None,
) -> ArtPersistenceReport:
    """Atomically publish one artifact tree to an auto-numbered timestamped folder."""

    source_path = Path(source).expanduser().resolve(strict=False)
    if not source_path.exists():
        raise ArtLoopError(f"artifact source does not exist: {source_path}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", project_name):
        raise ArtLoopError(f"unsafe project name: {project_name!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
        raise ArtLoopError(f"unsafe artifact label: {label!r}")

    project_root = Path(persistent_root).expanduser().resolve(strict=False) / project_name
    project_root.mkdir(parents=True, exist_ok=True)
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    timestamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    existing_numbers: list[int] = []
    prefix = f"{project_name}_"
    for candidate in project_root.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        remainder = candidate.name[len(prefix) :]
        number = remainder.split("_", 1)[0]
        if number.isdigit():
            existing_numbers.append(int(number))

    sequence = max(existing_numbers, default=0) + 1
    while True:
        folder_name = f"{project_name}_{sequence:04d}_{timestamp}_{label}"
        destination = project_root / folder_name
        partial = project_root / f"{folder_name}.partial"
        try:
            partial.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            sequence += 1
            continue
        if destination.exists():
            partial.rmdir()
            sequence += 1
            continue
        break

    staged = stage_drive_inputs(source_path, partial, overwrite=False)
    marker = {
        "schema_version": 1,
        "status": "complete",
        "project": project_name,
        "label": label,
        "sequence": sequence,
        "timestamp_utc": timestamp,
        "source": str(source_path),
        "file_count": len(staged),
        "total_bytes": sum(item.size_bytes for item in staged),
        "files": [
            {
                "path": str(item.destination.relative_to(partial)),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in staged
        ],
    }
    marker_path = partial / "COPY_COMPLETE.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, destination)
    return ArtPersistenceReport(
        source=source_path,
        destination=destination,
        project=project_name,
        label=label,
        sequence=sequence,
        timestamp_utc=timestamp,
        file_count=len(staged),
        total_bytes=sum(item.size_bytes for item in staged),
    )


def safe_transition_name(from_id: str, to_id: str) -> str:
    """Return a stable filesystem-safe transition label."""

    value = f"{from_id}_to_{to_id}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ArtLoopError(f"unsafe transition identifier: {value!r}")
    return value
