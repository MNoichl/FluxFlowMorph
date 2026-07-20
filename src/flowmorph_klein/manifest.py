"""Validated manifests for one or more endpoint-image pairs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from .config import ALLOWED_MODEL_IDS, BASE_MODEL_ID, FP8_MODEL_ID, StrictConfigModel
from .errors import ManifestError
from .types import HardwareProfile


class ManifestDefaults(StrictConfigModel):
    model_id: str = BASE_MODEL_ID
    profile: HardwareProfile = HardwareProfile.AUTO
    width: int = Field(default=512, gt=0)
    height: int = Field(default=512, gt=0)
    frame_count: int = Field(default=20, ge=2)
    seed: int = 42
    guidance_scale: float = Field(default=4.0, ge=0.0)
    lora_source: str | None = None
    lora_scale_fit: float = Field(default=1.0, ge=0.0)
    lora_scale_render: float = Field(default=1.0, ge=0.0)
    source_prompt: str | None = None
    target_prompt: str | None = None
    bridge_prompt: str | None = "a smooth transformation between the two subjects"
    negative_prompt: str = ""

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if value not in ALLOWED_MODEL_IDS:
            raise ValueError(
                "manifest model_id must be FLUX.2 Klein Base 9B or its explicit "
                "experimental Base-9B FP8 variant"
            )
        return value

    @model_validator(mode="after")
    def validate_model_profile_pair(self) -> "ManifestDefaults":
        if self.model_id == FP8_MODEL_ID:
            if self.profile is not HardwareProfile.FP8_9B_EXPERIMENTAL:
                raise ValueError("the FP8 model requires profile fp8_9b_experimental")
        elif self.profile is HardwareProfile.FP8_9B_EXPERIMENTAL:
            raise ValueError("the FP8 profile requires the Base-9B FP8 model")
        return self


class ManifestPair(StrictConfigModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    source_image: Path
    target_image: Path
    source_prompt: str | None = None
    target_prompt: str | None = None
    bridge_prompt: str | None = None
    negative_prompt: str | None = None


class InputManifest(StrictConfigModel):
    project_name: str = Field(min_length=1, max_length=128)
    defaults: ManifestDefaults = Field(default_factory=ManifestDefaults)
    pairs: tuple[ManifestPair, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_pair_ids(self) -> "InputManifest":
        ids = [pair.id for pair in self.pairs]
        duplicates = sorted({pair_id for pair_id in ids if ids.count(pair_id) > 1})
        if duplicates:
            raise ValueError("duplicate manifest pair ids: " + ", ".join(duplicates))
        return self


def _as_manifest(source: InputManifest | Mapping[str, Any] | str | Path) -> tuple[InputManifest, Path | None]:
    if isinstance(source, InputManifest):
        return source, None
    base_dir: Path | None = None
    if isinstance(source, (str, Path)):
        manifest_path = Path(source).expanduser()
        base_dir = manifest_path.resolve(strict=False).parent
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as error:
            raise ManifestError(f"cannot load manifest {manifest_path}: {error}") from error
        if not isinstance(data, dict):
            raise ManifestError(f"manifest {manifest_path} must contain a YAML mapping")
    else:
        data = dict(source)
    try:
        return InputManifest.model_validate(data), base_dir
    except ValidationError as error:
        raise ManifestError(f"invalid input manifest: {error}") from error


def load_manifest(path: str | Path) -> InputManifest:
    """Load and structurally validate a manifest without checking its files."""

    manifest, _ = _as_manifest(path)
    return manifest


def _resolve_input_path(path: Path, root: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    resolved = (resolved_root / expanded).resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ManifestError(
            f"relative manifest path {path!s} escapes its input root {resolved_root}"
        ) from error
    return resolved


def validate_manifest(
    source: InputManifest | Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
    require_files: bool = True,
) -> InputManifest:
    """Validate a manifest and resolve endpoint paths to absolute paths.

    Relative paths are confined to the manifest directory (or explicit
    ``base_dir``) to prevent accidental traversal out of a staged input bundle.
    """

    manifest, discovered_base = _as_manifest(source)
    root = Path(base_dir).expanduser() if base_dir is not None else discovered_base
    if root is None:
        root = Path.cwd()

    resolved_pairs: list[ManifestPair] = []
    missing: list[str] = []
    non_files: list[str] = []
    for pair in manifest.pairs:
        source_path = _resolve_input_path(pair.source_image, root)
        target_path = _resolve_input_path(pair.target_image, root)
        if require_files:
            for candidate in (source_path, target_path):
                if not candidate.exists():
                    missing.append(str(candidate))
                elif not candidate.is_file():
                    non_files.append(str(candidate))
        resolved_pairs.append(
            pair.model_copy(
                update={"source_image": source_path, "target_image": target_path}
            )
        )

    if missing:
        raise ManifestError(
            "manifest image paths do not exist: " + ", ".join(sorted(set(missing)))
        )
    if non_files:
        raise ManifestError(
            "manifest image paths must be files: " + ", ".join(sorted(set(non_files)))
        )
    return manifest.model_copy(update={"pairs": tuple(resolved_pairs)})


def get_manifest_pair(manifest: InputManifest, pair_id: str) -> ManifestPair:
    """Return one pair by stable ID."""

    for pair in manifest.pairs:
        if pair.id == pair_id:
            return pair
    raise ManifestError(f"manifest has no pair with id {pair_id!r}")


def pair_config_overrides(
    manifest: InputManifest,
    pair: ManifestPair | str,
) -> dict[str, Any]:
    """Translate manifest defaults and one pair into project config overlays."""

    selected = get_manifest_pair(manifest, pair) if isinstance(pair, str) else pair
    defaults = manifest.defaults

    def first(pair_value: str | None, default_value: str | None) -> str | None:
        return pair_value if pair_value is not None else default_value

    return {
        "project": {"name": manifest.project_name},
        "model": {"id": defaults.model_id, "profile": defaults.profile},
        "lora": {
            "source": defaults.lora_source,
            "fit_scale": defaults.lora_scale_fit,
            "render_scale": defaults.lora_scale_render,
        },
        "input": {
            "source_image": selected.source_image,
            "target_image": selected.target_image,
            "source_prompt": first(selected.source_prompt, defaults.source_prompt),
            "target_prompt": first(selected.target_prompt, defaults.target_prompt),
            "bridge_prompt": first(selected.bridge_prompt, defaults.bridge_prompt),
            "negative_prompt": first(selected.negative_prompt, defaults.negative_prompt),
            "width": defaults.width,
            "height": defaults.height,
        },
        "flowmorph": {"frame_count": defaults.frame_count},
        "guidance": {"scale": defaults.guidance_scale},
        "reproducibility": {"seed": defaults.seed},
    }
