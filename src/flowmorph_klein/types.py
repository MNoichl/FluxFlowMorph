"""Small dependency-free contracts shared by the package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class StringEnum(str, Enum):
    """A Python 3.10-compatible string enum."""

    def __str__(self) -> str:
        return self.value


class RunMode(StringEnum):
    REFERENCE = "reference"
    SMOKE = "smoke"
    DIAGNOSTIC = "diagnostic"
    EXPERIMENTAL = "experimental"


class HardwareProfile(StringEnum):
    AUTO = "auto"
    A100_80GB_FULL = "a100_80gb_full"
    A100_40GB_CHECKPOINTED = "a100_40gb_checkpointed"
    FP8_9B_EXPERIMENTAL = "fp8_9b_experimental"
    UNSUPPORTED_LOW_VRAM = "unsupported_low_vram"


class ResizeMode(StringEnum):
    STRETCH = "stretch"
    CENTER_CROP = "center_crop"
    CONTAIN_AND_PAD = "contain_and_pad"


class ComputeDType(StringEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class QuantizationMode(StringEnum):
    NONE = "none"
    FP8 = "fp8"


class AttentionBackend(StringEnum):
    SDPA = "sdpa"


class LossMode(StringEnum):
    CODE_L2_NORM = "code_l2_norm"
    PAPER_L2_SQUARED = "paper_l2_squared"


class OptimizerName(StringEnum):
    ADAMW = "adamw"


class InterpolationMode(StringEnum):
    DECOUPLED = "decoupled"


class AlphaSchedule(StringEnum):
    LINEAR = "linear"


class RenderConditioningMode(StringEnum):
    SOURCE = "source"
    TARGET = "target"
    SHARED_BRIDGE = "shared_bridge"
    PROMPT_SCHEDULE = "prompt_schedule"
    INTERPOLATED_EMBEDDINGS = "interpolated_embeddings"
    NEAREST_ENDPOINT = "nearest_endpoint"


class CFGExecution(StringEnum):
    SEQUENTIAL = "sequential"
    BATCHED = "batched"


@dataclass(frozen=True, slots=True)
class ColabPaths:
    """Resolved local and optional persistent roots for a Colab run."""

    project_root: Path
    input_root: Path
    work_root: Path
    result_root: Path
    hf_cache: Path
    drive_root: Path | None = None


@dataclass(frozen=True, slots=True)
class FileChecksum:
    """Checksum metadata for one staged file."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class StagedFile:
    """A file copied into local run storage."""

    source_name: str
    destination: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """A decoded endpoint image and its reproducibility metadata.

    ``image`` is intentionally typed as ``Any`` so importing shared contracts
    does not import Pillow in configuration-only commands.
    """

    image: Any
    source_path: Path | None
    output_path: Path | None
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    resize_mode: ResizeMode
    original_sha256: str | None
    preprocessing_sha256: str
