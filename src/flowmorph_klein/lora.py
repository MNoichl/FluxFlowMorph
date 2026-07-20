"""FLUX.2 Klein Base-9B LoRA inspection, validation, and activation.

Version one intentionally supports ordinary LoRA only.  It rejects LoHa and
LoKr, keeps PEFT adapters unfused, and separates structural validation from
the mandatory numerical before/after check.  No Diffusers or safetensors
module is imported until the corresponding operation is requested.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .hf_assets import ResolvedHuggingFaceFile, resolve_huggingface_file


DEFAULT_ADAPTER_NAME = "flowmorph_adapter"
BASE_9B_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"


class LoraValidationError(ValueError):
    """A supplied adapter cannot be established as compatible."""


@dataclass(frozen=True, slots=True)
class SafetensorsInspection:
    path: Path | None
    keys: tuple[str, ...]
    metadata: dict[str, str]
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, str]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: object) -> bool:
        return key in self.keys


@dataclass(frozen=True, slots=True)
class LoraValidationReport:
    adapter_format: str
    compatible_tensor_count: int
    recognized_keys: tuple[str, ...]
    unrecognized_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_verified_keys: tuple[str, ...]
    architecture_evidence: tuple[str, ...]
    base_model_provenance: str
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.compatible_tensor_count > 0 and not self.missing_keys

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "adapter_format": self.adapter_format,
            "compatible_tensor_count": self.compatible_tensor_count,
            "recognized_keys": list(self.recognized_keys),
            "unrecognized_keys": list(self.unrecognized_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "shape_verified_keys": list(self.shape_verified_keys),
            "architecture_evidence": list(self.architecture_evidence),
            "base_model_provenance": self.base_model_provenance,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class AdapterActivationReport:
    adapter_name: str
    active_adapters: tuple[str, ...]
    listed_adapters: dict[str, tuple[str, ...]]
    active: bool

    def __bool__(self) -> bool:
        return self.active

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "active_adapters": list(self.active_adapters),
            "listed_adapters": {
                component: list(names)
                for component, names in self.listed_adapters.items()
            },
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class LoraNumericalReport:
    changed: bool
    maximum_absolute_difference: float
    mean_absolute_difference: float
    l2_difference: float
    element_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "maximum_absolute_velocity_difference": self.maximum_absolute_difference,
            "mean_absolute_velocity_difference": self.mean_absolute_difference,
            "l2_velocity_difference": self.l2_difference,
            "element_count": self.element_count,
        }


@dataclass(frozen=True, slots=True)
class LoraLoadReport:
    source: ResolvedHuggingFaceFile
    adapter_name: str
    scale: float
    fused: bool
    validation: LoraValidationReport
    activation: AdapterActivationReport
    adapter_parameter_count: int
    adapter_parameters_frozen: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.source.repo_id,
            "requested_revision": self.source.requested_revision,
            "resolved_revision": self.source.resolved_revision,
            "weight_name": self.source.weight_name,
            "sha256": self.source.sha256,
            "size_bytes": self.source.size_bytes,
            "adapter_name": self.adapter_name,
            "scale": self.scale,
            "fused": self.fused,
            "validation": self.validation.as_dict(),
            "active_adapters": list(self.activation.active_adapters),
            "listed_adapters": {
                component: list(names)
                for component, names in self.activation.listed_adapters.items()
            },
            "adapter_parameter_count": self.adapter_parameter_count,
            "adapter_parameters_frozen": self.adapter_parameters_frozen,
        }


def inspect_safetensors_keys(
    source: str | Path | Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SafetensorsInspection:
    """Inspect keys, metadata, shapes, and dtypes without loading all tensors."""

    if isinstance(source, Mapping):
        shapes: dict[str, tuple[int, ...]] = {}
        dtypes: dict[str, str] = {}
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                shapes[str(key)] = tuple(int(dim) for dim in value.shape)
                dtypes[str(key)] = str(value.dtype).removeprefix("torch.")
            elif isinstance(value, (tuple, list)) and all(isinstance(dim, int) for dim in value):
                shapes[str(key)] = tuple(value)
                dtypes[str(key)] = "unknown"
            else:
                shape = getattr(value, "shape", None)
                if shape is not None:
                    shapes[str(key)] = tuple(int(dim) for dim in shape)
                    dtypes[str(key)] = str(getattr(value, "dtype", "unknown"))
        keys = tuple(sorted(str(key) for key in source))
        return SafetensorsInspection(
            path=None,
            keys=keys,
            metadata={str(key): str(value) for key, value in (metadata or {}).items()},
            shapes=shapes,
            dtypes=dtypes,
        )

    path = Path(source).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"LoRA safetensors file does not exist: {path}")
    if path.suffix.lower() != ".safetensors":
        raise LoraValidationError("LoRA inspection supports only .safetensors files")
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - dependency error in production only
        raise RuntimeError("safetensors is required to inspect a LoRA checkpoint") from error

    shapes = {}
    dtypes = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = tuple(sorted(handle.keys()))
        file_metadata = handle.metadata() or {}
        for key in keys:
            tensor_slice = handle.get_slice(key)
            shapes[key] = tuple(int(dim) for dim in tensor_slice.get_shape())
            dtypes[key] = str(tensor_slice.get_dtype())
    merged_metadata = {str(key): str(value) for key, value in file_metadata.items()}
    merged_metadata.update({str(key): str(value) for key, value in (metadata or {}).items()})
    return SafetensorsInspection(path, keys, merged_metadata, shapes, dtypes)


_LORA_SUFFIX_RE = re.compile(
    r"(?P<marker>\.lora_(?P<side>A|B)(?:\.[^.]+)?|\.lora_(?P<direction>down|up))\.weight$"
)


def _adapter_tensor_parts(key: str) -> tuple[str, str] | None:
    match = _LORA_SUFFIX_RE.search(key)
    if match is None:
        return None
    stem = key[: match.start()]
    side = match.group("side")
    if side is None:
        side = "A" if match.group("direction") == "down" else "B"
    return stem, side


def _is_flux2_transformer_stem(stem: str) -> bool:
    lowered = stem.lower()
    flux2_markers = (
        "transformer_blocks.",
        "single_transformer_blocks.",
        "double_blocks.",
        "single_blocks.",
        "lora_unet_double_blocks_",
        "lora_unet_single_blocks_",
        "lora_unet_img_in",
        "lora_unet_txt_in",
        "lora_unet_time_in",
        "lora_unet_final_layer",
        "x_embedder",
        "context_embedder",
        "img_in",
        "txt_in",
        "time_in",
        "final_layer",
        "proj_out",
        "norm_out",
        "stream_modulation",
    )
    return any(marker in lowered for marker in flux2_markers)


def _adapter_format(keys: Iterable[str]) -> str:
    keys_tuple = tuple(keys)
    if any(".lora_down.weight" in key or ".lora_up.weight" in key for key in keys_tuple):
        return "kohya" if any(key.startswith("lora_unet_") for key in keys_tuple) else "ai_toolkit_down_up"
    if any(key.startswith("base_model.model.") for key in keys_tuple):
        return "peft"
    if any(key.startswith("diffusion_model.") for key in keys_tuple):
        return "ai_toolkit"
    return "diffusers"


def _metadata_text(metadata: Mapping[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(metadata.items())).lower()


def _provenance_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, bool, bool, bool]:
    text = _metadata_text(metadata)
    flux1 = any(marker in text for marker in ("flux.1", "flux1", "flux-dev", "flux_schnell", "flux-schnell"))
    four_b = bool(
        re.search(r"flux[._ -]?2[._ /-]*klein[^\n]*?(?:base[._ -]*)?4b", text)
        or "klein-base-4b" in text
    )
    base_9b = "flux.2-klein-base-9b" in text or "flux2-klein-base-9b" in text
    distilled_9b = "distill" in text
    if not base_9b and re.search(r"flux[._ -]?2[._ /-]*klein[._ /-]*9b", text):
        distilled_9b = True
    if flux1:
        provenance = "flux1"
    elif four_b:
        provenance = "flux2_klein_4b"
    elif base_9b:
        provenance = "flux2_klein_base_9b"
    elif distilled_9b:
        provenance = "flux2_klein_distilled_9b"
    else:
        provenance = "unknown"
    return provenance, flux1, four_b, distilled_9b


def _shape_architecture_evidence(shapes: Mapping[str, tuple[int, ...]], keys: Iterable[str]) -> tuple[set[str], bool]:
    evidence: set[str] = set()
    dimensions = {dimension for shape in shapes.values() for dimension in shape}
    four_b_shape = bool(dimensions.intersection({3072, 7680}))
    if dimensions.intersection({4096, 12288}):
        evidence.add("adapter tensor dimensions include a Base/Distilled-9B width (4096 or 12288)")
    if four_b_shape:
        evidence.add("adapter tensor dimensions include a Klein-4B width (3072 or 7680)")

    for key in keys:
        lowered = key.lower()
        for expression, threshold, label in (
            (r"single_transformer_blocks\.(\d+)", 20, "single block index >=20 proves the 24-block 9B layout"),
            (r"single_blocks[._](\d+)", 20, "single block index >=20 proves the 24-block 9B layout"),
            (r"lora_unet_single_blocks_(\d+)", 20, "single block index >=20 proves the 24-block 9B layout"),
            (r"transformer_blocks\.(\d+)", 5, "dual block index >=5 proves the 8-block 9B layout"),
            (r"double_blocks[._](\d+)", 5, "dual block index >=5 proves the 8-block 9B layout"),
            (r"lora_unet_double_blocks_(\d+)", 5, "dual block index >=5 proves the 8-block 9B layout"),
        ):
            match = re.search(expression, lowered)
            if match and int(match.group(1)) >= threshold:
                evidence.add(label)
    return evidence, four_b_shape


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _validate_loaded_transformer_architecture(transformer: Any) -> tuple[str, ...]:
    config = getattr(transformer, "config", None)
    if config is None:
        return ()
    expected = {
        "in_channels": 128,
        "num_layers": 8,
        "num_single_layers": 24,
        "attention_head_dim": 128,
        "num_attention_heads": 32,
        "joint_attention_dim": 12288,
        "guidance_embeds": False,
    }
    checked: list[str] = []
    mismatches: list[str] = []
    for name, expected_value in expected.items():
        actual = _config_value(config, name)
        if actual is None:
            continue
        checked.append(f"transformer.config.{name}={actual!r}")
        if actual != expected_value:
            mismatches.append(f"{name}={actual!r} (expected {expected_value!r})")
    if mismatches:
        raise LoraValidationError(
            "loaded transformer is not FLUX.2 Klein Base/Distilled 9B: " + ", ".join(mismatches)
        )
    return tuple(checked)


def _target_stems(stem: str) -> tuple[str, ...]:
    normalized = stem
    for prefix in ("base_model.model.", "diffusion_model.", "transformer."):
        while normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    if normalized.startswith(("transformer_blocks.", "single_transformer_blocks.")):
        return (normalized,)
    if normalized in {"x_embedder", "context_embedder", "proj_out"} or normalized.startswith(
        ("time_guidance_embed.", "norm_out.", "single_stream_modulation.", "double_stream_modulation_")
    ):
        return (normalized,)

    single = re.fullmatch(r"single_blocks\.(\d+)\.linear([12])", normalized)
    if single:
        block, linear = single.groups()
        target = "to_qkv_mlp_proj" if linear == "1" else "to_out"
        return (f"single_transformer_blocks.{block}.attn.{target}",)

    double = re.fullmatch(
        r"double_blocks\.(\d+)\.(img_attn|txt_attn|img_mlp|txt_mlp)\.(qkv|proj|[02])",
        normalized,
    )
    if double:
        block, stream, projection = double.groups()
        prefix = f"transformer_blocks.{block}"
        if stream == "img_attn" and projection == "qkv":
            return tuple(f"{prefix}.attn.to_{name}" for name in ("q", "k", "v"))
        if stream == "txt_attn" and projection == "qkv":
            return tuple(f"{prefix}.attn.add_{name}_proj" for name in ("q", "k", "v"))
        if stream == "img_attn" and projection == "proj":
            return (f"{prefix}.attn.to_out.0",)
        if stream == "txt_attn" and projection == "proj":
            return (f"{prefix}.attn.to_add_out",)
        feed_forward = "ff" if stream == "img_mlp" else "ff_context"
        linear = "linear_in" if projection == "0" else "linear_out"
        return (f"{prefix}.{feed_forward}.{linear}",)

    extra_mappings = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
        "final_layer.linear": "proj_out",
        "final_layer.adaln_modulation.1": "norm_out.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "lora_unet_img_in": "x_embedder",
        "lora_unet_txt_in": "context_embedder",
        "lora_unet_time_in_in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "lora_unet_time_in_out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "lora_unet_final_layer_linear": "proj_out",
    }
    if normalized.lower() in extra_mappings:
        return (extra_mappings[normalized.lower()],)

    kohya_single = re.fullmatch(r"lora_unet_single_blocks_(\d+)_linear([12])", normalized)
    if kohya_single:
        block, linear = kohya_single.groups()
        target = "to_qkv_mlp_proj" if linear == "1" else "to_out"
        return (f"single_transformer_blocks.{block}.attn.{target}",)
    kohya_double = re.fullmatch(
        r"lora_unet_double_blocks_(\d+)_(img_attn|txt_attn|img_mlp|txt_mlp)_(qkv|proj|[02])",
        normalized,
    )
    if kohya_double:
        block, stream, projection = kohya_double.groups()
        return _target_stems(f"double_blocks.{block}.{stream}.{projection}")
    return ()


def _parameter_shapes(transformer: Any) -> dict[str, tuple[int, ...]]:
    named_parameters = getattr(transformer, "named_parameters", None)
    if not callable(named_parameters):
        return {}
    return {
        name: tuple(int(dimension) for dimension in parameter.shape)
        for name, parameter in named_parameters()
    }


def _verify_mapped_shapes(
    pairs: Mapping[str, dict[str, str]],
    inspection: SafetensorsInspection,
    transformer: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    base_shapes = _parameter_shapes(transformer)
    if not base_shapes:
        return (), ()
    verified: list[str] = []
    unexpected: list[str] = []
    for stem, sides in pairs.items():
        targets = _target_stems(stem)
        if not targets:
            continue
        target_shapes = [base_shapes.get(f"{target}.weight") for target in targets]
        if any(shape is None for shape in target_shapes):
            unexpected.extend(sides.values())
            continue
        concrete_target_shapes = [shape for shape in target_shapes if shape is not None]
        a_key = sides["A"]
        b_key = sides["B"]
        a_shape = inspection.shapes.get(a_key)
        b_shape = inspection.shapes.get(b_key)
        if a_shape is None or b_shape is None:
            continue
        if (
            len(a_shape) != 2
            or len(b_shape) != 2
            or any(len(shape) != 2 for shape in concrete_target_shapes)
        ):
            raise LoraValidationError(f"LoRA and target weights must be matrices for {stem!r}")
        expected_input = concrete_target_shapes[0][1]
        expected_output = sum(shape[0] for shape in concrete_target_shapes)
        if any(shape[1] != expected_input for shape in concrete_target_shapes):
            raise LoraValidationError(f"fused target projections disagree on input width for {stem!r}")
        if a_shape[0] != b_shape[1] or a_shape[1] != expected_input or b_shape[0] != expected_output:
            raise LoraValidationError(
                f"LoRA shape mismatch for {stem}: A={a_shape}, B={b_shape}, "
                f"target projections={tuple(concrete_target_shapes)}"
            )
        verified.extend((a_key, b_key))
    return tuple(sorted(verified)), tuple(sorted(set(unexpected)))


def _coerce_inspection(
    value: SafetensorsInspection | Mapping[str, Any] | Iterable[str] | str | Path,
    metadata: Mapping[str, Any] | None,
) -> SafetensorsInspection:
    if isinstance(value, SafetensorsInspection):
        if metadata:
            merged = dict(value.metadata)
            merged.update({str(key): str(item) for key, item in metadata.items()})
            return SafetensorsInspection(value.path, value.keys, merged, value.shapes, value.dtypes)
        return value
    if isinstance(value, (str, Path, Mapping)):
        return inspect_safetensors_keys(value, metadata=metadata)
    keys = tuple(sorted(str(key) for key in value))
    return SafetensorsInspection(
        path=None,
        keys=keys,
        metadata={str(key): str(item) for key, item in (metadata or {}).items()},
        shapes={},
        dtypes={},
    )


def validate_flux2_klein_9b_lora(
    inspection_or_keys: SafetensorsInspection | Mapping[str, Any] | Iterable[str] | str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    transformer: Any | None = None,
    require_base_9b_provenance: bool = False,
    allow_distilled_9b: bool = False,
) -> LoraValidationReport:
    """Validate adapter format, tensor pairs, architecture, and provenance.

    Base and distilled Klein 9B have identical tensor shapes.  Therefore a
    distilled-vs-Base decision is made only from checkpoint provenance; shape
    inference is never presented as proof of that distinction.
    """

    inspection = _coerce_inspection(inspection_or_keys, metadata)
    if not inspection.keys:
        raise LoraValidationError("LoRA checkpoint contains no tensors")
    lowered_keys = tuple(key.lower() for key in inspection.keys)
    if any(
        marker in key
        for key in lowered_keys
        for marker in ("loha", "lokr", "hada_w", "hada_t", "lokr_w")
    ):
        raise LoraValidationError("LoHa and LoKr adapters are unsupported in version 1")

    provenance, is_flux1, is_four_b, is_distilled = _provenance_from_metadata(inspection.metadata)
    if is_flux1:
        raise LoraValidationError("adapter metadata identifies a FLUX.1 model, not FLUX.2 Klein 9B")
    if is_four_b:
        raise LoraValidationError("adapter metadata identifies FLUX.2 Klein 4B, not Klein 9B")
    if is_distilled and not allow_distilled_9b:
        raise LoraValidationError(
            "adapter metadata identifies distilled FLUX.2 Klein 9B; set the explicit "
            "allow_distilled_9b override only after reviewing that mismatch"
        )

    architecture_evidence, four_b_shape = _shape_architecture_evidence(
        inspection.shapes,
        inspection.keys,
    )
    if four_b_shape and not any("4096 or 12288" in item for item in architecture_evidence):
        raise LoraValidationError("adapter tensor shapes are consistent with Klein 4B, not 9B")
    if transformer is not None:
        architecture_evidence.update(_validate_loaded_transformer_architecture(transformer))

    recognized: set[str] = set()
    unrecognized: set[str] = set()
    pairs: dict[str, dict[str, str]] = {}
    auxiliary: set[str] = set()
    dora_keys: list[str] = []
    for key in inspection.keys:
        lowered = key.lower()
        if "dora_scale" in lowered:
            dora_keys.append(key)
            auxiliary.add(key)
            continue
        if lowered.endswith(".alpha") and lowered.startswith("lora_unet_"):
            auxiliary.add(key)
            continue
        parts = _adapter_tensor_parts(key)
        if parts is None:
            unrecognized.add(key)
            continue
        stem, side = parts
        if not _is_flux2_transformer_stem(stem):
            unrecognized.add(key)
            continue
        recognized.add(key)
        pairs.setdefault(stem, {})[side] = key

    missing: list[str] = []
    complete_pairs: dict[str, dict[str, str]] = {}
    for stem, sides in sorted(pairs.items()):
        if set(sides) == {"A", "B"}:
            complete_pairs[stem] = sides
        else:
            missing_side = "B" if "A" in sides else "A"
            missing.append(f"{stem}: missing LoRA {missing_side} tensor")
    if not complete_pairs:
        raise LoraValidationError(
            "zero compatible FLUX.2 transformer LoRA A/B (or down/up) tensor pairs were found"
        )
    if missing:
        raise LoraValidationError("incomplete LoRA tensor pairs: " + "; ".join(missing))

    shape_verified: tuple[str, ...] = ()
    unexpected_from_shapes: tuple[str, ...] = ()
    if transformer is not None:
        shape_verified, unexpected_from_shapes = _verify_mapped_shapes(
            complete_pairs,
            inspection,
            transformer,
        )
        mapped_pair_count = sum(bool(_target_stems(stem)) for stem in complete_pairs)
        if mapped_pair_count and inspection.shapes and not shape_verified:
            raise LoraValidationError("zero LoRA tensor pairs map to the loaded transformer")

    warnings: list[str] = []
    if dora_keys:
        warnings.append(
            "DoRA scale tensors are present; pinned Diffusers ignores those tensors and loads the LoRA factors only"
        )
    if is_distilled:
        warnings.append("distilled Klein-9B provenance was explicitly overridden for a Base-9B run")
    if provenance == "unknown":
        warnings.append(
            "checkpoint metadata does not identify its base model; 9B tensor shape checks cannot distinguish "
            "Base from distilled Klein 9B"
        )
        if require_base_9b_provenance:
            raise LoraValidationError(
                "adapter lacks verifiable FLUX.2 Klein Base-9B provenance; disable strict provenance only "
                "through an explicit reviewed override"
            )

    unexpected = tuple(sorted(set(unrecognized).union(unexpected_from_shapes)))
    return LoraValidationReport(
        adapter_format=_adapter_format(inspection.keys),
        compatible_tensor_count=sum(len(sides) for sides in complete_pairs.values()),
        recognized_keys=tuple(sorted(recognized.union(auxiliary))),
        unrecognized_keys=tuple(sorted(unrecognized)),
        missing_keys=(),
        unexpected_keys=unexpected,
        shape_verified_keys=shape_verified,
        architecture_evidence=tuple(sorted(architecture_evidence)),
        base_model_provenance=provenance,
        warnings=tuple(warnings),
    )


def compute_adapter_fingerprint(source: str | Path | ResolvedHuggingFaceFile) -> str:
    """Return the SHA-256 fingerprint of a selected adapter file."""

    if isinstance(source, ResolvedHuggingFaceFile):
        return source.sha256
    path = Path(source).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"adapter file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_active_adapter(
    pipeline: Any,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    *,
    strict: bool = True,
) -> AdapterActivationReport:
    """Verify both registration and active-adapter state through native APIs."""

    get_active = getattr(pipeline, "get_active_adapters", None)
    get_list = getattr(pipeline, "get_list_adapters", None)
    if not callable(get_active) or not callable(get_list):
        raise RuntimeError("pipeline does not expose the native Diffusers adapter inspection APIs")
    active_adapters = tuple(str(name) for name in (get_active() or ()))
    raw_listed = get_list() or {}
    if not isinstance(raw_listed, Mapping):
        raise RuntimeError("pipeline.get_list_adapters() returned an invalid value")
    listed = {
        str(component): tuple(str(name) for name in names)
        for component, names in raw_listed.items()
    }
    transformer_adapters = listed.get("transformer", ())
    is_active = adapter_name in active_adapters and adapter_name in transformer_adapters
    report = AdapterActivationReport(adapter_name, active_adapters, listed, is_active)
    if strict and not report.active:
        raise RuntimeError(
            f"adapter {adapter_name!r} is not both registered on the transformer and active"
        )
    return report


def compare_lora_velocities(
    baseline: torch.Tensor,
    adapted: torch.Tensor,
    *,
    atol: float = 0.0,
) -> LoraNumericalReport:
    """Produce the mandatory numerical evidence that an adapter changes output."""

    if baseline.shape != adapted.shape:
        raise ValueError("baseline and adapted velocity tensors must have identical shapes")
    if not baseline.is_floating_point() or not adapted.is_floating_point():
        raise TypeError("velocity tensors must be floating point")
    if not torch.isfinite(baseline).all() or not torch.isfinite(adapted).all():
        raise ValueError("velocity tensors must contain only finite values")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    difference = (adapted.detach().to(torch.float32) - baseline.detach().to(torch.float32)).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    mean = float(difference.mean().item()) if difference.numel() else 0.0
    l2 = float(torch.linalg.vector_norm(difference).item()) if difference.numel() else 0.0
    return LoraNumericalReport(
        changed=maximum > atol,
        maximum_absolute_difference=maximum,
        mean_absolute_difference=mean,
        l2_difference=l2,
        element_count=difference.numel(),
    )


def run_lora_velocity_smoke_test(
    pipeline: Any,
    predict_velocity: Any,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    scale: float = 1.0,
    atol: float = 0.0,
    require_change: bool = True,
) -> LoraNumericalReport:
    """Evaluate one deterministic velocity callback with LoRA off and on.

    ``predict_velocity`` is a zero-argument callback closing over one fixed
    state, timestep, IDs, and prompt package. The adapter is always restored
    active before returning or raising.
    """

    if not callable(predict_velocity):
        raise TypeError("predict_velocity must be a zero-argument callable")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("LoRA scale must be finite and non-negative")
    disable = getattr(pipeline, "disable_lora", None)
    enable = getattr(pipeline, "enable_lora", None)
    setter = getattr(pipeline, "set_adapters", None)
    if not callable(disable) or not callable(enable) or not callable(setter):
        raise RuntimeError("pipeline lacks native enable/disable/set adapter APIs")

    try:
        disable()
        with torch.no_grad():
            baseline = predict_velocity()
        if not isinstance(baseline, torch.Tensor):
            raise TypeError("predict_velocity callback must return a torch.Tensor")
    finally:
        enable()
        setter(adapter_name, adapter_weights=float(scale))
    verify_active_adapter(pipeline, adapter_name)
    with torch.no_grad():
        adapted = predict_velocity()
    if not isinstance(adapted, torch.Tensor):
        raise TypeError("predict_velocity callback must return a torch.Tensor")
    report = compare_lora_velocities(baseline, adapted, atol=atol)
    if require_change and not report.changed:
        raise RuntimeError(
            "active LoRA did not change the deterministic velocity tensor; loading alone is not evidence"
        )
    return report


def _adapter_parameter_status(transformer: Any) -> tuple[int, bool]:
    named_parameters = getattr(transformer, "named_parameters", None)
    if not callable(named_parameters):
        return 0, True
    parameters = [
        parameter
        for name, parameter in named_parameters()
        if "lora_" in name.lower() or ".lora" in name.lower()
    ]
    return sum(parameter.numel() for parameter in parameters), all(
        not parameter.requires_grad and parameter.grad is None for parameter in parameters
    )


def load_flux2_lora(
    pipeline: Any,
    source: str | Path | ResolvedHuggingFaceFile,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    scale: float = 1.0,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    subfolder: str | None = None,
    weight_name: str | None = None,
    local_files_only: bool = False,
    require_base_9b_provenance: bool = True,
    allow_distilled_9b: bool = False,
    api: Any | None = None,
    download_fn: Any | None = None,
) -> LoraLoadReport:
    """Resolve, validate, natively load, activate, scale, and freeze one LoRA."""

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", adapter_name):
        raise ValueError("adapter_name must be a stable alphanumeric identifier")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("LoRA scale must be finite and non-negative")
    if int(getattr(pipeline, "num_fused_loras", 0) or 0) != 0:
        raise RuntimeError("pipeline already contains fused LoRA weights; version 1 requires unfused adapters")
    resolved = (
        source
        if isinstance(source, ResolvedHuggingFaceFile)
        else resolve_huggingface_file(
            source,
            revision=revision,
            subfolder=subfolder,
            weight_name=weight_name,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            api=api,
            download_fn=download_fn,
        )
    )
    inspection = inspect_safetensors_keys(resolved.local_path)
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise TypeError("pipeline must expose its FLUX.2 transformer")
    pipeline_config = getattr(pipeline, "config", None)
    if bool(_config_value(pipeline_config, "is_distilled")):
        raise LoraValidationError("the loaded pipeline is distilled; FlowMorph requires Klein Base 9B")
    validation = validate_flux2_klein_9b_lora(
        inspection,
        transformer=transformer,
        require_base_9b_provenance=require_base_9b_provenance,
        allow_distilled_9b=allow_distilled_9b,
    )

    loader = getattr(pipeline, "load_lora_weights", None)
    setter = getattr(pipeline, "set_adapters", None)
    if not callable(loader) or not callable(setter):
        raise RuntimeError("pipeline lacks native Diffusers LoRA loading/adapter APIs")
    # Passing the selected local file prevents Diffusers from guessing among
    # multiple repository weights. Explicit safetensors forbids pickle fallback.
    loader(
        str(resolved.local_path),
        adapter_name=adapter_name,
        use_safetensors=True,
    )
    setter(adapter_name, adapter_weights=float(scale))
    fused_names = tuple(str(name) for name in (getattr(pipeline, "fused_loras", ()) or ()))
    if adapter_name in fused_names:
        raise RuntimeError("adapter was unexpectedly fused; FlowMorph version 1 requires PEFT hooks")

    # The fitting objective differentiates only with respect to the packed
    # state and endpoint variables. Freeze base and PEFT parameters alike.
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    evaluation = getattr(transformer, "eval", None)
    if callable(evaluation):
        evaluation()

    activation = verify_active_adapter(pipeline, adapter_name)
    adapter_parameter_count, adapter_parameters_frozen = _adapter_parameter_status(transformer)
    if adapter_parameter_count <= 0:
        raise RuntimeError("adapter APIs report active but no LoRA parameters exist on the transformer")
    if not adapter_parameters_frozen:
        raise RuntimeError("one or more LoRA parameters remain trainable or retain gradients")
    return LoraLoadReport(
        source=resolved,
        adapter_name=adapter_name,
        scale=float(scale),
        fused=False,
        validation=validation,
        activation=activation,
        adapter_parameter_count=adapter_parameter_count,
        adapter_parameters_frozen=adapter_parameters_frozen,
    )
