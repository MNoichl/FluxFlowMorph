"""CHIMERA-style zero-shot morphing for FLUX.2 Klein.

This module ports the architecture-independent parts of CHIMERA to the
repository's native FLUX.2 Euler flow-matching stack:

* reverse Euler inversion of both endpoint latents;
* representative early/middle/late transformer feature caches;
* FFT-calibrated Layer- and Timestep-wise Frequency Matching (LTM);
* depth/timestep-aware Adaptive Cache Injection (ACI);
* linear Inversion-Denoising Timestep Mapping (IDM);
* early-step Semantic Anchor Prompting (SAP); and
* the paper's Global-Local Consistency Score (GLCS) aggregation.

The paper's released algorithm is U-Net-centric and its public repository did
not contain implementation code when this port was written.  FLUX has no
down/mid/up blocks, so the three feature groups are mapped to representative
early/middle/late transformer depths.  A resumable calibration pass computes
the paper's radial FFT descriptors for every group and timestep and derives the
frequency-nearest lookup used by inversion and denoising.  The default cache is
int8-quantized on CPU and sampled every second inversion step so a 9B, 1024px
Colab run remains practical.  Both cache choices are explicit, configurable
approximations rather than silent claims of bit-for-bit parity with the
unpublished reference implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .conditioning import (
    ConditioningPackage,
    interpolate_conditioning,
    stack_conditioning_packages,
)
from .flow_schedule import FlowSchedule, euler_flow_update
from .flux2_model import predict_cfg_velocity, predict_conditional_velocity
from .interpolation import slerp, slerp_direction_and_magnitude
from .pipeline import FlowMorphRunner, PipelineError
from .renderer import RenderedLatentFrame
from .sequence import EncodedSequenceImage, FlowMorphSequenceSession
from .types import RenderConditioningMode


Tensor = torch.Tensor
CHIMERA_GROUPS = ("early", "middle", "late")
CacheStorage = Literal["int8", "float16", "bfloat16", "float32"]
GroupName = Literal["early", "middle", "late"]
LTMMode = Literal["fft", "linear"]
ConditioningInterpolation = Literal["linear", "slerp"]
LTM_CALIBRATION_VERSION = 2
LTM_MINIMUM_GROUP_FRACTION = 0.10
LTM_TIMESTEP_SMOOTHING_RADIUS = 1


def center_weighted_alpha_schedule(
    midpoint_count: int,
    *,
    strength: float = 0.5,
) -> tuple[float, ...]:
    """Return a monotone endpoint-preserving schedule concentrated at 0.5.

    ``strength=0`` is the usual uniform schedule.  Positive strengths apply
    ``u + strength * sin(2πu) / (2π)`` so samples on both sides move toward
    the midpoint without changing the number of FLUX renders.  Restricting
    strength to less than one keeps the continuous mapping strictly monotone.
    """

    if midpoint_count < 1:
        raise ValueError("midpoint_count must be positive")
    if not math.isfinite(strength) or not 0.0 <= strength < 1.0:
        raise ValueError("alpha warp strength must lie in [0, 1)")
    denominator = midpoint_count + 1
    return tuple(
        u + strength * math.sin(2.0 * math.pi * u) / (2.0 * math.pi)
        for index in range(1, midpoint_count + 1)
        for u in (index / denominator,)
    )


def allocate_perceptual_subdivisions(
    distances: Sequence[float],
    *,
    average_multiplier: int,
    minimum_multiplier: int = 2,
    maximum_multiplier: int | None = None,
    maximum_weight_ratio: float = 2.5,
) -> tuple[int, ...]:
    """Distribute a fixed RIFE budget according to adjacent-frame distance.

    The returned integer multipliers always sum to
    ``len(distances) * average_multiplier``.  A robust upper clip prevents one
    pathological edge from consuming the complete budget, while minimum and
    maximum multipliers keep every original transition represented.
    """

    values = tuple(float(value) for value in distances)
    if not values:
        raise ValueError("perceptual allocation needs at least one distance")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("perceptual distances must be finite and non-negative")
    if average_multiplier < 1 or minimum_multiplier < 1:
        raise ValueError("RIFE multipliers must be positive")
    if minimum_multiplier > average_multiplier:
        raise ValueError("minimum_multiplier cannot exceed average_multiplier")
    if maximum_multiplier is None:
        maximum_multiplier = len(values) * average_multiplier
    if maximum_multiplier < average_multiplier:
        raise ValueError("maximum_multiplier cannot be below average_multiplier")
    if not math.isfinite(maximum_weight_ratio) or maximum_weight_ratio < 1.0:
        raise ValueError("maximum_weight_ratio must be finite and at least one")

    positive = sorted(value for value in values if value > 0.0)
    if positive:
        middle = len(positive) // 2
        median = (
            positive[middle]
            if len(positive) % 2
            else (positive[middle - 1] + positive[middle]) / 2.0
        )
        ceiling = median * maximum_weight_ratio
        epsilon = max(median * 1e-6, torch.finfo(torch.float64).eps)
        weights = [max(epsilon, min(value, ceiling)) for value in values]
    else:
        weights = [1.0] * len(values)

    target = len(values) * average_multiplier
    allocations = [minimum_multiplier] * len(values)
    remaining = target - sum(allocations)
    capacities = [maximum_multiplier - minimum_multiplier] * len(values)
    ideal_extras = [0.0] * len(values)
    active = set(range(len(values)))
    while remaining > 0 and active:
        active_weight = sum(weights[index] for index in active)
        provisional = {
            index: remaining * weights[index] / active_weight
            for index in active
        }
        capped = [
            index
            for index in active
            if provisional[index] >= capacities[index]
        ]
        if not capped:
            for index in active:
                ideal_extras[index] = provisional[index]
            break
        for index in capped:
            ideal_extras[index] = float(capacities[index])
            remaining -= capacities[index]
            active.remove(index)

    for index, extra in enumerate(ideal_extras):
        allocations[index] += int(math.floor(extra))
    units_left = target - sum(allocations)
    while units_left:
        candidates = [
            index
            for index, allocation in enumerate(allocations)
            if allocation < maximum_multiplier
        ]
        if not candidates:
            raise RuntimeError("bounded RIFE allocation exhausted before reaching its budget")
        chosen = max(
            candidates,
            key=lambda index: (
                ideal_extras[index] - math.floor(ideal_extras[index]),
                weights[index],
                -index,
            ),
        )
        allocations[chosen] += 1
        ideal_extras[chosen] = float(math.floor(ideal_extras[chosen]))
        units_left -= 1

    return tuple(allocations)


@dataclass(frozen=True, slots=True)
class ChimeraConfig:
    """Runtime settings for the FLUX CHIMERA port.

    ``cache_stride=1`` and ``cache_storage='float32'`` are the closest settings
    to the paper's uncompressed per-timestep cache.  The defaults trade a
    bounded amount of guidance fidelity for substantially lower host memory.
    """

    inversion_steps: int = 50
    denoising_steps: int = 50
    aci_weight: float = 0.4
    sap_active_ratio: float = 0.2
    anchor_max_tokens: int = 64
    anchor_reliability_threshold: float = 0.45
    ltm_mode: LTMMode = "fft"
    ltm_bands: int = 16
    ltm_channel_chunk_size: int = 128
    cache_stride: int = 2
    cache_storage: CacheStorage = "int8"
    render_batch_size: int = 2
    render_batch_max: int = 10
    auto_render_batch_size: bool = True
    batch_memory_reserve_fraction: float = 0.10
    batch_memory_reserve_gib: float = 2.0
    batch_estimate_overhead: float = 1.25
    decode_batch_size: int = 4
    guidance_scale: float = 7.0
    lora_scale: float = 1.2
    conditioning_interpolation: ConditioningInterpolation = "slerp"
    cfg_execution: Literal["sequential", "batched"] = "batched"
    oom_backoff: bool = True

    def __post_init__(self) -> None:
        if self.inversion_steps < 2 or self.denoising_steps < 2:
            raise ValueError("CHIMERA inversion and denoising need at least two steps")
        if not 0.0 <= self.aci_weight <= 2.0:
            raise ValueError("aci_weight must lie in [0, 2]")
        if not 0.0 <= self.sap_active_ratio <= 1.0:
            raise ValueError("sap_active_ratio must lie in [0, 1]")
        if self.anchor_max_tokens < 1:
            raise ValueError("anchor_max_tokens must be positive")
        if not -1.0 <= self.anchor_reliability_threshold <= 1.0:
            raise ValueError("anchor_reliability_threshold must lie in [-1, 1]")
        if self.ltm_mode not in {"fft", "linear"}:
            raise ValueError("ltm_mode must be 'fft' or 'linear'")
        if self.ltm_bands < 2:
            raise ValueError("ltm_bands must be at least two")
        if self.ltm_channel_chunk_size < 1:
            raise ValueError("ltm_channel_chunk_size must be positive")
        if self.cache_stride < 1:
            raise ValueError("cache_stride must be positive")
        if self.cache_storage not in {"int8", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported cache storage {self.cache_storage!r}")
        if self.render_batch_size < 1 or self.decode_batch_size < 1:
            raise ValueError("CHIMERA batch sizes must be positive")
        if self.render_batch_max < self.render_batch_size:
            raise ValueError("render_batch_max must be at least render_batch_size")
        if not 0.0 <= self.batch_memory_reserve_fraction < 1.0:
            raise ValueError("batch_memory_reserve_fraction must lie in [0, 1)")
        if not math.isfinite(self.batch_memory_reserve_gib) or self.batch_memory_reserve_gib < 0:
            raise ValueError("batch_memory_reserve_gib must be finite and non-negative")
        if not math.isfinite(self.batch_estimate_overhead) or self.batch_estimate_overhead < 1:
            raise ValueError("batch_estimate_overhead must be finite and at least one")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if not math.isfinite(self.lora_scale) or self.lora_scale <= 0:
            raise ValueError("lora_scale must be finite and positive")
        if self.conditioning_interpolation not in {"linear", "slerp"}:
            raise ValueError("conditioning_interpolation must be 'linear' or 'slerp'")


def estimate_safe_cuda_batch_size(
    *,
    current_batch_size: int,
    baseline_allocated_bytes: int,
    peak_allocated_bytes: int,
    free_before_bytes: int,
    total_bytes: int,
    maximum_batch_size: int,
    reserve_fraction: float = 0.10,
    reserve_bytes: int = 2 * 1024**3,
    overhead_factor: float = 1.25,
) -> int:
    """Estimate a guarded batch ceiling from one successful CUDA render.

    The estimate preserves both a fractional and absolute free-memory reserve
    and pads the observed per-item working set for allocator/nonlinear effects.
    It never recommends less than the already successful batch.
    """

    if current_batch_size < 1 or maximum_batch_size < current_batch_size:
        raise ValueError("invalid current/maximum batch sizes")
    if min(baseline_allocated_bytes, peak_allocated_bytes, free_before_bytes, total_bytes) < 0:
        raise ValueError("CUDA memory measurements must be non-negative")
    if peak_allocated_bytes < baseline_allocated_bytes:
        raise ValueError("peak CUDA allocation cannot be below the baseline")
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must lie in [0, 1)")
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be non-negative")
    if not math.isfinite(overhead_factor) or overhead_factor < 1:
        raise ValueError("overhead_factor must be finite and at least one")

    working_bytes = peak_allocated_bytes - baseline_allocated_bytes
    if working_bytes <= 0:
        return current_batch_size
    per_item_bytes = working_bytes / current_batch_size
    reserve = max(int(total_bytes * reserve_fraction), int(reserve_bytes))
    usable_free = max(0, int(free_before_bytes) - reserve)
    estimated = int(usable_free // (per_item_bytes * overhead_factor))
    return min(maximum_batch_size, max(current_batch_size, estimated))


@dataclass(slots=True)
class AdaptiveBatchSizer:
    """Learn the largest guarded render batch with bounded binary backoff."""

    initial_batch_size: int
    maximum_batch_size: int
    candidate: int = field(init=False)
    largest_success: int = field(default=0, init=False)
    smallest_failure: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.initial_batch_size < 1:
            raise ValueError("initial_batch_size must be positive")
        if self.maximum_batch_size < self.initial_batch_size:
            raise ValueError("maximum_batch_size must be at least initial_batch_size")
        self.candidate = self.initial_batch_size

    @property
    def tuned_batch_size(self) -> int:
        return max(1, self.largest_success or self.candidate)

    def next_batch_size(self, remaining: int) -> int:
        if remaining < 1:
            raise ValueError("remaining must be positive")
        return min(self.candidate, remaining)

    def record_oom(self, attempted_batch_size: int) -> int:
        if attempted_batch_size < 1:
            raise ValueError("attempted_batch_size must be positive")
        self.smallest_failure = (
            attempted_batch_size
            if self.smallest_failure is None
            else min(self.smallest_failure, attempted_batch_size)
        )
        high = self.smallest_failure - 1
        if self.largest_success:
            if high <= self.largest_success:
                self.candidate = self.largest_success
            else:
                self.candidate = (self.largest_success + high + 1) // 2
        else:
            self.candidate = max(1, (attempted_batch_size + 1) // 2)
        return self.candidate

    def record_success(
        self,
        successful_batch_size: int,
        *,
        safe_ceiling_hint: int,
    ) -> int:
        if successful_batch_size < 1:
            raise ValueError("successful_batch_size must be positive")
        self.largest_success = max(self.largest_success, successful_batch_size)
        high = min(self.maximum_batch_size, max(self.largest_success, safe_ceiling_hint))
        if self.smallest_failure is not None:
            high = min(high, self.smallest_failure - 1)
            if high > self.largest_success:
                self.candidate = (self.largest_success + high + 1) // 2
            else:
                self.candidate = self.largest_success
        else:
            self.candidate = high
        return self.candidate

    def report(self) -> dict[str, int | None]:
        return {
            "candidate": self.candidate,
            "largest_success": self.largest_success,
            "smallest_failure": self.smallest_failure,
            "maximum": self.maximum_batch_size,
        }


@dataclass(frozen=True, slots=True)
class FluxFeatureGroup:
    """One representative transformer block for a CHIMERA depth group."""

    name: GroupName
    stream: Literal["double", "single"]
    index: int
    combined_depth: int
    module: Any = field(repr=False, compare=False)

    @property
    def label(self) -> str:
        prefix = "transformer_blocks" if self.stream == "double" else "single_transformer_blocks"
        return f"{prefix}.{self.index}"


def select_flux_feature_groups(transformer: Any) -> tuple[FluxFeatureGroup, ...]:
    """Map CHIMERA's three scale groups to FLUX transformer depth thirds."""

    double = getattr(transformer, "transformer_blocks", None)
    single = getattr(transformer, "single_transformer_blocks", None)
    if double is None or single is None:
        raise TypeError(
            "CHIMERA requires Flux2Transformer2DModel-style transformer_blocks "
            "and single_transformer_blocks"
        )
    double_count = len(double)
    single_count = len(single)
    total = double_count + single_count
    if total < 3:
        raise ValueError("CHIMERA requires at least three transformer blocks")

    # Centers of the three depth thirds avoid both the input/output projections
    # and make the selection stable across 4B/9B layer counts.
    combined_indices = [
        min(total - 1, max(0, int(round((total - 1) * fraction))))
        for fraction in (1.0 / 6.0, 0.5, 5.0 / 6.0)
    ]
    groups: list[FluxFeatureGroup] = []
    for name, combined in zip(CHIMERA_GROUPS, combined_indices, strict=True):
        if combined < double_count:
            groups.append(
                FluxFeatureGroup(name, "double", combined, combined, double[combined])
            )
        else:
            index = combined - double_count
            groups.append(
                FluxFeatureGroup(name, "single", index, combined, single[index])
            )
    if len({id(group.module) for group in groups}) != 3:
        raise ValueError("representative CHIMERA feature groups must be distinct")
    return tuple(groups)


def flux_depth_ltm(step_index: int, step_count: int) -> GroupName:
    """Return the legacy fixed-third coarse-to-fine LTM approximation."""

    if step_count < 1:
        raise ValueError("step_count must be positive")
    if not 0 <= step_index < step_count:
        raise IndexError("step_index is outside the schedule")
    progress = (step_index + 0.5) / step_count
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def _finite_descriptor(values: Sequence[float] | Tensor, *, bands: int) -> Tensor:
    descriptor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if descriptor.numel() != bands:
        raise ValueError(f"frequency descriptor must contain exactly {bands} bands")
    if not bool(torch.isfinite(descriptor).all().item()):
        raise ValueError("frequency descriptors must be finite")
    if bool((descriptor < 0).any().item()):
        raise ValueError("frequency descriptors must be non-negative")
    return descriptor


def _normalized_descriptor(values: Sequence[float] | Tensor, *, bands: int) -> Tensor:
    descriptor = _finite_descriptor(values, bands=bands)
    total = descriptor.sum()
    if float(total.item()) <= torch.finfo(descriptor.dtype).eps:
        raise ValueError("frequency descriptor must contain positive spectral energy")
    return descriptor / total


def _smooth_timestep_descriptors(values: Tensor, *, radius: int) -> Tensor:
    if values.ndim != 2:
        raise ValueError("timestep descriptor matrix must have shape (steps, bands)")
    if radius < 0:
        raise ValueError("timestep smoothing radius must be non-negative")
    if radius == 0 or values.shape[0] == 1:
        return values.clone()
    smoothed = torch.stack(
        [
            values[max(0, index - radius) : min(values.shape[0], index + radius + 1)].mean(
                dim=0
            )
            for index in range(values.shape[0])
        ]
    )
    return smoothed / torch.clamp(
        smoothed.sum(dim=1, keepdim=True),
        min=torch.finfo(smoothed.dtype).eps,
    )


def match_ltm_prototypes(
    layer_prototypes: Mapping[GroupName, Sequence[float] | Tensor],
    timestep_prototypes: Sequence[Sequence[float] | Tensor],
    *,
    bands: int,
) -> tuple[GroupName, ...]:
    """Apply CHIMERA's per-timestep L1 frequency-prototype argmin."""

    if bands < 2:
        raise ValueError("bands must be at least two")
    if set(layer_prototypes) != set(CHIMERA_GROUPS):
        raise ValueError("layer prototypes must contain early, middle, and late")
    layers = {
        group: _finite_descriptor(layer_prototypes[group], bands=bands)
        for group in CHIMERA_GROUPS
    }
    if not timestep_prototypes:
        raise ValueError("at least one timestep prototype is required")
    mapping: list[GroupName] = []
    for values in timestep_prototypes:
        timestep = _finite_descriptor(values, bands=bands)
        distances = {
            group: float(torch.sum(torch.abs(layers[group] - timestep)).item())
            for group in CHIMERA_GROUPS
        }
        # Tuple ordering makes exact ties deterministic and preserves the
        # declared early -> middle -> late group order.
        mapping.append(
            min(CHIMERA_GROUPS, key=lambda group: (distances[group], CHIMERA_GROUPS.index(group)))
        )
    return tuple(mapping)


def match_monotonic_ltm_prototypes(
    layer_prototypes: Mapping[GroupName, Sequence[float] | Tensor],
    timestep_prototypes: Sequence[Sequence[float] | Tensor],
    *,
    bands: int,
    minimum_group_fraction: float = LTM_MINIMUM_GROUP_FRACTION,
) -> tuple[GroupName, ...]:
    """Find the lowest-cost contiguous coarse-to-fine FLUX depth schedule.

    Independent FFT argmins can oscillate between transformer depths even
    though diffusion denoising progresses from coarse structure to fine
    detail.  This constrained fit chooses two transition points while
    requiring a small, non-zero dwell in each depth group.
    """

    if not 0.0 <= minimum_group_fraction < 1.0 / 3.0:
        raise ValueError("minimum_group_fraction must lie in [0, 1/3)")
    if bands < 2:
        raise ValueError("bands must be at least two")
    if set(layer_prototypes) != set(CHIMERA_GROUPS):
        raise ValueError("layer prototypes must contain early, middle, and late")
    if not timestep_prototypes:
        raise ValueError("at least one timestep prototype is required")

    step_count = len(timestep_prototypes)
    if step_count < 3:
        return tuple(flux_depth_ltm(index, step_count) for index in range(step_count))
    layers = {
        group: _finite_descriptor(layer_prototypes[group], bands=bands)
        for group in CHIMERA_GROUPS
    }
    distances = torch.stack(
        [
            torch.stack(
                [
                    torch.sum(
                        torch.abs(
                            layers[group]
                            - _finite_descriptor(values, bands=bands)
                        )
                    )
                    for group in CHIMERA_GROUPS
                ]
            )
            for values in timestep_prototypes
        ]
    )
    prefix = torch.cat(
        [
            torch.zeros(1, len(CHIMERA_GROUPS), dtype=distances.dtype),
            torch.cumsum(distances, dim=0),
        ],
        dim=0,
    )
    minimum_dwell = max(1, int(math.ceil(step_count * minimum_group_fraction)))
    candidates: list[tuple[float, int, int]] = []
    for first_transition in range(
        minimum_dwell,
        step_count - 2 * minimum_dwell + 1,
    ):
        for second_transition in range(
            first_transition + minimum_dwell,
            step_count - minimum_dwell + 1,
        ):
            cost = (
                prefix[first_transition, 0]
                + (prefix[second_transition, 1] - prefix[first_transition, 1])
                + (prefix[step_count, 2] - prefix[second_transition, 2])
            )
            candidates.append(
                (float(cost.item()), first_transition, second_transition)
            )
    if not candidates:
        return tuple(flux_depth_ltm(index, step_count) for index in range(step_count))
    _, first_transition, second_transition = min(candidates)
    return (
        ("early",) * first_transition
        + ("middle",) * (second_transition - first_transition)
        + ("late",) * (step_count - second_transition)
    )


def ltm_mapping_report(mapping: Sequence[GroupName]) -> dict[str, Any]:
    """Summarize depth coverage and coarse-to-fine ordering."""

    values = tuple(mapping)
    if not values:
        raise ValueError("LTM mapping must not be empty")
    if any(group not in CHIMERA_GROUPS for group in values):
        raise ValueError("LTM mapping contains an unknown feature group")
    indices = [CHIMERA_GROUPS.index(group) for group in values]
    counts = {group: values.count(group) for group in CHIMERA_GROUPS}
    return {
        "step_count": len(values),
        "group_counts": counts,
        "groups_used": sum(count > 0 for count in counts.values()),
        "monotonic_coarse_to_fine": all(
            left <= right for left, right in zip(indices, indices[1:])
        ),
        "starts_with": values[0],
        "ends_with": values[-1],
    }


@dataclass(frozen=True, slots=True)
class LTMCalibration:
    """Immutable FFT prototypes and the resulting FLUX layer lookup."""

    bands: int
    sample_count: int
    group_modules: tuple[str, str, str]
    layer_prototypes: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    timestep_prototypes: tuple[tuple[float, ...], ...]
    mapping: tuple[GroupName, ...]
    descriptor_normalized: bool = False
    version: int = 1
    mapping_strategy: str = "independent"
    independent_mapping: tuple[GroupName, ...] = ()
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_modules",
            tuple(str(value) for value in self.group_modules),
        )
        object.__setattr__(
            self,
            "layer_prototypes",
            tuple(
                tuple(float(value) for value in descriptor)
                for descriptor in self.layer_prototypes
            ),
        )
        object.__setattr__(
            self,
            "timestep_prototypes",
            tuple(
                tuple(float(value) for value in descriptor)
                for descriptor in self.timestep_prototypes
            ),
        )
        object.__setattr__(self, "mapping", tuple(self.mapping))
        object.__setattr__(self, "independent_mapping", tuple(self.independent_mapping))
        if self.version not in {1, LTM_CALIBRATION_VERSION}:
            raise ValueError("unsupported LTM calibration version")
        if self.bands < 2:
            raise ValueError("LTM calibration bands must be at least two")
        if self.sample_count < 1:
            raise ValueError("LTM calibration requires at least one sample")
        if len(self.group_modules) != len(CHIMERA_GROUPS):
            raise ValueError("LTM calibration must identify all feature-group modules")
        if len(self.layer_prototypes) != len(CHIMERA_GROUPS):
            raise ValueError("LTM calibration must contain three layer prototypes")
        layers = {
            group: _finite_descriptor(values, bands=self.bands)
            for group, values in zip(CHIMERA_GROUPS, self.layer_prototypes, strict=True)
        }
        if not self.timestep_prototypes:
            raise ValueError("LTM calibration must contain timestep prototypes")
        for values in self.timestep_prototypes:
            _finite_descriptor(values, bands=self.bands)
        independent = match_ltm_prototypes(
            layers,
            self.timestep_prototypes,
            bands=self.bands,
        )
        if not self.independent_mapping:
            object.__setattr__(self, "independent_mapping", independent)
        elif tuple(self.independent_mapping) != independent:
            raise ValueError("stored independent LTM mapping does not match its prototypes")
        if self.mapping_strategy == "independent":
            expected = independent
        elif self.mapping_strategy == "fft_monotonic":
            expected = match_monotonic_ltm_prototypes(
                layers,
                self.timestep_prototypes,
                bands=self.bands,
            )
        elif self.mapping_strategy == "fixed_coarse_to_fine_fallback":
            expected = tuple(
                flux_depth_ltm(index, len(self.timestep_prototypes))
                for index in range(len(self.timestep_prototypes))
            )
            if not self.fallback_reason:
                raise ValueError("fallback LTM calibration must record its reason")
        else:
            raise ValueError(f"unsupported LTM mapping strategy {self.mapping_strategy!r}")
        if tuple(self.mapping) != expected:
            raise ValueError("LTM mapping does not match its stored strategy and prototypes")

    @property
    def step_count(self) -> int:
        return len(self.mapping)

    @property
    def group_module_map(self) -> dict[GroupName, str]:
        return dict(zip(CHIMERA_GROUPS, self.group_modules, strict=True))

    @property
    def layer_prototype_map(self) -> dict[GroupName, tuple[float, ...]]:
        return dict(zip(CHIMERA_GROUPS, self.layer_prototypes, strict=True))

    @property
    def mapping_report(self) -> dict[str, Any]:
        return ltm_mapping_report(self.mapping)

    @property
    def independent_mapping_report(self) -> dict[str, Any]:
        return ltm_mapping_report(self.independent_mapping)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def group_for_step(self, step_index: int) -> GroupName:
        if not 0 <= step_index < self.step_count:
            raise IndexError("step_index is outside the calibrated LTM schedule")
        return self.mapping[step_index]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": self.version,
            "bands": self.bands,
            "sample_count": self.sample_count,
            "descriptor_normalized": self.descriptor_normalized,
            "group_modules": self.group_module_map,
            "layer_prototypes": {
                group: list(values)
                for group, values in self.layer_prototype_map.items()
            },
            "timestep_prototypes": [list(values) for values in self.timestep_prototypes],
            "mapping": list(self.mapping),
        }
        if self.version >= LTM_CALIBRATION_VERSION:
            payload.update(
                {
                    "mapping_strategy": self.mapping_strategy,
                    "independent_mapping": list(self.independent_mapping),
                    "fallback_reason": self.fallback_reason,
                }
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LTMCalibration":
        version = int(payload.get("version", 0))
        if version not in {1, LTM_CALIBRATION_VERSION}:
            raise ValueError("unsupported LTM calibration version")
        bands = int(payload["bands"])
        modules = payload["group_modules"]
        layers = payload["layer_prototypes"]
        return cls(
            bands=bands,
            sample_count=int(payload["sample_count"]),
            descriptor_normalized=bool(payload.get("descriptor_normalized", False)),
            group_modules=tuple(str(modules[group]) for group in CHIMERA_GROUPS),
            layer_prototypes=tuple(
                tuple(float(value) for value in layers[group])
                for group in CHIMERA_GROUPS
            ),
            timestep_prototypes=tuple(
                tuple(float(value) for value in values)
                for values in payload["timestep_prototypes"]
            ),
            mapping=tuple(str(group) for group in payload["mapping"]),
            version=version,
            mapping_strategy=str(payload.get("mapping_strategy", "independent")),
            independent_mapping=tuple(
                str(group) for group in payload.get("independent_mapping", ())
            ),
            fallback_reason=(
                str(payload["fallback_reason"])
                if payload.get("fallback_reason") is not None
                else None
            ),
        )


class LTMPrototypeAccumulator:
    """Accumulate pair-independent group/timestep FFT prototype statistics."""

    def __init__(
        self,
        *,
        step_count: int,
        group_modules: Mapping[GroupName, str],
        bands: int = 16,
    ) -> None:
        if step_count < 1:
            raise ValueError("step_count must be positive")
        if bands < 2:
            raise ValueError("bands must be at least two")
        if set(group_modules) != set(CHIMERA_GROUPS):
            raise ValueError("group_modules must identify all CHIMERA groups")
        self.step_count = int(step_count)
        self.bands = int(bands)
        self.group_modules = {group: str(group_modules[group]) for group in CHIMERA_GROUPS}
        self._sums = torch.zeros(
            len(CHIMERA_GROUPS),
            self.step_count,
            self.bands,
            dtype=torch.float64,
        )
        self._counts = torch.zeros(
            len(CHIMERA_GROUPS),
            self.step_count,
            dtype=torch.int64,
        )
        self._timestep_sums = torch.zeros(
            self.step_count,
            self.bands,
            dtype=torch.float64,
        )
        self._timestep_counts = torch.zeros(
            self.step_count,
            dtype=torch.int64,
        )

    def add(self, group: GroupName, step_index: int, descriptor: Tensor) -> None:
        if group not in CHIMERA_GROUPS:
            raise ValueError(f"unknown CHIMERA group {group!r}")
        if not 0 <= step_index < self.step_count:
            raise IndexError("step_index is outside the calibration schedule")
        values = _normalized_descriptor(
            descriptor.detach().to("cpu"),
            bands=self.bands,
        )
        group_index = CHIMERA_GROUPS.index(group)
        self._sums[group_index, step_index] += values
        self._counts[group_index, step_index] += 1

    def add_timestep(self, step_index: int, descriptor: Tensor) -> None:
        """Accumulate a dedicated model-output spectrum for one timestep."""

        if not 0 <= step_index < self.step_count:
            raise IndexError("step_index is outside the calibration schedule")
        values = _normalized_descriptor(
            descriptor.detach().to("cpu"),
            bands=self.bands,
        )
        self._timestep_sums[step_index] += values
        self._timestep_counts[step_index] += 1

    def finalize(self) -> LTMCalibration:
        if bool((self._counts == 0).any().item()):
            missing = int((self._counts == 0).sum().item())
            raise RuntimeError(f"LTM calibration is incomplete ({missing} group/timestep cells missing)")
        if bool((self._timestep_counts == 0).any().item()):
            missing = int((self._timestep_counts == 0).sum().item())
            raise RuntimeError(
                f"LTM timestep calibration is incomplete ({missing} timesteps missing)"
            )
        unique_counts = torch.unique(self._counts)
        if unique_counts.numel() != 1:
            raise RuntimeError("LTM calibration samples are imbalanced across groups or timesteps")
        sample_count = int(unique_counts.item())
        if not bool((self._timestep_counts == sample_count).all().item()):
            raise RuntimeError("LTM layer and timestep calibration sample counts disagree")
        cell_means = self._sums / self._counts.unsqueeze(-1)
        layer_values = cell_means.mean(dim=1)
        timestep_values = self._timestep_sums / self._timestep_counts.unsqueeze(-1)
        timestep_values = _smooth_timestep_descriptors(
            timestep_values,
            radius=LTM_TIMESTEP_SMOOTHING_RADIUS,
        )
        layer_map = {
            group: layer_values[index]
            for index, group in enumerate(CHIMERA_GROUPS)
        }
        independent_mapping = match_ltm_prototypes(
            layer_map,
            tuple(timestep_values[index] for index in range(self.step_count)),
            bands=self.bands,
        )
        independent_report = ltm_mapping_report(independent_mapping)
        if independent_report["groups_used"] < 2:
            mapping = tuple(
                flux_depth_ltm(index, self.step_count)
                for index in range(self.step_count)
            )
            mapping_strategy = "fixed_coarse_to_fine_fallback"
            fallback_reason = (
                "independent FFT argmin collapsed to "
                f"{independent_report['groups_used']} of {len(CHIMERA_GROUPS)} groups"
            )
        else:
            mapping = match_monotonic_ltm_prototypes(
                layer_map,
                tuple(timestep_values[index] for index in range(self.step_count)),
                bands=self.bands,
            )
            mapping_strategy = "fft_monotonic"
            fallback_reason = None
        return LTMCalibration(
            bands=self.bands,
            sample_count=sample_count,
            group_modules=tuple(self.group_modules[group] for group in CHIMERA_GROUPS),
            layer_prototypes=tuple(
                tuple(float(value) for value in layer_values[index].tolist())
                for index in range(len(CHIMERA_GROUPS))
            ),
            timestep_prototypes=tuple(
                tuple(float(value) for value in timestep_values[index].tolist())
                for index in range(self.step_count)
            ),
            mapping=mapping,
            descriptor_normalized=True,
            version=LTM_CALIBRATION_VERSION,
            mapping_strategy=mapping_strategy,
            independent_mapping=independent_mapping,
            fallback_reason=fallback_reason,
        )


def resolve_ltm_group(
    step_index: int,
    step_count: int,
    calibration: LTMCalibration | None,
) -> GroupName:
    """Resolve a calibrated LTM group or explicitly use the linear fallback."""

    if calibration is None:
        return flux_depth_ltm(step_index, step_count)
    if calibration.step_count != step_count:
        raise ValueError(
            "LTM calibration schedule length does not match the active diffusion schedule"
        )
    return calibration.group_for_step(step_index)


def map_denoising_to_inversion_step(
    denoising_index: int,
    *,
    denoising_steps: int,
    inversion_steps: int,
) -> int:
    """Linearly map a denoising index to its corresponding inversion index."""

    if denoising_steps < 1 or inversion_steps < 1:
        raise ValueError("step counts must be positive")
    if not 0 <= denoising_index < denoising_steps:
        raise IndexError("denoising_index is outside the schedule")
    if denoising_steps == 1 or inversion_steps == 1:
        return 0
    position = denoising_index * (inversion_steps - 1) / (denoising_steps - 1)
    return min(inversion_steps - 1, max(0, int(round(position))))


def nearest_cached_step(
    requested_step: int,
    available_steps: Sequence[int],
) -> int:
    """Resolve cache-stride gaps deterministically, preferring the earlier step."""

    if not available_steps:
        raise ValueError("available_steps must not be empty")
    return min((int(step) for step in available_steps), key=lambda step: (abs(step - requested_step), step))


@dataclass(frozen=True, slots=True)
class StoredFeature:
    """One CPU-resident feature, optionally symmetric-int8 quantized."""

    values: Tensor
    scale: Tensor
    storage: CacheStorage

    @classmethod
    def from_tensor(cls, tensor: Tensor, storage: CacheStorage) -> "StoredFeature":
        source = tensor.detach().to("cpu")
        if storage == "int8":
            maximum = source.float().abs().amax()
            scale = torch.clamp(maximum / 127.0, min=torch.finfo(torch.float32).tiny)
            values = torch.round(source.float() / scale).clamp(-127, 127).to(torch.int8)
            return cls(values.contiguous(), scale.reshape(()).cpu(), storage)
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[storage]
        return cls(source.to(dtype=dtype).contiguous(), torch.ones((), dtype=torch.float32), storage)

    def materialize(self, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
        values = self.values.to(device=device)
        if self.storage == "int8":
            return (values.to(torch.float32) * self.scale.to(device=device)).to(dtype=dtype)
        return values.to(dtype=dtype)

    @property
    def storage_bytes(self) -> int:
        return self.values.numel() * self.values.element_size() + self.scale.numel() * self.scale.element_size()


@dataclass(frozen=True, slots=True)
class ChimeraEndpointCache:
    """Inverted endpoint latent plus its sparse representative feature cache."""

    key: str
    inverted_latent: Tensor
    features: Mapping[GroupName, Mapping[int, StoredFeature]]
    inversion_steps: int
    image_token_count: int
    group_modules: Mapping[GroupName, str]

    def __post_init__(self) -> None:
        if self.inverted_latent.ndim != 3 or self.inverted_latent.shape[0] != 1:
            raise ValueError("inverted_latent must have shape (1, image_tokens, channels)")
        if self.inversion_steps < 1:
            raise ValueError("inversion_steps must be positive")
        if self.image_token_count != self.inverted_latent.shape[1]:
            raise ValueError("image_token_count disagrees with inverted_latent")
        if not any(self.features.get(group) for group in CHIMERA_GROUPS):
            raise ValueError("endpoint cache must contain at least one feature")

    def feature(self, group: GroupName, requested_step: int) -> StoredFeature:
        group_features = self.features.get(group, {})
        if not group_features:
            raise KeyError(f"endpoint cache contains no {group!r} features")
        resolved = nearest_cached_step(requested_step, tuple(group_features))
        return group_features[resolved]

    @property
    def storage_bytes(self) -> int:
        feature_bytes = sum(
            stored.storage_bytes
            for group_features in self.features.values()
            for stored in group_features.values()
        )
        latent = self.inverted_latent
        return feature_bytes + latent.numel() * latent.element_size()


def _replace_image_tensor(output: Any, tensor: Tensor) -> Any:
    if isinstance(output, tuple):
        if len(output) < 2:
            raise TypeError("double-stream Flux block output must contain image features")
        values = list(output)
        values[-1] = tensor
        return tuple(values)
    if isinstance(output, list):
        if len(output) < 2:
            raise TypeError("double-stream Flux block output must contain image features")
        values = list(output)
        values[-1] = tensor
        return values
    if isinstance(output, Tensor):
        return tensor
    raise TypeError(f"unsupported Flux block output {type(output).__name__}")


def _output_image_tensor(output: Any, image_token_count: int) -> Tensor:
    tensor = output[-1] if isinstance(output, (tuple, list)) else output
    if not isinstance(tensor, Tensor) or tensor.ndim != 3:
        raise TypeError("Flux feature hooks require a rank-three tensor output")
    if tensor.shape[1] < image_token_count:
        raise ValueError("Flux feature contains fewer tokens than the image latent")
    return tensor[:, -image_token_count:, :]


def _batch_slerp(a: Tensor, b: Tensor, alphas: Tensor) -> Tensor:
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != 1:
        raise ValueError("cached feature endpoints must share shape (1, tokens, channels)")
    values = []
    for amount in alphas.detach().to("cpu", dtype=torch.float64).tolist():
        values.append(slerp(a, b, float(amount)))
    return torch.cat(values, dim=0)


class FluxFeatureController:
    """Forward-hook controller for endpoint capture and ACI residual injection."""

    def __init__(
        self,
        transformer: Any,
        *,
        image_token_count: int,
        storage: CacheStorage = "int8",
    ) -> None:
        if image_token_count < 1:
            raise ValueError("image_token_count must be positive")
        self.groups = select_flux_feature_groups(transformer)
        self.image_token_count = int(image_token_count)
        self.storage = storage
        self._handles: list[Any] = []
        self._mode: Literal["idle", "calibrate", "capture", "inject"] = "idle"
        self._calibration_accumulator: LTMPrototypeAccumulator | None = None
        self._calibration_step: int | None = None
        self._calibration_channel_chunk_size = 128
        self._capture_key: str | None = None
        self._capture_step: int | None = None
        self._capture_group: GroupName | None = None
        self._captured: dict[str, dict[GroupName, dict[int, StoredFeature]]] = {}
        self._source: ChimeraEndpointCache | None = None
        self._target: ChimeraEndpointCache | None = None
        self._inject_step: int | None = None
        self._inject_group: GroupName | None = None
        self._inject_alphas: Tensor | None = None
        self._inject_weight = 0.0

    def __enter__(self) -> "FluxFeatureController":
        if self._handles:
            raise RuntimeError("FluxFeatureController is already installed")
        for group in self.groups:
            handle = group.module.register_forward_hook(self._make_hook(group.name))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._mode = "idle"
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, group: GroupName):
        def hook(module, inputs, output):
            del module, inputs
            if self._mode == "calibrate":
                assert self._calibration_accumulator is not None
                assert self._calibration_step is not None
                feature = _output_image_tensor(output, self.image_token_count)
                if feature.shape[0] != 1:
                    raise ValueError("LTM calibration requires feature batch size one")
                descriptor = radial_frequency_descriptor(
                    feature,
                    bands=self._calibration_accumulator.bands,
                    channel_chunk_size=self._calibration_channel_chunk_size,
                    normalize=True,
                )
                self._calibration_accumulator.add(
                    group,
                    self._calibration_step,
                    descriptor,
                )
                return output
            if self._mode == "capture" and self._capture_group == group:
                assert self._capture_key is not None and self._capture_step is not None
                feature = _output_image_tensor(output, self.image_token_count)
                if feature.shape[0] != 1:
                    raise ValueError("endpoint inversion cache capture requires batch size one")
                self._captured.setdefault(self._capture_key, {}).setdefault(group, {})[
                    self._capture_step
                ] = StoredFeature.from_tensor(feature, self.storage)
                return output
            if self._mode != "inject" or self._inject_group != group:
                return output
            assert self._source is not None and self._target is not None
            assert self._inject_step is not None and self._inject_alphas is not None
            image = _output_image_tensor(output, self.image_token_count)
            source = self._source.feature(group, self._inject_step).materialize(
                device=image.device,
                dtype=image.dtype,
            )
            target = self._target.feature(group, self._inject_step).materialize(
                device=image.device,
                dtype=image.dtype,
            )
            alphas = self._inject_alphas
            if image.shape[0] == 2 * alphas.numel():
                # predict_cfg_velocity concatenates [conditional, unconditional].
                alphas = alphas.repeat(2)
            elif image.shape[0] != alphas.numel():
                raise ValueError(
                    "ACI alpha batch does not match transformer feature batch: "
                    f"{alphas.numel()} versus {image.shape[0]}"
                )
            cached = _batch_slerp(source, target, alphas).to(
                device=image.device,
                dtype=image.dtype,
            )
            updated_image = image + self._inject_weight * cached
            tensor = output[-1] if isinstance(output, (tuple, list)) else output
            updated = tensor.clone()
            updated[:, -self.image_token_count :, :] = updated_image
            return _replace_image_tensor(output, updated)

        return hook

    @contextmanager
    def calibrate(
        self,
        *,
        step: int,
        accumulator: LTMPrototypeAccumulator,
        channel_chunk_size: int = 128,
    ):
        if self._mode != "idle":
            raise RuntimeError("nested CHIMERA feature-controller modes are not allowed")
        if channel_chunk_size < 1:
            raise ValueError("channel_chunk_size must be positive")
        self._mode = "calibrate"
        self._calibration_accumulator = accumulator
        self._calibration_step = int(step)
        self._calibration_channel_chunk_size = int(channel_chunk_size)
        try:
            yield
        finally:
            self._mode = "idle"
            self._calibration_accumulator = None
            self._calibration_step = None
            self._calibration_channel_chunk_size = 128

    @contextmanager
    def capture(self, *, key: str, step: int, group: GroupName):
        if self._mode != "idle":
            raise RuntimeError("nested CHIMERA feature-controller modes are not allowed")
        self._mode = "capture"
        self._capture_key = str(key)
        self._capture_step = int(step)
        self._capture_group = group
        try:
            yield
        finally:
            self._mode = "idle"
            self._capture_key = None
            self._capture_step = None
            self._capture_group = None

    @contextmanager
    def inject(
        self,
        *,
        source: ChimeraEndpointCache,
        target: ChimeraEndpointCache,
        inversion_step: int,
        group: GroupName,
        alphas: Tensor,
        weight: float,
    ):
        if self._mode != "idle":
            raise RuntimeError("nested CHIMERA feature-controller modes are not allowed")
        self._mode = "inject"
        self._source = source
        self._target = target
        self._inject_step = int(inversion_step)
        self._inject_group = group
        self._inject_alphas = torch.as_tensor(alphas, dtype=torch.float64).reshape(-1)
        self._inject_weight = float(weight)
        try:
            yield
        finally:
            self._mode = "idle"
            self._source = None
            self._target = None
            self._inject_step = None
            self._inject_group = None
            self._inject_alphas = None
            self._inject_weight = 0.0

    def endpoint_cache(
        self,
        *,
        key: str,
        inverted_latent: Tensor,
        inversion_steps: int,
    ) -> ChimeraEndpointCache:
        features = self._captured.get(key)
        if not features:
            raise KeyError(f"no CHIMERA features were captured for {key!r}")
        return ChimeraEndpointCache(
            key=key,
            inverted_latent=inverted_latent.detach().to("cpu", dtype=torch.float32),
            features={
                group: dict(group_features)
                for group, group_features in features.items()
            },
            inversion_steps=inversion_steps,
            image_token_count=self.image_token_count,
            group_modules={group.name: group.label for group in self.groups},
        )


def calibrate_flux_ltm(
    *,
    endpoint_samples: Sequence[tuple[Tensor, ConditioningPackage]],
    schedule: FlowSchedule,
    transformer: Any,
    image_ids: Tensor,
    bands: int = 16,
    channel_chunk_size: int = 128,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> LTMCalibration:
    """Derive CHIMERA's FFT layer/timestep lookup from calibration endpoints.

    Only radial descriptors are retained.  Full transformer features are never
    stored during calibration, keeping host memory independent of sample count.
    """

    if not endpoint_samples:
        raise ValueError("LTM calibration requires at least one endpoint sample")
    if schedule.num_inference_steps < 2:
        raise ValueError("LTM calibration requires at least two scheduler points")
    if bands < 2:
        raise ValueError("bands must be at least two")
    if channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be positive")
    first_latent = endpoint_samples[0][0]
    if first_latent.ndim != 3 or first_latent.shape[0] != 1:
        raise ValueError(
            "LTM calibration samples must have latent shape (1, image_tokens, channels)"
        )
    image_token_count = int(first_latent.shape[1])
    controller = FluxFeatureController(
        transformer,
        image_token_count=image_token_count,
        storage="float32",
    )
    accumulator = LTMPrototypeAccumulator(
        step_count=schedule.num_inference_steps,
        group_modules={group.name: group.label for group in controller.groups},
        bands=bands,
    )
    with controller, torch.inference_mode():
        for sample_index, (clean_latent, conditioning) in enumerate(endpoint_samples):
            if clean_latent.ndim != 3 or clean_latent.shape[0] != 1:
                raise ValueError(
                    f"LTM calibration sample {sample_index} must have latent shape "
                    "(1, image_tokens, channels)"
                )
            if clean_latent.shape[1] != image_token_count:
                raise ValueError("all LTM calibration samples must use the same image geometry")
            state = clean_latent.detach().clone()
            for schedule_index in reversed(range(schedule.num_inference_steps)):
                with controller.calibrate(
                    step=schedule_index,
                    accumulator=accumulator,
                    channel_chunk_size=channel_chunk_size,
                ):
                    velocity = predict_conditional_velocity(
                        transformer,
                        state,
                        schedule.timesteps[schedule_index].to(device=state.device),
                        conditioning,
                        image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                accumulator.add_timestep(
                    schedule_index,
                    radial_frequency_descriptor(
                        velocity,
                        bands=bands,
                        channel_chunk_size=channel_chunk_size,
                        normalize=True,
                    ),
                )
                current_sigma = float(schedule.sigmas[schedule_index + 1].item())
                next_sigma = float(schedule.sigmas[schedule_index].item())
                state = (
                    state.to(torch.float32)
                    + (next_sigma - current_sigma) * velocity.to(torch.float32)
                ).to(dtype=velocity.dtype)
                if not bool(torch.isfinite(state).all().item()):
                    raise FloatingPointError(
                        "LTM calibration inversion produced non-finite values at "
                        f"sample {sample_index}, scheduler index {schedule_index}"
                    )
    return accumulator.finalize()


def invert_endpoint(
    *,
    key: str,
    clean_latent: Tensor,
    schedule: FlowSchedule,
    transformer: Any,
    conditioning: ConditioningPackage,
    image_ids: Tensor,
    controller: FluxFeatureController,
    ltm_calibration: LTMCalibration | None = None,
    cache_stride: int = 1,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> ChimeraEndpointCache:
    """Reverse the native Euler ODE while collecting sparse FLUX features."""

    if clean_latent.ndim != 3 or clean_latent.shape[0] != 1:
        raise ValueError("CHIMERA endpoint inversion requires latent batch size one")
    if schedule.num_inference_steps < 2:
        raise ValueError("CHIMERA inversion requires at least two scheduler points")
    if cache_stride < 1:
        raise ValueError("cache_stride must be positive")
    if controller.image_token_count != clean_latent.shape[1]:
        raise ValueError("feature-controller token count disagrees with the endpoint latent")

    state = clean_latent.detach().clone()
    step_count = schedule.num_inference_steps
    previous_group: GroupName | None = None
    with torch.inference_mode():
        for schedule_index in reversed(range(step_count)):
            group = resolve_ltm_group(schedule_index, step_count, ltm_calibration)
            # A calibrated mapping can switch groups at arbitrary steps.  Cache
            # every transition as well as the configured stride so even a
            # one-step group run always has a retrievable feature.
            group_transition = previous_group is not None and group != previous_group
            capture = (
                schedule_index % cache_stride == 0
                or schedule_index in {0, step_count - 1}
                or group_transition
            )
            if capture:
                context = controller.capture(key=key, step=schedule_index, group=group)
            else:
                context = nullcontext()
            with context:
                velocity = predict_conditional_velocity(
                    transformer,
                    state,
                    schedule.timesteps[schedule_index].to(device=state.device),
                    conditioning,
                    image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            current_sigma = float(schedule.sigmas[schedule_index + 1].item())
            next_sigma = float(schedule.sigmas[schedule_index].item())
            # Reverse of the repository's validated denoising Euler update.
            state = (
                state.to(torch.float32)
                + (next_sigma - current_sigma) * velocity.to(torch.float32)
            ).to(dtype=velocity.dtype)
            if not bool(torch.isfinite(state).all().item()):
                raise FloatingPointError(
                    f"CHIMERA inversion produced non-finite values at scheduler index {schedule_index}"
                )
            previous_group = group
    return controller.endpoint_cache(
        key=key,
        inverted_latent=state,
        inversion_steps=step_count,
    )


def append_anchor_conditioning(
    base: ConditioningPackage,
    anchor: ConditioningPackage,
    *,
    max_anchor_tokens: int | None = None,
) -> ConditioningPackage:
    """Implement SAP by appending anchor tokens to the active text context."""

    if base.prompt_embeds.shape[2] != anchor.prompt_embeds.shape[2]:
        raise ValueError("base and anchor embedding widths must match")
    count = anchor.sequence_length
    if max_anchor_tokens is not None:
        if max_anchor_tokens < 1:
            raise ValueError("max_anchor_tokens must be positive")
        count = min(count, max_anchor_tokens)
    anchor_embeds = anchor.prompt_embeds[:, :count]
    anchor_ids = anchor.text_ids[:, :count]
    batch = base.batch_size
    if anchor_embeds.shape[0] == 1 and batch > 1:
        anchor_embeds = anchor_embeds.expand(batch, -1, -1)
        anchor_ids = anchor_ids.expand(batch, -1, -1)
    elif anchor_embeds.shape[0] != batch:
        raise ValueError("anchor conditioning batch must be one or match the base batch")
    return ConditioningPackage(
        prompt=(f"sap:{base.prompt!r}+{anchor.prompt!r}",),
        prompt_embeds=torch.cat(
            (
                base.prompt_embeds,
                anchor_embeds.to(
                    device=base.prompt_embeds.device,
                    dtype=base.prompt_embeds.dtype,
                ),
            ),
            dim=1,
        ).detach(),
        text_ids=torch.cat(
            (base.text_ids, anchor_ids.to(device=base.text_ids.device)),
            dim=1,
        ).detach(),
    )


def prompt_anchor_reliability(
    anchor: ConditioningPackage,
    endpoint_a: ConditioningPackage,
    endpoint_b: ConditioningPackage,
) -> tuple[float, float, float]:
    """Flux-native pooled-embedding proxy for CHIMERA's CLIP reliability gate."""

    widths = {
        anchor.feature_width,
        endpoint_a.feature_width,
        endpoint_b.feature_width,
    }
    if len(widths) != 1:
        raise ValueError("anchor and endpoint embedding widths must match")

    def pooled(package: ConditioningPackage) -> Tensor:
        return package.prompt_embeds.float().mean(dim=1).mean(dim=0)

    anchor_vector = pooled(anchor)
    a_vector = pooled(endpoint_a).to(anchor_vector.device)
    b_vector = pooled(endpoint_b).to(anchor_vector.device)
    similarity_a = float(torch.nn.functional.cosine_similarity(anchor_vector, a_vector, dim=0).item())
    similarity_b = float(torch.nn.functional.cosine_similarity(anchor_vector, b_vector, dim=0).item())
    return similarity_a, similarity_b, min(similarity_a, similarity_b)


def interpolate_chimera_conditioning(
    source: ConditioningPackage,
    target: ConditioningPackage,
    alpha: float,
    *,
    mode: ConditioningInterpolation = "slerp",
) -> ConditioningPackage:
    """Interpolate CHIMERA text embeddings without midpoint norm collapse.

    The shared conditioning helper intentionally retains linear interpolation
    for older workflows. CHIMERA defaults to a global embedding SLERP, whose
    magnitude follows the linear interpolation of endpoint magnitudes while
    its direction follows the embedding-space great circle.
    """

    linear = interpolate_conditioning(source, target, alpha)
    if mode == "linear":
        return linear
    if mode != "slerp":
        raise ValueError("CHIMERA conditioning interpolation must be linear or slerp")
    target_embeds = target.prompt_embeds.to(
        device=source.prompt_embeds.device,
        dtype=source.prompt_embeds.dtype,
    )
    embeds = slerp_direction_and_magnitude(
        source.prompt_embeds,
        target_embeds,
        alpha,
    ).detach()
    return ConditioningPackage(
        prompt=(f"chimera-slerp:{alpha:.8g}",),
        prompt_embeds=embeds,
        text_ids=linear.text_ids,
    )


def conditioning_interpolation_report(
    source: ConditioningPackage,
    target: ConditioningPackage,
    alpha: float,
    *,
    mode: ConditioningInterpolation = "slerp",
) -> dict[str, float | str]:
    """Return JSON-safe norm diagnostics for one interpolated prompt."""

    linear = interpolate_conditioning(source, target, alpha)
    active = interpolate_chimera_conditioning(source, target, alpha, mode=mode)
    source_norm = float(torch.linalg.vector_norm(source.prompt_embeds.float()).item())
    target_norm = float(torch.linalg.vector_norm(target.prompt_embeds.float()).item())
    expected_norm = (1.0 - alpha) * source_norm + alpha * target_norm
    linear_norm = float(torch.linalg.vector_norm(linear.prompt_embeds.float()).item())
    active_norm = float(torch.linalg.vector_norm(active.prompt_embeds.float()).item())
    denominator = max(expected_norm, torch.finfo(torch.float32).eps)
    return {
        "alpha": float(alpha),
        "mode": mode,
        "source_embedding_norm": source_norm,
        "target_embedding_norm": target_norm,
        "expected_embedding_norm": expected_norm,
        "linear_embedding_norm": linear_norm,
        "linear_norm_retention": linear_norm / denominator,
        "active_embedding_norm": active_norm,
        "active_norm_retention": active_norm / denominator,
    }


def render_chimera_morph(
    source: ChimeraEndpointCache,
    target: ChimeraEndpointCache,
    *,
    schedule: FlowSchedule,
    transformer: Any,
    image_ids: Tensor,
    source_conditioning: ConditioningPackage,
    target_conditioning: ConditioningPackage,
    anchor_conditioning: ConditioningPackage,
    unconditional_conditioning: ConditioningPackage,
    alphas: Sequence[float],
    config: ChimeraConfig,
    ltm_calibration: LTMCalibration | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    diagnostics: list[dict[str, float | str | None]] | None = None,
) -> tuple[RenderedLatentFrame, ...]:
    """Denoise slerped endpoint latents with IDM, ACI, and early-step SAP."""

    if schedule.num_inference_steps != config.denoising_steps:
        raise ValueError("schedule length disagrees with ChimeraConfig.denoising_steps")
    amounts = tuple(float(alpha) for alpha in alphas)
    if not amounts or any(not 0.0 < alpha < 1.0 for alpha in amounts):
        raise ValueError("CHIMERA render alphas must be non-empty and strictly interior")
    if source.image_token_count != target.image_token_count:
        raise ValueError("endpoint caches use different image token counts")
    if config.ltm_mode == "fft":
        if ltm_calibration is None:
            raise ValueError("FFT LTM rendering requires a completed LTM calibration")
        if ltm_calibration.bands != config.ltm_bands:
            raise ValueError("LTM calibration band count disagrees with ChimeraConfig")
        active_ltm = ltm_calibration
    else:
        active_ltm = None

    initial = torch.cat(
        [slerp(source.inverted_latent, target.inverted_latent, alpha) for alpha in amounts],
        dim=0,
    ).to(device=image_ids.device, dtype=torch.float32)
    base_conditionings = [
        interpolate_chimera_conditioning(
            source_conditioning,
            target_conditioning,
            alpha,
            mode=config.conditioning_interpolation,
        )
        for alpha in amounts
    ]
    base_batch = stack_conditioning_packages(base_conditionings).to(initial.device)
    anchor = anchor_conditioning.to(initial.device)
    unconditional = unconditional_conditioning.to(initial.device)
    alpha_tensor = torch.tensor(amounts, dtype=torch.float64)
    sap_steps = int(math.ceil(config.sap_active_ratio * schedule.num_inference_steps))
    cfg_residual_sums = torch.zeros(
        len(amounts),
        device=initial.device,
        dtype=torch.float64,
    )
    cfg_residual_early_sums = torch.zeros_like(cfg_residual_sums)
    cfg_residual_late_sums = torch.zeros_like(cfg_residual_sums)
    cfg_residual_count = 0
    cfg_residual_early_count = 0
    cfg_residual_late_count = 0

    state = initial
    controller = FluxFeatureController(
        transformer,
        image_token_count=source.image_token_count,
        storage=config.cache_storage,
    )
    with controller, torch.inference_mode():
        for denoising_index in range(schedule.num_inference_steps):
            inversion_index = map_denoising_to_inversion_step(
                denoising_index,
                denoising_steps=schedule.num_inference_steps,
                inversion_steps=source.inversion_steps,
            )
            group = resolve_ltm_group(
                inversion_index,
                source.inversion_steps,
                active_ltm,
            )
            sap_active = denoising_index < sap_steps
            conditional = (
                append_anchor_conditioning(
                    base_batch,
                    anchor,
                    max_anchor_tokens=config.anchor_max_tokens,
                )
                if sap_active
                else base_batch
            )
            # Batched CFG requires matching conditional/unconditional token
            # lengths.  SAP deliberately changes only the conditional branch,
            # so its short early phase runs sequential CFG.
            execution = "sequential" if sap_active else config.cfg_execution

            def record_cfg_residual(residual: Tensor) -> None:
                nonlocal cfg_residual_count
                nonlocal cfg_residual_early_count
                nonlocal cfg_residual_late_count
                if residual.shape[0] != len(amounts):
                    raise ValueError("CFG diagnostic batch disagrees with CHIMERA alphas")
                values = (
                    residual.float()
                    .reshape(residual.shape[0], -1)
                    .square()
                    .mean(dim=1)
                    .sqrt()
                    .detach()
                    .to(dtype=torch.float64)
                )
                cfg_residual_sums.add_(values)
                cfg_residual_count += 1
                if sap_active:
                    cfg_residual_early_sums.add_(values)
                    cfg_residual_early_count += 1
                else:
                    cfg_residual_late_sums.add_(values)
                    cfg_residual_late_count += 1

            with controller.inject(
                source=source,
                target=target,
                inversion_step=inversion_index,
                group=group,
                alphas=alpha_tensor,
                weight=config.aci_weight,
            ):
                velocity = predict_cfg_velocity(
                    transformer,
                    state,
                    schedule.timesteps[denoising_index].to(device=state.device),
                    conditional,
                    unconditional,
                    image_ids,
                    guidance_scale=config.guidance_scale,
                    cfg_enabled=config.guidance_scale > 1.0,
                    cfg_execution=execution,
                    joint_attention_kwargs=joint_attention_kwargs,
                    cfg_residual_callback=record_cfg_residual,
                )
            state = euler_flow_update(
                state,
                velocity,
                schedule.sigmas[denoising_index],
                schedule.sigmas[denoising_index + 1],
            )
            if not bool(torch.isfinite(state).all().item()):
                raise FloatingPointError(
                    f"CHIMERA denoising produced non-finite values at step {denoising_index}"
                )

    if diagnostics is not None:
        for index, alpha in enumerate(amounts):
            row: dict[str, float | str | None] = conditioning_interpolation_report(
                source_conditioning,
                target_conditioning,
                alpha,
                mode=config.conditioning_interpolation,
            )
            row.update(
                {
                    "guidance_scale": float(config.guidance_scale),
                    "cfg_residual_rms_mean": (
                        float((cfg_residual_sums[index] / cfg_residual_count).item())
                        if cfg_residual_count
                        else None
                    ),
                    "cfg_residual_rms_sap": (
                        float(
                            (
                                cfg_residual_early_sums[index]
                                / cfg_residual_early_count
                            ).item()
                        )
                        if cfg_residual_early_count
                        else None
                    ),
                    "cfg_residual_rms_post_sap": (
                        float(
                            (
                                cfg_residual_late_sums[index]
                                / cfg_residual_late_count
                            ).item()
                        )
                        if cfg_residual_late_count
                        else None
                    ),
                }
            )
            diagnostics.append(row)

    return tuple(
        RenderedLatentFrame(
            index=index,
            alpha=alpha,
            start_state=initial[index : index + 1].detach().cpu(),
            final_latent=state[index : index + 1].detach().cpu(),
            conditioning_mode=RenderConditioningMode.INTERPOLATED_EMBEDDINGS,
        )
        for index, alpha in enumerate(amounts)
    )


class ChimeraFlux2Session:
    """One-model sequence facade for pairwise CHIMERA inversion and rendering."""

    def __init__(self, runner: FlowMorphRunner, *, config: ChimeraConfig) -> None:
        if not runner._prepared:
            raise PipelineError("ChimeraFlux2Session requires a prepared FlowMorphRunner")
        runner._require_prepared_values()
        if runner.schedule is None or runner.schedule.num_inference_steps != config.denoising_steps:
            raise PipelineError("prepared runner schedule disagrees with CHIMERA denoising steps")
        self.runner = runner
        self.config = config
        self.assets = FlowMorphSequenceSession(
            runner,
            render_batch_size=config.render_batch_size,
            decode_batch_size=config.decode_batch_size,
            cfg_execution=config.cfg_execution,
            oom_backoff=config.oom_backoff,
        )
        self.ltm_calibration: LTMCalibration | None = None
        self.render_batch_sizer = AdaptiveBatchSizer(
            config.render_batch_size,
            config.render_batch_max,
        )
        self.last_render_batch_size = 1
        self.last_conditioning_diagnostics: tuple[
            dict[str, float | str | None], ...
        ] = ()

    @property
    def device(self) -> torch.device:
        return self.assets.device

    @property
    def last_decode_batch_size(self) -> int:
        return self.assets.last_decode_batch_size

    @property
    def render_batch_report(self) -> dict[str, int | None]:
        return self.render_batch_sizer.report()

    @property
    def conditioning_diagnostics_report(
        self,
    ) -> tuple[dict[str, float | str | None], ...]:
        return self.last_conditioning_diagnostics

    def seed_prepared_assets(self, source_key: str, target_key: str):
        return self.assets.seed_prepared_assets(source_key, target_key)

    def encode_missing_assets(self, **kwargs):
        return self.assets.encode_missing_assets(**kwargs)

    def decode_frames_to_paths(self, frames, output_paths, **kwargs):
        return self.assets.decode_frames_to_paths(frames, output_paths, **kwargs)

    def set_ltm_calibration(self, calibration: LTMCalibration) -> None:
        """Validate and install a persisted FFT LTM calibration."""

        runner = self.runner
        if runner.schedule is None or runner.pipeline is None:
            raise PipelineError("prepared runner lacks CHIMERA model state")
        if calibration.step_count != runner.schedule.num_inference_steps:
            raise PipelineError("LTM calibration schedule length disagrees with the runner")
        if calibration.bands != self.config.ltm_bands:
            raise PipelineError("LTM calibration band count disagrees with ChimeraConfig")
        expected_modules = {
            group.name: group.label
            for group in select_flux_feature_groups(runner.pipeline.transformer)
        }
        if calibration.group_module_map != expected_modules:
            raise PipelineError(
                "LTM calibration feature-group modules disagree with the loaded FLUX model"
            )
        self.ltm_calibration = calibration

    def calibrate_ltm(
        self,
        samples: Sequence[tuple[EncodedSequenceImage, ConditioningPackage]],
    ) -> LTMCalibration:
        """Calibrate FFT LTM from encoded anchor images and their prompts."""

        if not samples:
            raise ValueError("at least one encoded anchor is required for LTM calibration")
        runner = self.runner
        if runner.schedule is None or runner.pipeline is None or runner.image_ids is None:
            raise PipelineError("prepared runner lacks CHIMERA model state")
        runner._set_lora_scale(self.config.lora_scale)
        calibration = calibrate_flux_ltm(
            endpoint_samples=tuple(
                (
                    asset.latent.to(self.device, dtype=torch.float32),
                    conditioning.to(self.device),
                )
                for asset, conditioning in samples
            ),
            schedule=runner.schedule,
            transformer=runner.pipeline.transformer,
            image_ids=runner.image_ids.to(self.device),
            bands=self.config.ltm_bands,
            channel_chunk_size=self.config.ltm_channel_chunk_size,
        )
        self.set_ltm_calibration(calibration)
        return calibration

    def _active_ltm_calibration(self) -> LTMCalibration | None:
        if self.config.ltm_mode == "linear":
            return None
        if self.ltm_calibration is None:
            raise PipelineError(
                "FFT LTM is enabled but no calibration is installed; call "
                "calibrate_ltm() or set_ltm_calibration() first"
            )
        return self.ltm_calibration

    def invert_pair(
        self,
        *,
        pair_key: str,
        source_asset: EncodedSequenceImage,
        target_asset: EncodedSequenceImage,
        source_conditioning: ConditioningPackage,
        target_conditioning: ConditioningPackage,
    ) -> tuple[ChimeraEndpointCache, ChimeraEndpointCache]:
        runner = self.runner
        if runner.schedule is None or runner.pipeline is None or runner.image_ids is None:
            raise PipelineError("prepared runner lacks CHIMERA model state")
        if runner.schedule.num_inference_steps != self.config.inversion_steps:
            raise PipelineError(
                "this memory-bounded port currently requires equal inversion and denoising step counts"
            )
        runner._set_lora_scale(self.config.lora_scale)
        transformer = runner.pipeline.transformer
        image_ids = runner.image_ids.to(self.device)
        controller = FluxFeatureController(
            transformer,
            image_token_count=source_asset.latent.shape[1],
            storage=self.config.cache_storage,
        )
        active_ltm = self._active_ltm_calibration()
        with controller:
            source = invert_endpoint(
                key=f"{pair_key}:A",
                clean_latent=source_asset.latent.to(self.device, dtype=torch.float32),
                schedule=runner.schedule,
                transformer=transformer,
                conditioning=source_conditioning.to(self.device),
                image_ids=image_ids,
                controller=controller,
                ltm_calibration=active_ltm,
                cache_stride=self.config.cache_stride,
            )
            target = invert_endpoint(
                key=f"{pair_key}:B",
                clean_latent=target_asset.latent.to(self.device, dtype=torch.float32),
                schedule=runner.schedule,
                transformer=transformer,
                conditioning=target_conditioning.to(self.device),
                image_ids=image_ids,
                controller=controller,
                ltm_calibration=active_ltm,
                cache_stride=self.config.cache_stride,
            )
        return source, target

    def render_pair(
        self,
        *,
        source_cache: ChimeraEndpointCache,
        target_cache: ChimeraEndpointCache,
        source_conditioning: ConditioningPackage,
        target_conditioning: ConditioningPackage,
        anchor_conditioning: ConditioningPackage,
        alphas: Sequence[float],
    ) -> tuple[RenderedLatentFrame, ...]:
        runner = self.runner
        if not alphas:
            raise ValueError("CHIMERA render_pair requires at least one alpha")
        if (
            runner.schedule is None
            or runner.pipeline is None
            or runner.image_ids is None
            or runner.conditioning_cache is None
        ):
            raise PipelineError("prepared runner lacks CHIMERA rendering state")
        runner._set_lora_scale(self.config.lora_scale)
        transformer = runner.pipeline.transformer
        output: list[RenderedLatentFrame] = []
        conditioning_diagnostics: list[dict[str, float | str | None]] = []
        position = 0
        largest_used = 0
        while position < len(alphas):
            remaining = len(alphas) - position
            active_batch = self.render_batch_sizer.next_batch_size(remaining)
            chunk = tuple(alphas[position : position + active_batch])
            tune_cuda = (
                self.config.auto_render_batch_size
                and torch.cuda.is_available()
                and self.device.type == "cuda"
            )
            baseline_allocated = 0
            free_before = 0
            total_memory = 0
            if tune_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                baseline_allocated = int(torch.cuda.memory_allocated(self.device))
                free_before, total_memory = (
                    int(value) for value in torch.cuda.mem_get_info(self.device)
                )
            try:
                chunk_diagnostics: list[dict[str, float | str | None]] = []
                frames = render_chimera_morph(
                    source_cache,
                    target_cache,
                    schedule=runner.schedule,
                    transformer=transformer,
                    image_ids=runner.image_ids.to(self.device),
                    source_conditioning=source_conditioning,
                    target_conditioning=target_conditioning,
                    anchor_conditioning=anchor_conditioning,
                    unconditional_conditioning=runner.conditioning_cache.unconditional,
                    alphas=chunk,
                    config=self.config,
                    ltm_calibration=self._active_ltm_calibration(),
                    diagnostics=chunk_diagnostics,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                if not (self.config.oom_backoff and is_oom and active_batch > 1):
                    raise
                retry_batch = self.render_batch_sizer.record_oom(active_batch)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    "CHIMERA render OOM; bounded backoff selected "
                    f"batch_size={retry_batch}"
                )
                continue
            conditioning_diagnostics.extend(chunk_diagnostics)
            safe_hint = active_batch
            if tune_cuda:
                safe_hint = estimate_safe_cuda_batch_size(
                    current_batch_size=active_batch,
                    baseline_allocated_bytes=baseline_allocated,
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                    free_before_bytes=free_before,
                    total_bytes=total_memory,
                    maximum_batch_size=self.config.render_batch_max,
                    reserve_fraction=self.config.batch_memory_reserve_fraction,
                    reserve_bytes=int(self.config.batch_memory_reserve_gib * 1024**3),
                    overhead_factor=self.config.batch_estimate_overhead,
                )
            next_batch = self.render_batch_sizer.record_success(
                active_batch,
                safe_ceiling_hint=safe_hint,
            )
            largest_used = max(largest_used, active_batch)
            for frame in frames:
                output.append(
                    RenderedLatentFrame(
                        index=position + frame.index,
                        alpha=frame.alpha,
                        start_state=frame.start_state,
                        final_latent=frame.final_latent,
                        conditioning_mode=frame.conditioning_mode,
                    )
                )
            position += len(chunk)
            if position < len(alphas) and next_batch != active_batch:
                print(
                    "CHIMERA adaptive batching: "
                    f"successful={active_batch}, next={next_batch}, "
                    f"safe_hint={safe_hint}"
                )
        self.last_render_batch_size = largest_used
        self.last_conditioning_diagnostics = tuple(conditioning_diagnostics)
        return tuple(output)


def radial_frequency_descriptor(
    feature: Tensor,
    *,
    bands: int = 16,
    channel_chunk_size: int = 128,
    normalize: bool = True,
) -> Tensor:
    """Return CHIMERA's channel-mean radial FFT-magnitude descriptor.

    FLUX features are token grids.  Square token counts use their natural
    square layout; non-square counts are factored into the closest rectangle.
    Channel chunks bound the transient complex FFT allocation without changing
    the descriptor. LTM-v2 uses normalized band energy so activation scale
    differences between FLUX transformer depths cannot dominate matching;
    ``normalize=False`` remains available for raw-spectrum diagnostics.
    """

    if feature.ndim == 3:
        if feature.shape[0] != 1:
            raise ValueError("frequency descriptor expects one feature sample")
        feature = feature[0]
    if feature.ndim != 2:
        raise ValueError("feature must have shape (tokens, channels) or (1, tokens, channels)")
    if bands < 2:
        raise ValueError("bands must be at least two")
    if channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be positive")
    tokens, channels = feature.shape
    if tokens < 1 or channels < 1:
        raise ValueError("feature token and channel dimensions must be non-empty")
    height = int(math.isqrt(tokens))
    while height > 1 and tokens % height:
        height -= 1
    width = tokens // height
    spatial = feature.float().transpose(0, 1).reshape(channels, height, width)
    magnitude_sum = torch.zeros(
        height,
        width,
        device=feature.device,
        dtype=torch.float32,
    )
    for start in range(0, channels, channel_chunk_size):
        chunk = spatial[start : start + channel_chunk_size]
        magnitude_sum += torch.fft.fftshift(
            torch.fft.fft2(chunk),
            dim=(-2, -1),
        ).abs().sum(dim=0)
    magnitude = magnitude_sum / channels
    yy = torch.arange(height, device=magnitude.device, dtype=torch.float32) - (height - 1) / 2
    xx = torch.arange(width, device=magnitude.device, dtype=torch.float32) - (width - 1) / 2
    radius = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    maximum = torch.clamp(radius.max(), min=torch.finfo(torch.float32).eps)
    indices = torch.clamp((radius / maximum * bands).to(torch.long), max=bands - 1)
    sums = torch.zeros(bands, device=magnitude.device, dtype=torch.float32)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, indices.reshape(-1), magnitude.reshape(-1))
    counts.scatter_add_(0, indices.reshape(-1), torch.ones_like(magnitude).reshape(-1))
    descriptor = sums / torch.clamp(counts, min=1)
    if normalize:
        descriptor = descriptor / torch.clamp(
            descriptor.sum(),
            min=torch.finfo(torch.float32).eps,
        )
    return descriptor


def compute_glcs_from_similarities(
    endpoint_a_similarities: Sequence[float],
    endpoint_b_similarities: Sequence[float],
    *,
    endpoint_similarity_matrix: Sequence[Sequence[float]],
    gamma: float = 2.0,
) -> dict[str, float]:
    """Compute CHIMERA's GCS/LCS/GLCS from any bounded similarity function."""

    a = torch.as_tensor(endpoint_a_similarities, dtype=torch.float64)
    b = torch.as_tensor(endpoint_b_similarities, dtype=torch.float64)
    endpoints = torch.as_tensor(endpoint_similarity_matrix, dtype=torch.float64)
    if a.ndim != 1 or b.shape != a.shape or a.numel() < 1:
        raise ValueError("endpoint similarity sequences must be equal non-empty vectors")
    if endpoints.shape != (2, 2):
        raise ValueError("endpoint_similarity_matrix must have shape (2, 2)")
    if not bool(torch.isfinite(torch.cat((a, b, endpoints.reshape(-1)))).all().item()):
        raise ValueError("similarities must be finite")
    if bool((a.abs() > 1).any().item()) or bool((b.abs() > 1).any().item()) or bool(
        (endpoints.abs() > 1).any().item()
    ):
        raise ValueError("similarities must lie in [-1, 1]")
    if not math.isfinite(gamma) or gamma < 1:
        raise ValueError("gamma must be finite and at least one")

    count = a.numel()
    global_terms = []
    local_terms = []
    for index in range(count):
        alpha = (index + 1) / (count + 1)
        expected_a = (1.0 - alpha) * endpoints[0, 0] + alpha * endpoints[0, 1]
        expected_b = (1.0 - alpha) * endpoints[1, 0] + alpha * endpoints[1, 1]
        global_value = torch.clamp(1 - torch.abs(a[index] - expected_a), 0, 1) * torch.clamp(
            1 - torch.abs(b[index] - expected_b), 0, 1
        )
        global_terms.append(global_value**gamma)

        if count == 1:
            local_a = (endpoints[0, 0] + endpoints[0, 1]) / 2
            local_b = (endpoints[1, 0] + endpoints[1, 1]) / 2
        elif index == 0:
            local_a, local_b = a[1], b[1]
        elif index == count - 1:
            local_a, local_b = a[-2], b[-2]
        else:
            local_a = (a[index - 1] + a[index + 1]) / 2
            local_b = (b[index - 1] + b[index + 1]) / 2
        local_terms.append(
            torch.clamp(1 - torch.abs(a[index] - local_a), 0, 1)
            * torch.clamp(1 - torch.abs(b[index] - local_b), 0, 1)
        )

    gcs = float(torch.stack(global_terms).mean().item())
    lcs = float(torch.stack(local_terms).mean().item())
    return {"gcs": gcs, "lcs": lcs, "glcs": math.sqrt(gcs * lcs)}


__all__ = [
    "CHIMERA_GROUPS",
    "LTM_CALIBRATION_VERSION",
    "LTM_TIMESTEP_SMOOTHING_RADIUS",
    "AdaptiveBatchSizer",
    "ChimeraConfig",
    "ChimeraEndpointCache",
    "ChimeraFlux2Session",
    "FluxFeatureController",
    "FluxFeatureGroup",
    "LTMCalibration",
    "LTMPrototypeAccumulator",
    "StoredFeature",
    "append_anchor_conditioning",
    "allocate_perceptual_subdivisions",
    "calibrate_flux_ltm",
    "center_weighted_alpha_schedule",
    "conditioning_interpolation_report",
    "compute_glcs_from_similarities",
    "estimate_safe_cuda_batch_size",
    "flux_depth_ltm",
    "invert_endpoint",
    "interpolate_chimera_conditioning",
    "ltm_mapping_report",
    "map_denoising_to_inversion_step",
    "match_ltm_prototypes",
    "match_monotonic_ltm_prototypes",
    "nearest_cached_step",
    "prompt_anchor_reliability",
    "radial_frequency_descriptor",
    "render_chimera_morph",
    "resolve_ltm_group",
    "select_flux_feature_groups",
]
