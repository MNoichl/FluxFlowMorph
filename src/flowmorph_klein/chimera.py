"""CHIMERA-style zero-shot morphing for FLUX.2 Klein.

This module ports the architecture-independent parts of CHIMERA to the
repository's native FLUX.2 Euler flow-matching stack:

* reverse Euler inversion of both endpoint latents;
* representative early/middle/late transformer feature caches;
* depth/timestep-aware Adaptive Cache Injection (ACI);
* linear Inversion-Denoising Timestep Mapping (IDM);
* early-step Semantic Anchor Prompting (SAP); and
* the paper's Global-Local Consistency Score (GLCS) aggregation.

The paper's released algorithm is U-Net-centric and its public repository did
not contain implementation code when this port was written.  FLUX has no
down/mid/up blocks, so the three feature groups are mapped to representative
early/middle/late transformer depths.  This follows the FLUX analysis in the
paper's appendix.  The default cache is int8-quantized on CPU and sampled every
second inversion step so a 9B, 1024px Colab run remains practical.  Both
choices are explicit, configurable approximations rather than silent claims
of bit-for-bit parity with the unpublished reference implementation.
"""

from __future__ import annotations

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
from .interpolation import slerp
from .pipeline import FlowMorphRunner, PipelineError
from .renderer import RenderedLatentFrame
from .sequence import EncodedSequenceImage, FlowMorphSequenceSession
from .types import RenderConditioningMode


Tensor = torch.Tensor
CHIMERA_GROUPS = ("early", "middle", "late")
CacheStorage = Literal["int8", "float16", "bfloat16", "float32"]
GroupName = Literal["early", "middle", "late"]


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
    cache_stride: int = 2
    cache_storage: CacheStorage = "int8"
    render_batch_size: int = 2
    decode_batch_size: int = 4
    guidance_scale: float = 7.0
    lora_scale: float = 1.2
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
        if self.cache_stride < 1:
            raise ValueError("cache_stride must be positive")
        if self.cache_storage not in {"int8", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported cache storage {self.cache_storage!r}")
        if self.render_batch_size < 1 or self.decode_batch_size < 1:
            raise ValueError("CHIMERA batch sizes must be positive")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if not math.isfinite(self.lora_scale) or self.lora_scale <= 0:
            raise ValueError("lora_scale must be finite and positive")


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
    """Paper-backed coarse-to-fine LTM prior for a FLUX denoising schedule."""

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
        self._mode: Literal["idle", "capture", "inject"] = "idle"
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


def invert_endpoint(
    *,
    key: str,
    clean_latent: Tensor,
    schedule: FlowSchedule,
    transformer: Any,
    conditioning: ConditioningPackage,
    image_ids: Tensor,
    controller: FluxFeatureController,
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
    with torch.inference_mode():
        for schedule_index in reversed(range(step_count)):
            group = flux_depth_ltm(schedule_index, step_count)
            capture = schedule_index % cache_stride == 0 or schedule_index in {0, step_count - 1}
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
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> tuple[RenderedLatentFrame, ...]:
    """Denoise slerped endpoint latents with IDM, ACI, and early-step SAP."""

    if schedule.num_inference_steps != config.denoising_steps:
        raise ValueError("schedule length disagrees with ChimeraConfig.denoising_steps")
    amounts = tuple(float(alpha) for alpha in alphas)
    if not amounts or any(not 0.0 < alpha < 1.0 for alpha in amounts):
        raise ValueError("CHIMERA render alphas must be non-empty and strictly interior")
    if source.image_token_count != target.image_token_count:
        raise ValueError("endpoint caches use different image token counts")

    initial = torch.cat(
        [slerp(source.inverted_latent, target.inverted_latent, alpha) for alpha in amounts],
        dim=0,
    ).to(device=image_ids.device, dtype=torch.float32)
    base_conditionings = [
        interpolate_conditioning(source_conditioning, target_conditioning, alpha)
        for alpha in amounts
    ]
    base_batch = stack_conditioning_packages(base_conditionings).to(initial.device)
    anchor = anchor_conditioning.to(initial.device)
    unconditional = unconditional_conditioning.to(initial.device)
    alpha_tensor = torch.tensor(amounts, dtype=torch.float64)
    sap_steps = int(math.ceil(config.sap_active_ratio * schedule.num_inference_steps))

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
            group = flux_depth_ltm(denoising_index, schedule.num_inference_steps)
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
        self.last_render_batch_size = 1

    @property
    def device(self) -> torch.device:
        return self.assets.device

    @property
    def last_decode_batch_size(self) -> int:
        return self.assets.last_decode_batch_size

    def seed_prepared_assets(self, source_key: str, target_key: str):
        return self.assets.seed_prepared_assets(source_key, target_key)

    def encode_missing_assets(self, **kwargs):
        return self.assets.encode_missing_assets(**kwargs)

    def decode_frames_to_paths(self, frames, output_paths, **kwargs):
        return self.assets.decode_frames_to_paths(frames, output_paths, **kwargs)

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
        with controller:
            source = invert_endpoint(
                key=f"{pair_key}:A",
                clean_latent=source_asset.latent.to(self.device, dtype=torch.float32),
                schedule=runner.schedule,
                transformer=transformer,
                conditioning=source_conditioning.to(self.device),
                image_ids=image_ids,
                controller=controller,
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
        active_batch = min(self.config.render_batch_size, len(alphas))
        position = 0
        while position < len(alphas):
            chunk = tuple(alphas[position : position + active_batch])
            try:
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
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                if not (self.config.oom_backoff and is_oom and active_batch > 1):
                    raise
                active_batch = max(1, (active_batch + 1) // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"CHIMERA render OOM; retrying with batch_size={active_batch}")
                continue
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
        self.last_render_batch_size = active_batch
        return tuple(output)


def radial_frequency_descriptor(feature: Tensor, *, bands: int = 16) -> Tensor:
    """Return CHIMERA's channel-mean radial FFT-magnitude descriptor.

    FLUX features are token grids.  Square token counts use their natural
    square layout; non-square counts are factored into the closest rectangle.
    """

    if feature.ndim == 3:
        if feature.shape[0] != 1:
            raise ValueError("frequency descriptor expects one feature sample")
        feature = feature[0]
    if feature.ndim != 2:
        raise ValueError("feature must have shape (tokens, channels) or (1, tokens, channels)")
    if bands < 2:
        raise ValueError("bands must be at least two")
    tokens, channels = feature.shape
    height = int(math.isqrt(tokens))
    while height > 1 and tokens % height:
        height -= 1
    width = tokens // height
    spatial = feature.float().transpose(0, 1).reshape(channels, height, width)
    magnitude = torch.fft.fftshift(torch.fft.fft2(spatial), dim=(-2, -1)).abs().mean(dim=0)
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
    return descriptor / torch.clamp(descriptor.sum(), min=torch.finfo(torch.float32).eps)


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
    "ChimeraConfig",
    "ChimeraEndpointCache",
    "ChimeraFlux2Session",
    "FluxFeatureController",
    "FluxFeatureGroup",
    "StoredFeature",
    "append_anchor_conditioning",
    "compute_glcs_from_similarities",
    "flux_depth_ltm",
    "invert_endpoint",
    "map_denoising_to_inversion_step",
    "nearest_cached_step",
    "prompt_anchor_reliability",
    "radial_frequency_descriptor",
    "render_chimera_morph",
    "select_flux_feature_groups",
]
