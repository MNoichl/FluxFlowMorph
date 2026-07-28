"""Experimental periodic B-spline trajectories through fitted FlowMorph states.

This module is intentionally separate from the pairwise interpolation and
sequence implementations.  It reuses their fitted endpoints, conditioning
packages, sparse render chain, and decoder without changing pairwise behavior.

The curve is an interpolating periodic cubic B-spline: every fitted endpoint is
an exact knot, while position, first derivative, and second derivative agree at
the circular seam.  ``z`` and ``delta`` use ordinary tensor-valued splines.
``u`` follows FlowMorph's direction/magnitude idea by splining unit directions
and log magnitudes separately before recombination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .conditioning import ConditioningPackage
from .diagnostics import release_cuda_memory
from .flow_schedule import get_render_chain
from .flow_state import FlowMorphEndpoint, sigma_delta
from .pipeline import PipelineError
from .renderer import RenderedLatentFrame, render_latent_trajectory
from .sequence import FlowMorphSequenceSession
from .types import RenderConditioningMode


@dataclass(frozen=True, slots=True)
class PeriodicTimingPlan:
    """Regularized segment durations and their cumulative circular knots."""

    raw_distances: tuple[float, ...]
    segment_durations: tuple[float, ...]
    knot_times: tuple[float, ...]
    distance_strength: float
    distance_exponent: float
    maximum_segment_ratio: float

    def __post_init__(self) -> None:
        count = len(self.segment_durations)
        if count < 4 or len(self.raw_distances) != count:
            raise ValueError("periodic cubic timing requires at least four segments")
        if len(self.knot_times) != count + 1:
            raise ValueError("knot_times must include the terminal circular knot")
        if not math.isclose(self.knot_times[0], 0.0, abs_tol=1e-12):
            raise ValueError("periodic knot_times must begin at zero")
        if not math.isclose(self.knot_times[-1], 1.0, abs_tol=1e-12):
            raise ValueError("periodic knot_times must end at one")


@dataclass(frozen=True, slots=True)
class PeriodicSplineSample:
    """One unique frame time on the circular spline."""

    frame_index: int
    time: float
    segment_index: int
    segment_fraction: float
    anchor_index: int | None


def periodic_thumbnail_distances(
    image_paths: Sequence[str | Path],
    *,
    analysis_size: int = 128,
    color_weight: float = 0.75,
) -> tuple[float, ...]:
    """Measure robust, inexpensive visual chord lengths around an image loop.

    The proxy combines RGB RMS difference with first-derivative RMS difference.
    It is used only for timing allocation, not as an optimization objective.
    """

    if len(image_paths) < 4:
        raise ValueError("periodic image distance needs at least four images")
    if analysis_size < 32:
        raise ValueError("analysis_size must be at least 32")
    if not 0.0 <= color_weight <= 1.0:
        raise ValueError("color_weight must lie in [0, 1]")

    arrays: list[np.ndarray] = []
    for path in image_paths:
        with Image.open(path) as opened:
            resized = opened.convert("RGB").resize(
                (analysis_size, analysis_size),
                Image.Resampling.LANCZOS,
            )
            arrays.append(np.asarray(resized, dtype=np.float32) / 255.0)

    distances: list[float] = []
    for index, left in enumerate(arrays):
        right = arrays[(index + 1) % len(arrays)]
        color = float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))
        left_dx = np.diff(left, axis=1)
        right_dx = np.diff(right, axis=1)
        left_dy = np.diff(left, axis=0)
        right_dy = np.diff(right, axis=0)
        gradient = 0.5 * (
            float(np.sqrt(np.mean(np.square(left_dx - right_dx), dtype=np.float64)))
            + float(np.sqrt(np.mean(np.square(left_dy - right_dy), dtype=np.float64)))
        )
        distances.append(max(1e-8, color_weight * color + (1.0 - color_weight) * gradient))
    return tuple(distances)


def regularized_periodic_timing(
    distances: Sequence[float],
    *,
    distance_strength: float = 0.45,
    distance_exponent: float = 0.5,
    maximum_segment_ratio: float = 1.75,
) -> PeriodicTimingPlan:
    """Turn circular visual distances into restrained nonuniform knot spacing.

    ``distance_strength=0`` is uniform timing.  Square-root tempering and a
    symmetric log-ratio cap prevent one visually unusual gap from consuming the
    whole film.
    """

    values = np.asarray(tuple(distances), dtype=np.float64)
    if values.ndim != 1 or len(values) < 4:
        raise ValueError("periodic cubic timing requires at least four distances")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("distances must be finite and nonnegative")
    if not 0.0 <= distance_strength <= 1.0:
        raise ValueError("distance_strength must lie in [0, 1]")
    if not 0.0 < distance_exponent <= 1.0:
        raise ValueError("distance_exponent must lie in (0, 1]")
    if maximum_segment_ratio < 1.0:
        raise ValueError("maximum_segment_ratio must be at least one")

    positive = values[values > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    floor = max(np.finfo(np.float64).eps, scale * 1e-3)
    relative = np.maximum(values, floor) / max(scale, floor)
    tempered_log = distance_exponent * np.log(relative)
    tempered_log -= float(tempered_log.mean())
    half_limit = 0.5 * math.log(maximum_segment_ratio)
    tempered_log = np.clip(tempered_log, -half_limit, half_limit)
    distance_durations = np.exp(tempered_log)
    mixed = (1.0 - distance_strength) + distance_strength * distance_durations
    durations = mixed / mixed.sum()
    knots = np.concatenate(([0.0], np.cumsum(durations)))
    knots[-1] = 1.0
    return PeriodicTimingPlan(
        raw_distances=tuple(float(value) for value in values),
        segment_durations=tuple(float(value) for value in durations),
        knot_times=tuple(float(value) for value in knots),
        distance_strength=float(distance_strength),
        distance_exponent=float(distance_exponent),
        maximum_segment_ratio=float(maximum_segment_ratio),
    )


def allocate_periodic_segment_frames(
    segment_durations: Sequence[float],
    *,
    total_frames: int,
    minimum_frames_per_segment: int = 3,
) -> tuple[int, ...]:
    """Allocate an exact frame budget while retaining every anchor frame."""

    durations = np.asarray(tuple(segment_durations), dtype=np.float64)
    if durations.ndim != 1 or len(durations) < 4:
        raise ValueError("at least four segment durations are required")
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("segment durations must be finite and positive")
    if minimum_frames_per_segment < 2:
        raise ValueError("minimum_frames_per_segment must be at least two")
    minimum_total = len(durations) * minimum_frames_per_segment
    if total_frames < minimum_total:
        raise ValueError(
            f"total_frames must be at least {minimum_total} for this timing plan"
        )

    normalized = durations / durations.sum()
    remaining = total_frames - minimum_total
    ideal_extra = normalized * remaining
    extra = np.floor(ideal_extra).astype(np.int64)
    missing = remaining - int(extra.sum())
    if missing:
        order = np.argsort(-(ideal_extra - extra), kind="stable")
        extra[order[:missing]] += 1
    counts = extra + minimum_frames_per_segment
    if int(counts.sum()) != total_frames:
        raise AssertionError("periodic frame allocation lost the requested budget")
    return tuple(int(value) for value in counts)


def sample_periodic_timeline(
    timing: PeriodicTimingPlan,
    segment_frame_counts: Sequence[int],
) -> tuple[PeriodicSplineSample, ...]:
    """Sample each segment including its left anchor and excluding its right."""

    counts = tuple(int(value) for value in segment_frame_counts)
    if len(counts) != len(timing.segment_durations):
        raise ValueError("one frame count is required per periodic segment")
    if any(value < 2 for value in counts):
        raise ValueError("every segment requires an anchor plus an interior frame")

    samples: list[PeriodicSplineSample] = []
    for segment_index, count in enumerate(counts):
        start = timing.knot_times[segment_index]
        duration = timing.segment_durations[segment_index]
        for local_index in range(count):
            fraction = local_index / count
            samples.append(
                PeriodicSplineSample(
                    frame_index=len(samples),
                    time=float((start + fraction * duration) % 1.0),
                    segment_index=segment_index,
                    segment_fraction=float(fraction),
                    anchor_index=segment_index if local_index == 0 else None,
                )
            )
    return tuple(samples)


class PeriodicCubicBSplineBasis:
    """Small periodic interpolating basis evaluated independently of tensor size."""

    def __init__(self, knot_times: Sequence[float]) -> None:
        knots = np.asarray(tuple(knot_times), dtype=np.float64)
        if knots.ndim != 1 or len(knots) < 5:
            raise ValueError("periodic cubic B-spline requires at least four anchors")
        if not np.isfinite(knots).all() or not np.all(np.diff(knots) > 0):
            raise ValueError("knot_times must be finite and strictly increasing")
        if not math.isclose(float(knots[0]), 0.0, abs_tol=1e-12):
            raise ValueError("knot_times must begin at zero")
        if not math.isclose(float(knots[-1]), 1.0, abs_tol=1e-12):
            raise ValueError("knot_times must end at one")
        try:
            from scipy.interpolate import make_interp_spline
        except ImportError as error:  # pragma: no cover - Colab requirements pin SciPy.
            raise RuntimeError("periodic B-spline trajectories require scipy") from error

        anchor_count = len(knots) - 1
        identity = np.eye(anchor_count, dtype=np.float64)
        periodic_values = np.concatenate((identity, identity[:1]), axis=0)
        self._knots = knots
        self._spline = make_interp_spline(
            knots,
            periodic_values,
            k=3,
            bc_type="periodic",
            axis=0,
        )

    @property
    def anchor_count(self) -> int:
        return len(self._knots) - 1

    @property
    def knot_times(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._knots)

    def weights(
        self,
        times: Sequence[float] | np.ndarray,
        *,
        derivative: int = 0,
    ) -> np.ndarray:
        if derivative not in {0, 1, 2}:
            raise ValueError("only position, velocity, and acceleration are supported")
        values = np.asarray(times, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("spline times must be a finite one-dimensional sequence")
        wrapped = np.mod(values, 1.0)
        evaluator = self._spline if derivative == 0 else self._spline.derivative(derivative)
        result = np.asarray(evaluator(wrapped), dtype=np.float64)
        expected = (len(values), self.anchor_count)
        if result.shape != expected:
            raise AssertionError(f"unexpected periodic basis shape {result.shape}; expected {expected}")
        return result


def _same_scalar(left: float | torch.Tensor, right: float | torch.Tensor) -> bool:
    left_value = torch.as_tensor(left, dtype=torch.float64)
    right_value = torch.as_tensor(right, dtype=torch.float64)
    return (
        left_value.numel() == right_value.numel() == 1
        and bool(torch.equal(left_value, right_value))
    )


class PeriodicFlowMorphSpline:
    """Evaluate smooth circular states through cached fitted endpoints."""

    def __init__(
        self,
        endpoints: Sequence[FlowMorphEndpoint],
        basis: PeriodicCubicBSplineBasis,
        *,
        direction_epsilon: float = 1e-8,
    ) -> None:
        self.endpoints = tuple(endpoints)
        self.basis = basis
        self.direction_epsilon = float(direction_epsilon)
        if len(self.endpoints) != basis.anchor_count:
            raise ValueError("endpoint count must match periodic spline anchors")
        if direction_epsilon <= 0:
            raise ValueError("direction_epsilon must be positive")
        first = self.endpoints[0]
        for endpoint in self.endpoints[1:]:
            if endpoint.z.shape != first.z.shape:
                raise ValueError("all spline endpoints must have the same tensor shape")
            if not _same_scalar(endpoint.sigma_i, first.sigma_i):
                raise ValueError("all spline endpoints must share the start sigma")
            if not _same_scalar(endpoint.sigma_last, first.sigma_last):
                raise ValueError("all spline endpoints must share the terminal sigma")
        self.sigma_i = first.sigma_i
        self.sigma_last = first.sigma_last

    def _weights_tensor(
        self,
        times: Sequence[float],
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        weights = self.basis.weights(times)
        return torch.as_tensor(weights, device=device, dtype=torch.float32)

    def state_batch(
        self,
        times: Sequence[float],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Evaluate a batch of short-trajectory FlowMorph starting states."""

        if not times:
            raise ValueError("at least one spline time is required")
        weights = self._weights_tensor(times, device=device)
        z_stack = torch.cat(
            [endpoint.z.detach().to(device=device, dtype=torch.float32) for endpoint in self.endpoints],
            dim=0,
        )
        delta_stack = torch.cat(
            [
                endpoint.delta.detach().to(device=device, dtype=torch.float32)
                for endpoint in self.endpoints
            ],
            dim=0,
        )
        u_stack = torch.cat(
            [endpoint.u.detach().to(device=device, dtype=torch.float32) for endpoint in self.endpoints],
            dim=0,
        )
        z = torch.einsum("fa,a...->f...", weights, z_stack)
        delta = torch.einsum("fa,a...->f...", weights, delta_stack)

        flat_u = u_stack.reshape(len(self.endpoints), -1)
        magnitudes = torch.linalg.vector_norm(flat_u, dim=1)
        if bool((magnitudes > self.direction_epsilon).all().item()):
            directions = u_stack / magnitudes.reshape((-1,) + (1,) * (u_stack.ndim - 1))
            direction = torch.einsum("fa,a...->f...", weights, directions)
            direction_flat = direction.reshape(len(times), -1)
            direction_norm = torch.linalg.vector_norm(direction_flat, dim=1)
            if bool((direction_norm <= self.direction_epsilon).any().item()):
                raise FloatingPointError(
                    "periodic u-direction spline collapsed; reduce timing irregularity "
                    "or add a neighboring anchor"
                )
            log_magnitude = torch.log(magnitudes)
            magnitude = torch.exp(weights @ log_magnitude)
            u = direction / direction_norm.reshape(
                (-1,) + (1,) * (direction.ndim - 1)
            )
            u = u * magnitude.reshape((-1,) + (1,) * (u.ndim - 1))
        else:
            # Synthetic tests and degenerate endpoints can legitimately contain
            # zero u. Raw interpolation remains exact and finite in that case.
            u = torch.einsum("fa,a...->f...", weights, u_stack)

        state = z + delta - sigma_delta(
            self.sigma_i,
            self.sigma_last,
            like=z,
        ) * u
        if not bool(torch.isfinite(state).all().item()):
            raise FloatingPointError("periodic FlowMorph spline produced a non-finite state")
        return state.to(dtype=dtype)


class PeriodicConditioningSpline:
    """Spline compatible FLUX prompt embeddings over the same circular knots."""

    def __init__(
        self,
        conditionings: Sequence[ConditioningPackage],
        basis: PeriodicCubicBSplineBasis,
    ) -> None:
        self.conditionings = tuple(conditionings)
        self.basis = basis
        if len(self.conditionings) != basis.anchor_count:
            raise ValueError("conditioning count must match periodic spline anchors")
        first = self.conditionings[0]
        for package in self.conditionings[1:]:
            if package.prompt_embeds.shape != first.prompt_embeds.shape:
                raise ValueError("all spline prompt embedding shapes must match")
            if package.text_ids.shape != first.text_ids.shape or not torch.equal(
                package.text_ids.cpu(),
                first.text_ids.cpu(),
            ):
                raise ValueError(
                    "periodic prompt splining requires identical FLUX text position IDs"
                )

    def evaluate_batch(
        self,
        times: Sequence[float],
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> ConditioningPackage:
        if not times:
            raise ValueError("at least one conditioning time is required")
        weights = torch.as_tensor(
            self.basis.weights(times),
            device=device,
            dtype=torch.float32,
        )
        embeds = torch.cat(
            [
                package.prompt_embeds.detach().to(device=device, dtype=torch.float32)
                for package in self.conditionings
            ],
            dim=0,
        )
        interpolated = torch.einsum("fa,asw->fsw", weights, embeds)
        if dtype is not None:
            interpolated = interpolated.to(dtype=dtype)
        text_ids = self.conditionings[0].text_ids.to(device=device).expand(
            len(times),
            -1,
            -1,
        )
        return ConditioningPackage(
            prompt=tuple(f"periodic_bspline:{float(time) % 1.0:.10f}" for time in times),
            prompt_embeds=interpolated.detach(),
            text_ids=text_ids.detach(),
        )


class PeriodicSplineFlowMorphRenderer:
    """Alternative renderer for spline states using an existing sequence session."""

    def __init__(
        self,
        session: FlowMorphSequenceSession,
        trajectory: PeriodicFlowMorphSpline,
        conditioning: PeriodicConditioningSpline,
    ) -> None:
        if trajectory.basis.knot_times != conditioning.basis.knot_times:
            raise ValueError("state and conditioning splines must share knot times")
        self.session = session
        self.trajectory = trajectory
        self.conditioning = conditioning

    def render(self, times: Sequence[float]) -> tuple[RenderedLatentFrame, ...]:
        if not times:
            raise ValueError("at least one spline render time is required")
        runner = self.session.runner
        if runner.schedule is None:
            raise PipelineError("prepared runner lacks schedule")
        runner._set_lora_scale(runner.config.lora.render_scale)
        predictor = runner._bound_predictor()
        if self.session.cfg_execution is not None:
            predictor.cfg_execution = self.session.cfg_execution
        render_chain = get_render_chain(
            runner.schedule,
            runner.config.flowmorph.render_indices,
        )

        frames: list[RenderedLatentFrame] = []
        active_batch_size = min(self.session.render_batch_size, len(times))
        position = 0
        while position < len(times):
            chunk_times = tuple(
                float(value) % 1.0
                for value in times[position : position + active_batch_size]
            )
            try:
                states = self.trajectory.state_batch(
                    chunk_times,
                    device=self.session.device,
                    dtype=torch.float32,
                )
                conditionings = self.conditioning.evaluate_batch(
                    chunk_times,
                    device=self.session.device,
                )
                final_latents = render_latent_trajectory(
                    states,
                    predictor=predictor,
                    conditioning=conditionings,
                    render_chain=render_chain,
                    frame_index=position,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    "out of memory" in str(error).lower()
                )
                if self.session.oom_backoff and is_oom and active_batch_size > 1:
                    active_batch_size = max(1, (active_batch_size + 1) // 2)
                    release_cuda_memory()
                    print(
                        "Periodic spline render OOM; retrying with "
                        f"batch_size={active_batch_size}"
                    )
                    continue
                if (
                    self.session.oom_backoff
                    and is_oom
                    and getattr(predictor, "cfg_execution", None) == "batched"
                ):
                    predictor.cfg_execution = "sequential"
                    self.session.cfg_execution = "sequential"
                    release_cuda_memory()
                    print("Periodic spline render OOM; retrying sequential CFG")
                    continue
                raise

            for offset, time in enumerate(chunk_times):
                frame_slice = slice(offset, offset + 1)
                frames.append(
                    RenderedLatentFrame(
                        index=position + offset,
                        alpha=time,
                        start_state=states[frame_slice].detach().cpu(),
                        final_latent=final_latents[frame_slice].detach().cpu(),
                        conditioning_mode=RenderConditioningMode.PROMPT_SCHEDULE,
                    )
                )
            position += len(chunk_times)
        self.session.last_render_batch_size = active_batch_size
        del predictor
        return tuple(frames)


__all__ = [
    "PeriodicConditioningSpline",
    "PeriodicCubicBSplineBasis",
    "PeriodicFlowMorphSpline",
    "PeriodicSplineFlowMorphRenderer",
    "PeriodicSplineSample",
    "PeriodicTimingPlan",
    "allocate_periodic_segment_frames",
    "periodic_thumbnail_distances",
    "regularized_periodic_timing",
    "sample_periodic_timeline",
]
