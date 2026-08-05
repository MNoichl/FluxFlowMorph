"""Timing plans for cyclic videos assembled from dense interpolated frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class AnchorHoldPlan:
    """Dense-frame selections for an anchor-hold cyclic video."""

    dense_indices: tuple[int, ...]
    anchor_record_indices: tuple[int, ...]
    segment_frame_counts: tuple[int, ...]
    segment_hold_counts: tuple[int, ...]
    segment_transition_counts: tuple[int, ...]
    segments: tuple[dict[str, Any], ...]
    requested_hold_fraction: float


def _allocate_segment_frame_counts(
    spans: np.ndarray,
    total_frames: int,
) -> np.ndarray:
    """Allocate an exact integer frame budget while retaining every segment."""

    minimum_per_segment = 2  # At least one held frame and one moving frame.
    minimum_total = minimum_per_segment * len(spans)
    if total_frames < minimum_total:
        raise ValueError(
            f"anchor-hold video needs at least {minimum_total} final frames; "
            f"received {total_frames}"
        )
    remaining = total_frames - minimum_total
    ideal_extras = remaining * spans / float(spans.sum())
    extras = np.floor(ideal_extras).astype(np.int64)
    leftover = remaining - int(extras.sum())
    if leftover:
        remainders = ideal_extras - extras
        order = np.argsort(-remainders, kind="stable")
        extras[order[:leftover]] += 1
    return extras + minimum_per_segment


def plan_anchor_hold_timeline(
    records: Sequence[Mapping[str, Any]],
    pair_multipliers: Sequence[int],
    *,
    interpolated_round: int,
    final_frame_count: int,
    hold_fraction: float = 2.0 / 3.0,
    motion_weights: Sequence[float] | None = None,
) -> AnchorHoldPlan:
    """Build a cyclic timeline with long anchor holds and short transitions.

    ``pair_multipliers`` maps each source-frame edge to its number of dense RIFE
    subdivisions. Anchor records are those whose ``round`` differs from the
    final interpolation round. Each anchor-to-anchor beat keeps its original
    share of the total duration, but devotes ``hold_fraction`` of that beat to
    the opening anchor. The remaining frames traverse the complete dense motion
    path without including the next anchor, which begins the following hold.
    """

    record_count = len(records)
    if record_count < 2:
        raise ValueError("anchor-hold timing requires at least two source records")
    if len(pair_multipliers) != record_count:
        raise ValueError("pair multiplier count must equal the cyclic source record count")
    if any(int(value) < 2 for value in pair_multipliers):
        raise ValueError("every pair multiplier must be at least two")
    if not np.isfinite(hold_fraction) or not 0.0 < hold_fraction < 1.0:
        raise ValueError("hold_fraction must lie strictly between zero and one")
    if final_frame_count < 1:
        raise ValueError("final_frame_count must be positive")

    anchor_indices = tuple(
        index
        for index, record in enumerate(records)
        if record.get("round") != interpolated_round
    )
    if len(anchor_indices) < 2:
        raise ValueError("anchor-hold timing could not identify at least two main frames")

    multipliers = np.asarray(pair_multipliers, dtype=np.int64)
    dense_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(multipliers))
    )
    dense_count = int(dense_offsets[-1])
    if motion_weights is None:
        motion = np.ones(dense_count, dtype=np.float64)
    else:
        motion = np.asarray(motion_weights, dtype=np.float64)
        if motion.shape != (dense_count,):
            raise ValueError("motion weight count must equal the dense cyclic frame count")
        if not bool(np.all(np.isfinite(motion))) or bool(np.any(motion <= 0.0)):
            raise ValueError("motion weights must be finite and strictly positive")

    unwrapped_next_anchors: list[int] = []
    record_spans: list[int] = []
    for position, anchor_index in enumerate(anchor_indices):
        next_anchor = anchor_indices[(position + 1) % len(anchor_indices)]
        if next_anchor <= anchor_index:
            next_anchor += record_count
        unwrapped_next_anchors.append(next_anchor)
        record_spans.append(next_anchor - anchor_index)
    segment_counts = _allocate_segment_frame_counts(
        np.asarray(record_spans, dtype=np.float64),
        int(final_frame_count),
    )

    dense_indices: list[int] = []
    hold_counts: list[int] = []
    transition_counts: list[int] = []
    segment_reports: list[dict[str, Any]] = []
    for segment_index, (anchor_index, next_anchor_unwrapped) in enumerate(
        zip(anchor_indices, unwrapped_next_anchors, strict=True)
    ):
        segment_frame_count = int(segment_counts[segment_index])
        hold_count = int(round(segment_frame_count * hold_fraction))
        hold_count = min(max(hold_count, 1), segment_frame_count - 1)
        transition_count = segment_frame_count - hold_count

        next_anchor_index = next_anchor_unwrapped % record_count
        start_dense = int(dense_offsets[anchor_index])
        end_dense = int(dense_offsets[next_anchor_index])
        if next_anchor_unwrapped >= record_count:
            end_dense += dense_count
        dense_span = end_dense - start_dense
        if transition_count > dense_span - 1:
            raise ValueError(
                "anchor-hold transition requests more unique frames than the dense "
                "RIFE segment provides; increase RIFE_MULTIPLIER or reduce final FPS"
            )

        dense_indices.extend([start_dense % dense_count] * hold_count)
        edge_weights = np.asarray(
            [motion[(start_dense + offset) % dense_count] for offset in range(1, dense_span + 1)],
            dtype=np.float64,
        )
        positions = np.concatenate(
            (np.asarray([0.0]), np.cumsum(edge_weights))
        )
        targets = positions[-1] * (
            np.arange(1, transition_count + 1, dtype=np.float64)
            / float(transition_count + 1)
        )
        selected_offsets: list[int] = []
        previous_offset = 0
        for order, target in enumerate(targets):
            minimum = previous_offset + 1
            maximum = dense_span - (transition_count - order)
            insertion = int(np.searchsorted(positions, target, side="left"))
            candidates = {
                min(max(insertion, minimum), maximum),
                min(max(insertion - 1, minimum), maximum),
            }
            chosen = min(candidates, key=lambda index: abs(positions[index] - target))
            selected_offsets.append(chosen)
            previous_offset = chosen
        dense_indices.extend(
            (start_dense + offset) % dense_count for offset in selected_offsets
        )

        hold_counts.append(hold_count)
        transition_counts.append(transition_count)
        segment_reports.append(
            {
                "segment_index": segment_index,
                "anchor_record_index": anchor_index,
                "next_anchor_record_index": next_anchor_index,
                "anchor_uid": records[anchor_index].get("uid", records[anchor_index].get("id")),
                "next_anchor_uid": records[next_anchor_index].get(
                    "uid", records[next_anchor_index].get("id")
                ),
                "source_edge_count": int(record_spans[segment_index]),
                "dense_edge_count": dense_span,
                "final_frame_count": segment_frame_count,
                "hold_frame_count": hold_count,
                "transition_frame_count": transition_count,
                "actual_hold_fraction": hold_count / float(segment_frame_count),
            }
        )

    if len(dense_indices) != final_frame_count:
        raise RuntimeError("anchor-hold timing did not preserve the final frame budget")
    return AnchorHoldPlan(
        dense_indices=tuple(int(index) for index in dense_indices),
        anchor_record_indices=anchor_indices,
        segment_frame_counts=tuple(int(value) for value in segment_counts),
        segment_hold_counts=tuple(hold_counts),
        segment_transition_counts=tuple(transition_counts),
        segments=tuple(segment_reports),
        requested_hold_fraction=float(hold_fraction),
    )


__all__ = ["AnchorHoldPlan", "plan_anchor_hold_timeline"]
