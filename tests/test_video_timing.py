from __future__ import annotations

import pytest

from flowmorph_klein.video_timing import plan_anchor_hold_timeline


def _flat_records() -> list[dict[str, object]]:
    return [
        {"uid": "anchor_0", "round": 0},
        {"uid": "middle_0a", "round": 1},
        {"uid": "middle_0b", "round": 1},
        {"uid": "anchor_1", "round": 0},
        {"uid": "middle_1a", "round": 1},
        {"uid": "middle_1b", "round": 1},
        {"uid": "anchor_2", "round": 0},
        {"uid": "middle_2a", "round": 1},
        {"uid": "middle_2b", "round": 1},
    ]


def test_anchor_hold_plan_uses_two_thirds_holds_and_complete_transitions() -> None:
    records = _flat_records()
    plan = plan_anchor_hold_timeline(
        records,
        [4] * len(records),
        interpolated_round=1,
        final_frame_count=30,
        hold_fraction=2.0 / 3.0,
        motion_weights=[1.0] * 36,
    )

    assert plan.anchor_record_indices == (0, 3, 6)
    assert plan.segment_frame_counts == (10, 10, 10)
    assert plan.segment_hold_counts == (7, 7, 7)
    assert plan.segment_transition_counts == (3, 3, 3)
    assert len(plan.dense_indices) == 30
    assert plan.dense_indices[:7] == (0,) * 7
    assert plan.dense_indices[10:17] == (12,) * 7
    assert plan.dense_indices[20:27] == (24,) * 7
    assert 0 < plan.dense_indices[7] < plan.dense_indices[9] < 12
    assert 24 < plan.dense_indices[27] < plan.dense_indices[29] < 36


def test_anchor_hold_plan_preserves_exact_budget_for_unequal_segments() -> None:
    records = _flat_records()
    records[2]["round"] = 0
    plan = plan_anchor_hold_timeline(
        records,
        [5] * len(records),
        interpolated_round=1,
        final_frame_count=47,
        hold_fraction=0.6,
    )

    assert sum(plan.segment_frame_counts) == 47
    assert len(plan.dense_indices) == 47
    assert all(hold + transition == total for hold, transition, total in zip(
        plan.segment_hold_counts,
        plan.segment_transition_counts,
        plan.segment_frame_counts,
        strict=True,
    ))


@pytest.mark.parametrize("hold_fraction", [0.0, 1.0, -0.1, 1.1])
def test_anchor_hold_plan_rejects_invalid_hold_fraction(hold_fraction: float) -> None:
    records = _flat_records()
    with pytest.raises(ValueError, match="hold_fraction"):
        plan_anchor_hold_timeline(
            records,
            [4] * len(records),
            interpolated_round=1,
            final_frame_count=30,
            hold_fraction=hold_fraction,
        )
