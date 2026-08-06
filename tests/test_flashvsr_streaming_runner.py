from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "flashvsr_v11_streaming_runner.py"
SPEC = importlib.util.spec_from_file_location("flashvsr_v11_streaming_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
build_cyclic_stream_plan = MODULE.build_cyclic_stream_plan
compute_target_dimensions = MODULE.compute_target_dimensions


def test_four_x_dimensions_follow_flashvsr_grid() -> None:
    assert compute_target_dimensions(768, 768, 4.0) == (3072, 3072)
    assert compute_target_dimensions(1408, 768, 4.0) == (5632, 3072)
    with pytest.raises(ValueError):
        compute_target_dimensions(100, 100, 1.0)


def test_cyclic_stream_plan_warms_from_tail_and_preserves_exact_cycle() -> None:
    plan = build_cyclic_stream_plan(3770, warmup_frames=16)
    assert plan.pipeline_frames % 8 == 1
    assert plan.pipeline_output_frames == plan.pipeline_frames - 4
    assert plan.trim_start == 16
    assert plan.trim_count == 3770
    assert plan.source_indices[:16] == tuple(range(3754, 3770))
    assert plan.source_indices[16:20] == (0, 1, 2, 3)
    assert plan.trim_start + plan.trim_count <= plan.pipeline_output_frames


def test_cyclic_stream_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        build_cyclic_stream_plan(0)
    with pytest.raises(ValueError):
        build_cyclic_stream_plan(8, lookahead_frames=3)
