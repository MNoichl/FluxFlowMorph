from __future__ import annotations

import pytest
import torch

from flowmorph_klein.diagnostics import DiagnosticFailure, run_backward_probe


def _probe(predict_velocity):
    return run_backward_probe(
        z=torch.tensor([[[-0.5, 0.25, 1.0]]], dtype=torch.float32),
        sigma_i=0.75,
        sigma_last=0.0,
        timestep=torch.tensor(750.0),
        predict_velocity=predict_velocity,
        frozen_parameters=(),
    )


def test_backward_probe_proves_velocity_input_jacobian() -> None:
    report = _probe(lambda state, _timestep: state.square() + 0.125)

    assert report.passed
    assert report.velocity_input_gradient_norm > 0.0


def test_backward_probe_rejects_detached_velocity_despite_direct_state_gradient() -> None:
    with pytest.raises(DiagnosticFailure, match="detached from autograd"):
        _probe(lambda state, _timestep: state.detach() + 0.125)


def test_backward_probe_rejects_zero_velocity_input_jacobian() -> None:
    with pytest.raises(DiagnosticFailure, match="input gradient is zero"):
        _probe(lambda state, _timestep: state * 0.0 + 0.125)

