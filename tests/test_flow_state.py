import torch
import pytest

from flowmorph_klein.flow_state import (
    FlowMorphEndpoint,
    construct_flow_state,
    one_step_reconstruction,
    reconstruction_loss,
    state_from_pred_and_u,
)
from flowmorph_klein.types import LossMode


def test_state_and_reconstruction_use_released_sign_convention() -> None:
    z = torch.tensor([1.0, 2.0])
    delta = torch.tensor([0.25, -0.5])
    u = torch.tensor([2.0, -1.0])
    velocity = torch.tensor([0.5, 3.0])
    sigma_i = 0.75
    sigma_last = 0.0

    state = construct_flow_state(z, delta, u, sigma_i, sigma_last)
    expected_state = (z + delta) - (sigma_last - sigma_i) * u
    assert torch.equal(state, expected_state)
    assert torch.equal(state, state_from_pred_and_u(z + delta, u, sigma_i, sigma_last))

    reconstructed = one_step_reconstruction(state, velocity, sigma_i, sigma_last)
    assert torch.equal(reconstructed, state + (sigma_last - sigma_i) * velocity)


def test_loss_modes_are_unsquared_norm_and_mean_square() -> None:
    target = torch.zeros(4, dtype=torch.bfloat16)
    reconstruction = torch.tensor([1.0, -2.0, 2.0, -1.0], dtype=torch.bfloat16)

    code_loss = reconstruction_loss(reconstruction, target, LossMode.CODE_L2_NORM)
    paper_loss = reconstruction_loss(reconstruction, target, "paper_l2_squared")

    assert code_loss.dtype is torch.float32
    assert torch.allclose(code_loss, torch.sqrt(torch.tensor(10.0)))
    assert torch.equal(paper_loss, torch.tensor(2.5))


def test_endpoint_exposes_explicit_checkpoint_tensors() -> None:
    z = torch.arange(4, dtype=torch.float32)
    endpoint = FlowMorphEndpoint(
        z=z,
        delta=torch.ones_like(z),
        u=torch.full_like(z, 2.0),
        sigma_i=0.5,
        sigma_last=0.0,
        timestep_i=torch.tensor(500.0),
    )

    assert set(endpoint.tensor_dict()) == {"z", "delta", "u"}
    assert torch.equal(endpoint.pred, z + 1.0)
    assert torch.equal(endpoint.state, (z + 1.0) + 1.0)
    detached = endpoint.detached(device="cpu")
    assert not detached.z.requires_grad
    assert detached.z.data_ptr() != endpoint.z.data_ptr()


def test_endpoint_rejects_non_finite_checkpoint_state() -> None:
    finite = torch.zeros(2)
    with pytest.raises(ValueError, match="only finite values"):
        FlowMorphEndpoint(
            z=finite,
            delta=torch.tensor([0.0, float("nan")]),
            u=finite,
            sigma_i=0.5,
            sigma_last=0.0,
        )
