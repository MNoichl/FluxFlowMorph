import torch

from flowmorph_klein.flow_state import FlowMorphEndpoint
from flowmorph_klein.interpolation import (
    interpolate_endpoint,
    interpolate_flowmorph_state,
    slerp,
    slerp_direction_and_magnitude,
)


def test_slerp_identical_vectors_is_stable_and_preserves_dtype() -> None:
    vector = torch.tensor([1.0, 2.0, -3.0], dtype=torch.bfloat16)
    result = slerp(vector, vector, 0.37)
    assert result.dtype is torch.bfloat16
    assert torch.equal(result, vector)


def test_slerp_opposite_vectors_uses_deterministic_great_circle() -> None:
    source = torch.tensor([1.0, 0.0, 0.0])
    target = -source
    midpoint_a = slerp(source, target, 0.5)
    midpoint_b = slerp(source, target, 0.5)

    assert torch.equal(midpoint_a, midpoint_b)
    assert bool(torch.isfinite(midpoint_a).all())
    assert torch.allclose(torch.linalg.vector_norm(midpoint_a), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(torch.dot(midpoint_a, source), torch.tensor(0.0), atol=1e-6)
    assert torch.equal(slerp(source, target, 0.0), source)
    assert torch.equal(slerp(source, target, 1.0), target)


def test_direction_magnitude_slerp_handles_zero_without_nan() -> None:
    source = torch.zeros(3)
    target = torch.tensor([0.0, 4.0, 0.0])
    halfway = slerp_direction_and_magnitude(source, target, 0.5)

    assert bool(torch.isfinite(halfway).all())
    assert torch.equal(halfway, torch.tensor([0.0, 2.0, 0.0]))
    assert torch.equal(slerp_direction_and_magnitude(source, source, 0.5), source)


def test_direction_and_magnitude_are_decoupled() -> None:
    source = torch.tensor([2.0, 0.0])
    target = torch.tensor([0.0, 4.0])
    halfway = slerp_direction_and_magnitude(source, target, 0.5)

    expected_direction = torch.tensor([2**-0.5, 2**-0.5])
    assert torch.allclose(halfway / torch.linalg.vector_norm(halfway), expected_direction, atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(halfway), torch.tensor(3.0), atol=1e-6)


def test_interpolation_has_exact_endpoints_and_state() -> None:
    source = FlowMorphEndpoint(
        z=torch.tensor([1.0, 2.0]),
        delta=torch.tensor([0.1, 0.2]),
        u=torch.tensor([2.0, 0.0]),
        sigma_i=0.8,
        sigma_last=0.0,
        timestep_i=800.0,
    )
    target = FlowMorphEndpoint(
        z=torch.tensor([3.0, 4.0]),
        delta=torch.tensor([-0.1, 0.4]),
        u=torch.tensor([0.0, 4.0]),
        sigma_i=0.8,
        sigma_last=0.0,
        timestep_i=800.0,
    )

    assert torch.equal(interpolate_endpoint(source, target, 0.0).z, source.z)
    assert torch.equal(interpolate_endpoint(source, target, 1.0).u, target.u)
    middle = interpolate_endpoint(source, target, 0.5)
    assert torch.equal(middle.z, torch.tensor([2.0, 3.0]))
    assert torch.equal(interpolate_flowmorph_state(source, target, 0.5), middle.state)
