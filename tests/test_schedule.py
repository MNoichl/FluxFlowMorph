import numpy as np
import pytest
import torch

from flowmorph_klein.flow_schedule import (
    build_flowmorph_schedule,
    compute_empirical_mu,
    get_render_chain,
    get_start_state_metadata,
    klein_custom_sigmas,
    validate_sigma_order,
)


class FakeScheduler:
    def __init__(self, *, use_flow_sigmas: bool = False) -> None:
        self.config = {
            "use_flow_sigmas": use_flow_sigmas,
            "num_train_timesteps": 1000,
            "use_dynamic_shifting": True,
        }
        self.call = None

    def set_timesteps(self, *, sigmas=None, num_inference_steps=None, device=None, mu=None) -> None:
        self.call = {
            "sigmas": sigmas,
            "num_inference_steps": num_inference_steps,
            "device": device,
            "mu": mu,
        }
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        sigma_tensor = torch.as_tensor(sigmas, dtype=torch.float32)
        self.timesteps = sigma_tensor * 1000.0
        self.sigmas = torch.cat([sigma_tensor, torch.zeros(1)])


def test_compute_empirical_mu_matches_pinned_formula() -> None:
    image_seq_len = 1024
    steps = 100
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    expected = ((m_200 - m_10) / 190.0) * steps + (m_200 - 200.0 * ((m_200 - m_10) / 190.0))
    assert compute_empirical_mu(image_seq_len, steps) == pytest.approx(expected)
    assert compute_empirical_mu(5000, steps) == pytest.approx(a2 * 5000 + b2)


def test_real_scheduler_path_uses_klein_sigmas_and_actual_token_count() -> None:
    scheduler = FakeScheduler()
    packed = torch.empty(1, 1024, 64)
    schedule = build_flowmorph_schedule(
        scheduler,
        scheduler_points=100,
        packed_latents=packed,
        device="cpu",
    )

    assert scheduler.call["sigmas"] == klein_custom_sigmas(100)
    assert scheduler.call["num_inference_steps"] is None
    assert scheduler.call["mu"] == pytest.approx(compute_empirical_mu(1024, 100))
    assert schedule.image_seq_len == 1024
    assert schedule.num_inference_steps == 100
    assert schedule.used_klein_custom_sigmas

    other = FakeScheduler()
    other_schedule = build_flowmorph_schedule(other, scheduler_points=100, image_seq_len=2048)
    assert other_schedule.mu != schedule.mu


def test_use_flow_sigmas_delegates_schedule_creation() -> None:
    scheduler = FakeScheduler(use_flow_sigmas=True)
    schedule = build_flowmorph_schedule(scheduler, scheduler_points=5, image_seq_len=256)
    assert scheduler.call["sigmas"] is None
    assert scheduler.call["num_inference_steps"] == 5
    assert not schedule.used_klein_custom_sigmas


def test_materialized_cpu_schedule_and_sparse_chain() -> None:
    schedule = build_flowmorph_schedule(
        timesteps=[1000.0, 700.0, 400.0, 100.0],
        sigmas=[1.0, 0.7, 0.4, 0.1, 0.0],
    )
    metadata = get_start_state_metadata(schedule, 0)
    assert metadata.delta_sigma == pytest.approx(-1.0)
    assert metadata.timestep_i == pytest.approx(1000.0)

    chain = get_render_chain(schedule, (0, 2, 3))
    assert len(chain) == 3
    assert chain[0].current_sigma == pytest.approx(1.0)
    assert chain[0].next_sigma == pytest.approx(0.4)
    assert chain[-1].next_index is None
    assert chain[-1].next_sigma == pytest.approx(0.0)


def test_sigma_validation_rejects_wrong_direction() -> None:
    assert validate_sigma_order([1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="decreasing"):
        validate_sigma_order([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="terminal"):
        build_flowmorph_schedule(timesteps=[1.0, 0.5], sigmas=[1.0, 0.0])
