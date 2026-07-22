from types import SimpleNamespace

import pytest
import torch

from flowmorph_klein.conditioning import ConditioningPackage
from flowmorph_klein.flow_state import FlowMorphEndpoint
from flowmorph_klein.pipeline import PipelineError
from flowmorph_klein.renderer import RenderedLatentFrame
from flowmorph_klein.sequence import FlowMorphSequenceSession


def conditioning(prompt: str) -> ConditioningPackage:
    return ConditioningPackage(
        prompt,
        torch.zeros((1, 2, 4)),
        torch.zeros((1, 2, 4), dtype=torch.long),
    )


def endpoint(value: float) -> FlowMorphEndpoint:
    tensor = torch.full((1, 2, 4), value)
    return FlowMorphEndpoint(
        z=tensor,
        delta=torch.zeros_like(tensor),
        u=torch.zeros_like(tensor),
        sigma_i=0.5,
        sigma_last=0.0,
        timestep_i=1.0,
    )


class FakeRunner:
    def __init__(self):
        self._prepared = True
        self.device = torch.device("cpu")
        self.schedule = object()
        self.config = SimpleNamespace(
            lora=SimpleNamespace(render_scale=1.25),
            flowmorph=SimpleNamespace(render_indices=(1, 2)),
        )
        self.scales = []

    def _require_prepared_values(self):
        return (object(), object(), object(), object(), object(), object(), self.schedule)

    def _set_lora_scale(self, value):
        self.scales.append(value)

    def _bound_predictor(self):
        return object()


def test_sequence_session_requires_prepared_runner():
    runner = FakeRunner()
    runner._prepared = False
    with pytest.raises(PipelineError, match="prepared runner"):
        FlowMorphSequenceSession(runner)


def test_render_midpoints_uses_only_requested_interior_alphas(monkeypatch):
    runner = FakeRunner()
    session = FlowMorphSequenceSession(runner)
    observed = {}

    def fake_render_morph(source, target, **kwargs):
        observed.update(kwargs)
        return tuple(
            RenderedLatentFrame(
                index=index,
                alpha=alpha,
                start_state=torch.zeros((1, 2, 4)),
                final_latent=torch.full((1, 2, 4), alpha),
                conditioning_mode="prompt_schedule",
            )
            for index, alpha in enumerate(kwargs["alphas"])
        )

    monkeypatch.setattr("flowmorph_klein.sequence.render_morph", fake_render_morph)
    shared = conditioning("one shared prompt")
    alphas = (1 / 11, 2 / 11, 10 / 11)
    frames = session.render_midpoints(
        source=endpoint(0.0),
        target=endpoint(1.0),
        source_conditioning=conditioning("left"),
        target_conditioning=conditioning("right"),
        midpoint_conditionings=(shared, shared, shared),
        alphas=alphas,
    )

    assert runner.scales == [1.25]
    assert observed["alphas"] == alphas
    assert [item.prompt for item in observed["frame_conditionings"]] == [
        "one shared prompt",
        "one shared prompt",
        "one shared prompt",
    ]
    assert [frame.alpha for frame in frames] == list(alphas)


@pytest.mark.parametrize("alphas", [(), (0.0,), (1.0,), (-0.1,), (1.1,)])
def test_render_midpoints_rejects_non_interior_or_empty_alphas(alphas):
    session = FlowMorphSequenceSession(FakeRunner())
    with pytest.raises(ValueError):
        session.render_midpoints(
            source=endpoint(0.0),
            target=endpoint(1.0),
            source_conditioning=conditioning("left"),
            target_conditioning=conditioning("right"),
            midpoint_conditionings=tuple(conditioning("middle") for _ in alphas),
            alphas=alphas,
        )
