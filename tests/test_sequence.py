from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flowmorph_klein.conditioning import ConditioningPackage
from flowmorph_klein.flow_state import FlowMorphEndpoint
from flowmorph_klein.pipeline import PipelineError
from flowmorph_klein.renderer import RenderedLatentFrame
from flowmorph_klein.sequence import FlowMorphSequenceSession


def conditioning(prompt: str, value: float = 0.0) -> ConditioningPackage:
    return ConditioningPackage(
        prompt,
        torch.full((1, 2, 4), value),
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


def test_render_midpoints_uses_piecewise_endpoint_midpoint_conditioning(monkeypatch):
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
    shared = conditioning("one shared prompt", 2.0)
    alphas = (0.25, 0.5, 0.75)
    frames = session.render_midpoints(
        source=endpoint(0.0),
        target=endpoint(1.0),
        source_conditioning=conditioning("left", 0.0),
        target_conditioning=conditioning("right", 1.0),
        midpoint_conditionings=(shared, shared, shared),
        alphas=alphas,
    )

    assert runner.scales == [1.25]
    assert observed["alphas"] == alphas
    scheduled = observed["frame_conditionings"]
    assert [float(item.prompt_embeds.mean()) for item in scheduled] == [1.0, 2.0, 1.5]
    assert all(isinstance(item.prompt, tuple) and item.prompt[0].startswith("interpolated:") for item in scheduled)
    assert [frame.alpha for frame in frames] == list(alphas)


def test_render_endpoint_reconstructions_batches_each_cached_state_once(monkeypatch):
    runner = FakeRunner()
    session = FlowMorphSequenceSession(runner, render_batch_size=2)
    observed = []

    monkeypatch.setattr(
        "flowmorph_klein.sequence.get_render_chain",
        lambda schedule, indices: ("render-chain", schedule, indices),
    )

    def fake_render_latent_trajectory(
        initial_state,
        *,
        predictor,
        conditioning,
        render_chain,
        frame_index,
    ):
        del predictor
        observed.append({
            "states": initial_state.clone(),
            "prompt_batch": conditioning.prompt_embeds.shape[0],
            "render_chain": render_chain,
            "frame_index": frame_index,
        })
        return initial_state + 10.0

    monkeypatch.setattr(
        "flowmorph_klein.sequence.render_latent_trajectory",
        fake_render_latent_trajectory,
    )
    frames = session.render_endpoint_reconstructions(
        endpoints=(endpoint(0.0), endpoint(1.0), endpoint(2.0)),
        conditionings=(
            conditioning("zero", 0.0),
            conditioning("one", 1.0),
            conditioning("two", 2.0),
        ),
    )

    assert runner.scales == [1.25]
    assert [item["states"].shape[0] for item in observed] == [2, 1]
    assert [item["prompt_batch"] for item in observed] == [2, 1]
    assert [item["frame_index"] for item in observed] == [0, 2]
    assert [float(frame.start_state.mean()) for frame in frames] == [0.0, 1.0, 2.0]
    assert [float(frame.final_latent.mean()) for frame in frames] == [10.0, 11.0, 12.0]
    assert all(frame.alpha == 0.0 for frame in frames)
    assert session.last_render_batch_size == 2


@pytest.mark.parametrize(
    ("endpoints", "conditionings"),
    [
        ((), ()),
        ((endpoint(0.0),), ()),
        ((), (conditioning("zero"),)),
    ],
)
def test_render_endpoint_reconstructions_requires_matching_nonempty_inputs(
    endpoints,
    conditionings,
):
    session = FlowMorphSequenceSession(FakeRunner())
    with pytest.raises(ValueError, match="one conditioning per endpoint"):
        session.render_endpoint_reconstructions(
            endpoints=endpoints,
            conditionings=conditionings,
        )


def test_decode_frames_to_paths_uses_configured_batches(monkeypatch, tmp_path):
    runner = FakeRunner()
    runner.pipeline = SimpleNamespace(
        transformer=torch.nn.Linear(1, 1, bias=False),
        vae=torch.nn.Linear(1, 1, bias=False),
        image_processor=object(),
    )
    runner.image_ids = torch.zeros((1, 2, 4), dtype=torch.long)
    observed_batch_sizes = []

    def fake_decode(tokens, image_ids, vae, **kwargs):
        del vae, kwargs
        assert tokens.shape[0] == image_ids.shape[0]
        observed_batch_sizes.append(tokens.shape[0])
        return [Image.new("RGB", (4, 4), (index, 0, 0)) for index in range(tokens.shape[0])]

    monkeypatch.setattr("flowmorph_klein.sequence.decode_packed_latent", fake_decode)
    monkeypatch.setattr("flowmorph_klein.sequence._move_module", lambda *args: None)
    monkeypatch.setattr("flowmorph_klein.sequence.release_cuda_memory", lambda: None)
    session = FlowMorphSequenceSession(runner, decode_batch_size=2)
    frames = tuple(
        RenderedLatentFrame(
            index=index,
            alpha=index / 5,
            start_state=torch.zeros((1, 2, 4)),
            final_latent=torch.zeros((1, 2, 4)),
            conditioning_mode="prompt_schedule",
        )
        for index in range(5)
    )
    destinations = [tmp_path / f"{index}.png" for index in range(5)]

    result = session.decode_frames_to_paths(frames, destinations)

    assert observed_batch_sizes == [2, 2, 1]
    assert session.last_decode_batch_size == 2
    assert result == tuple(destinations)
    assert all(path.is_file() for path in destinations)


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
