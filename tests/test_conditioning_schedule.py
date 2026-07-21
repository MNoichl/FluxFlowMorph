from __future__ import annotations

import torch

from flowmorph_klein.conditioning import build_conditioning_cache


def test_conditioning_cache_encodes_and_deduplicates_prompt_schedule() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def encode_prompt(self, *, prompt, num_images_per_prompt, device=None):
            del num_images_per_prompt, device
            self.prompts.append(prompt)
            value = float(len(self.prompts))
            return torch.full((1, 2, 3), value), torch.zeros((1, 2, 4))

    pipeline = FakePipeline()
    cache = build_conditioning_cache(
        pipeline,
        source_prompt="source",
        target_prompt="target",
        bridge_prompt=None,
        bridge_prompts=("source", "middle", "middle", "target"),
        negative_prompt="",
    )

    assert len(cache.prompt_schedule) == 4
    assert cache.prompt_schedule[0] is cache.source
    assert cache.prompt_schedule[1] is cache.prompt_schedule[2]
    assert cache.prompt_schedule[-1] is cache.target
    assert pipeline.prompts == ["source", "target", "", "middle"]
    assert set(cache.prompt_hashes) >= {"source", "target", "schedule_000", "schedule_003"}
