from __future__ import annotations

import torch

from flowmorph_klein.conditioning import (
    build_conditioning_cache,
    encode_prompt_conditioning,
)


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


def test_prompt_encoding_retains_klein_chat_attention_mask() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "masked prompt"}]
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is False
            return "templated prompt"

        def __call__(self, text, **kwargs):
            assert text == "templated prompt"
            assert kwargs["max_length"] == 4
            return {"attention_mask": torch.tensor([[1, 1, 1, 0]])}

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def encode_prompt(self, **kwargs):
            assert kwargs["prompt"] == "masked prompt"
            return torch.ones(1, 4, 3), torch.zeros(1, 4, 4)

    package = encode_prompt_conditioning(FakePipeline(), "masked prompt")

    assert package.attention_mask is not None
    assert package.attention_mask.dtype is torch.bool
    assert package.attention_mask.tolist() == [[True, True, True, False]]
