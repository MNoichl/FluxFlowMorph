# License notes

This is a technical summary, not legal advice. Review the current upstream
terms before running or distributing outputs.

The production model is
[`black-forest-labs/FLUX.2-klein-base-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B),
revision `32773329fbe7e81a90ef971740e8ba4b0364ecf3`. Its model card identifies the
FLUX Non-Commercial License v2.1 and requires the account holder to accept the
gate and Acceptable Use Policy before downloading weights.

The license permits the model for defined non-commercial/non-production uses.
It states that BFL claims no ownership in outputs, but restricts using outputs
to train, fine-tune, or distill a model competitive with FLUX. It also requires
either content filtering or review of outputs for unlawful or infringing
content before display/distribution, and AI-generation disclosure where
applicable law requires it. Prohibited-use, privacy, biometric, surveillance,
and other restrictions remain the user's responsibility. Commercial or
production model use requires separate authorization from BFL.

The optional LoRA is user supplied. Its source, revision, filename, SHA-256,
and reported license are recorded at runtime. The user must verify that the
adapter's license permits the intended use and is compatible with the Base 9B
model license. No adapter license can be stated until `LORA_SOURCE` is set.

The `.flowmorph-klein.zip` archive excludes:

- base-model weights and caches;
- text-encoder and VAE weights/caches;
- the downloaded LoRA by default;
- Hugging Face credentials;
- pip and Python caches.

It includes derived frames, endpoint checkpoints, configuration, diagnostics,
metrics, and provenance. An endpoint checkpoint is derived numerical state and
is not intended as a redistribution mechanism for model or adapter weights.

