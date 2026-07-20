# Architecture

The package separates pure numerical behavior from the gated FLUX.2 backend.
Most tests therefore run without downloading nine billion parameters.

```text
configuration, contracts, typed errors
                  |
     pure numerical and I/O core
  state / schedule / SLERP / images / checkpoints
                  |
           FLUX.2 adapters
 latent codec / conditioning / velocity / LoRA
                  |
       sequential run orchestrator
                  |
   metrics / previews / atomic archive
```

The public runner advances through explicit phases:

```text
inputs_validated
  -> model_ready
  -> adapter_verified (or explicitly absent)
  -> backward_preflight_passed
  -> source_checkpointed
  -> target_checkpointed
  -> frames_rendered
  -> metrics_complete
  -> archive_validated
```

Every transition is written to `run_manifest.json`. A failure retains its
actual phase and diagnostic status; it cannot be promoted to a full
reproduction.

Core contracts are small dataclasses/protocol-shaped callables:

- an endpoint state contains explicit `z`, `delta`, `u`, and compatibility
  metadata;
- a conditioning package contains prompt embeddings, text IDs, and a prompt
  hash;
- a schedule contains all timesteps/sigmas and sparse render transitions;
- a velocity predictor maps a packed state, timestep, IDs, and conditioning to
  a packed flow tensor;
- a latent codec maps preprocessed RGB endpoints to/from packed FLUX.2 state.

The package never invokes `Flux2KleinPipeline.__call__` inside fitting. It
calls `pipe.transformer(...)` so input gradients and PEFT hooks remain active.
All model/adapter parameters are re-frozen after adapter loading. `pred` and
`u` are FP32 leaves, while constructed state is cast to the transformer dtype
without a detach.

Source and target are fitted sequentially through one frozen transformer. The
source checkpoint is persisted before the target optimizer is constructed.
Prompt packages and endpoint state move to CPU when inactive. Rendering uses
inference mode only after interpolation has produced the start state.

Archive creation uses an allowlist, secret scanning, ZIP64, a temporary file,
CRC validation, SHA-256, and atomic rename. This prevents model caches, LoRA
weights, or an archive-under-construction from recursively entering the final
artifact.

