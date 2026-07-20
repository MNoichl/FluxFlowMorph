# Memory profiles

Profiles never change model architecture, resolution, CFG, scheduler points,
optimization step counts, or the reference twenty-frame contract. `auto`
selects only an explicitly named profile after the backward preflight.

| Profile | Model | Controls | Current support status |
|---|---|---|---|
| `a100_80gb_full` | Base 9B BF16 | checkpointing; VAE/text encoder offload; sequential endpoints/CFG | implementation candidate; exact-model run not executed locally |
| `a100_40gb_checkpointed` | Base 9B BF16 | same plus strict phase cleanup | unsupported until production backward probe and one full endpoint pass |
| `fp8_9b_experimental` | Base 9B FP8 transformer injected into full Base pipeline | checkpointing and explicit dtype/storage audit | unverified: input gradients, LoRA, retained FP8 memory, and checkpointing |
| `unsupported_low_vram` | Base 9B access/preflight only | records failure diagnostics | diagnostic, never a production success |

Default reference controls are BF16 transformer compute, FP32 `pred`/`u`,
gradient checkpointing, sequential CFG, one model instance, text-encoder and
VAE offload, native SDPA, and TF32. Model CPU offload and sequential CPU
offload are intentionally rejected for fitting because inference offload
support does not prove gradients to the input state.

Full-shape `experimental` runs may explicitly disable gradient checkpointing,
select batched CFG, retain the text encoder or VAE on GPU, or choose a
non-reference transformer compute dtype. They retain the exact 9B model,
512×512 shape, 100+100 fitting steps, scheduler/render chain, twenty frames,
FP32 endpoint parameters, and mandatory current-session backward probe. These
switches are experiments, not validated memory profiles or reference results.

On OOM, the runner releases phase tensors and may retry only a profile the user
explicitly selected. It never chooses FLUX.1, 4B, distilled 9B, lower
resolution, fewer fitting steps, fewer frames, altered precision, or CFG scale
1 automatically. Failed profiles remain failures in `memory_report.json`.
