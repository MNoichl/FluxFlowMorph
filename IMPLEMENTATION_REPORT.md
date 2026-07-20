# Implementation report

Status date: 2026-07-20.

This report uses three validation labels:

- **Offline verified**: exercised locally without gated model weights, a CUDA
  device, or a user LoRA.
- **Implemented, integration pending**: code and an opt-in test gate exist, but
  the real FLUX.2 Klein Base 9B path was not executed.
- **Not run**: no result artifact or performance claim exists.

The final full-suite command `pytest -q` completed with **188 passed and 7
skipped in 1.54 seconds**. The seven skipped tests are the explicitly opt-in
CUDA/gated-model integrations; they did not execute. Python byte-compilation
of `src/` and `scripts/` passed, and the notebook JSON plus all 24 code cells
parsed successfully. The machine used for this report had Python 3.12.11,
PyTorch 2.9.1 with `torch.cuda.is_available() == False`, no CUDA runtime, no
gated-model credentials, and no user-supplied LoRA. Consequently, this is an
implementation candidate with a verified offline core; it does **not** yet
satisfy the specification's production acceptance criteria.

## 1. Summary of the implementation

The project supplies a strict configuration model, Colab notebook facade,
exact-model and exact-Diffusers guards, deterministic image preprocessing, FLUX.2 latent
transforms, Qwen conditioning caches, low-level differentiable transformer
calls, external classifier-free guidance, native unfused LoRA loading and
validation, the FlowMorph endpoint equations, FP32 endpoint optimization,
checkpoint/resume support, decoupled endpoint interpolation, sparse Euler
rendering, metrics, visualizations, video writers, manifests, acceptance
auditing, and atomic allowlisted archives. Mandatory output and provenance
switches fail closed at configuration validation. The backward preflight may
retry one CUDA OOM once only under the already selected profile, without a
model, profile, precision, resolution, step, frame, schedule, or CFG change.
Before accepting ordinary endpoint gradients, that preflight separately
backpropagates a deterministic velocity-only projection to the transformer
input, records `velocity_input_gradient_norm`, and rejects detached, missing,
non-finite, or zero input gradients.
Specifically, raw frames, display frames, endpoint states, loss history, ZIP
creation, environment recording, and checksum recording cannot be disabled;
the archive suffix is fixed to `.flowmorph-klein.zip`.

The reference contract is exactly
`black-forest-labs/FLUX.2-klein-base-9B` at 512x512, 100 scheduler points,
start index 35, 100 source steps followed by 100 target steps, render indices
35/55/75/95, external CFG scale 4.0, and 20 frames. Configuration rejects
silent FLUX.1, 4B, distilled-9B, lower-resolution, lower-step, lower-frame, or
CFG-disabled fallback. The implementation never trains model or LoRA weights;
only endpoint `pred` and `u` are trainable.

Validation status is deliberately narrower than implementation scope:

| Area | Status |
|---|---|
| Pure equations, schedules, interpolation, configuration, I/O, checkpoints, archive policy, CLI/notebook structure | **Offline verified; 184 tests passed** |
| Gated Base-9B load, real VAE parity, real velocity parity, input backward, real LoRA activation | **Implemented, integration pending** |
| Complete 100+100 fit, 20 decoded frames, production metrics, runtime, VRAM, final run archive | **Not run** |

## 2. Exact FlowMorph sources inspected

The final WACV 2026 FlowMorph paper and the pinned VITA repository were
inspected. Repository files inspected were:

- `README.md`
- `docs/METHOD.md`
- `configs/default.yaml`
- `flowmorph/flow_interpolation.py`
- `flowmorph/flux_optim.py`
- `flowmorph/utils.py`
- `flowmorph/prompt_interpolator.py`
- `scripts/run_flow_interpolation.sh`
- `scripts/ablations/run_prompt_separation.sh`

For the adaptation, the BFL model card/repository and the Diffusers FLUX.2
implementation were also inspected, including
`pipeline_flux2_klein.py`, `image_processor.py`, `transformer_flux2.py`,
`attention_processor.py`, `autoencoder_kl_flux2.py`,
`scheduling_flow_match_euler_discrete.py`, `lora_pipeline.py`, LoRA conversion
utilities, and the official Klein DreamBooth LoRA example. The relevant APIs
include `Flux2KleinPipeline`, `Flux2Transformer2DModel`,
`AutoencoderKLFlux2`, `FlowMatchEulerDiscreteScheduler`,
`Flux2LoraLoaderMixin`, `retrieve_latents`, `compute_empirical_mu`,
`_patchify_latents`, `_pack_latents`, `_unpack_latents_with_ids`,
`encode_prompt`, `load_lora_weights`, and `set_adapters`.

## 3. Exact repository commits

| Repository/source | Commit | Status |
|---|---|---|
| `VITA-Group/FlowMorph` | `0db52344ad0ec6963f74a508db831e506058b2f7` | inspected |
| `huggingface/diffusers` v0.39.0 source | `a3608b512ed7248499a44c61d954965ed9bdae4d` | inspected and pinned; real 9B validation pending |
| `black-forest-labs/flux2` | `50fe5162777813d869182b139e83b10743caef15` | inspected |
| Diffusers main compared during research | `86e6dac5360703ddf09fe250db50be667eb93662` | compared only; not selected |

The full Diffusers SHA is pinned in `requirements-colab.txt`. Before accepting
a loaded model, the runner fail-closes unless distribution metadata reports
version exactly `0.39.0` and PEP 610 VCS provenance reports that exact commit.
It then requires the exact pipeline, transformer, VAE, scheduler, and Qwen text
encoder classes; explicit undistilled status; the expected 9B transformer
configuration; non-empty parameters; and the exact Hub revision. These guards
are implemented and offline-inspected, but the SHA must not be described as a
*tested 9B commit* until the gated load, VAE, scheduler/velocity, LoRA, and
backward gates all pass. An immutable pin and strict runtime checks do not turn
source inspection into CUDA validation.

## 4. Exact model revision

The reference model is
`black-forest-labs/FLUX.2-klein-base-9B` at revision
`32773329fbe7e81a90ef971740e8ba4b0364ecf3`.

The only separately configured experimental artifact is
`black-forest-labs/FLUX.2-klein-base-9b-fp8` at revision
`9ecf2143d71542449960c5584340269c6d401449`. Its explicit implementation
downloads the BFL single-file transformer, constructs
`Flux2Transformer2DModel` from the pinned Base-9B transformer configuration,
enables float8-e4m3fn layerwise storage with configured compute dtype, and
injects that transformer into a complete pipeline loaded from the exact BF16
Base-9B revision. It verifies access to both repositories and refuses an FP8
load with no float8 parameters. This path was not loaded or executed, is not a
reference fallback, and has no support claim. The code rejects 4B and
distilled production substitutions.

## 5. Exact LoRA source, revision, and fingerprint

No LoRA was supplied or resolved for this implementation session:

| Field | Value |
|---|---|
| Source | not configured |
| Requested revision | not applicable |
| Resolved revision | not applicable |
| Weight filename | not applicable |
| SHA-256 fingerprint | not available |
| Adapter name | configured default `flowmorph_adapter` |
| Fit/render scale | configured default 1.0 / 1.0 |

Therefore no real-adapter compatibility or activation claim is made. On a real
run, the resolver records repository/file provenance, resolved revision,
filename, safetensors metadata/key family, mapped tensor count, local SHA-256,
active-adapter API results, and the disabled-versus-enabled velocity delta.

## 6. Exact dependency versions

The production/Colab requirements declare the following exact direct pins:

| Dependency | Declared version/revision |
|---|---|
| Diffusers | Git SHA `a3608b512ed7248499a44c61d954965ed9bdae4d` |
| Transformers | `4.56.1` |
| Accelerate | `1.12.0` |
| PEFT | `0.19.1` |
| Safetensors | `0.8.0` |
| Hugging Face Hub | `0.36.2` |
| hf-xet | `1.5.2` |
| sentencepiece | `0.2.2` |
| protobuf | `7.35.1` |
| Pillow | `12.3.0` |
| NumPy | `2.5.1` |
| SciPy | `1.18.0` |
| PyYAML | `6.0.3` |
| Pydantic | `2.13.4` |
| tqdm | `4.69.0` |
| Rich | `15.0.0` |
| psutil | `7.2.2` |
| pandas | `3.0.3` |
| Matplotlib | `3.11.1` |
| scikit-image | `0.26.0` |
| torchmetrics | `1.9.0` |
| LPIPS | `0.1.4` |
| MoviePy | `1.0.3` |
| imageio | `2.37.4` |
| imageio-ffmpeg | `0.6.0` |
| ipywidgets | `8.1.8` |
| anywidget | `0.11.0` |
| zstandard | `0.25.0` |

Development requirements declare `pytest==9.1.1`, `pytest-cov==7.1.0`, and
`ruff==0.14.14`. The package requires Python `>=3.10` and declares
`torch>=2.6`; the Colab workflow intentionally preserves a compatible
CUDA-enabled Colab PyTorch build instead of reinstalling it blindly. The
official BFL environment inspected used PyTorch 2.8.0 and torchvision 0.23.0.

The offline test environment was **not** the declared production lock set. It
used Python 3.12.11 and PyTorch 2.9.1 CPU; it inherited, among other versions,
Transformers 5.0.0rc3, Hugging Face Hub 1.3.5, Pydantic 2.12.5, PyYAML 6.0.2,
NumPy 2.4.0, Pillow 12.0.0, and Safetensors 0.7.0. Diffusers, PEFT, and
Accelerate were not installed in that local harness. A clean
`--ignore-installed` resolver dry-run against the exact
`requirements-colab.txt` completed successfully, so pip found a consistent
candidate resolution for the declared direct pins and their transitive
dependencies. That was resolution only: the set was not installed, imported,
or executed together, and no CUDA/Colab compatibility claim follows. The
184-pass result therefore remains evidence for the available offline contracts,
not for the pinned production stack.

## 7. Original FlowMorph behavior

For a clean latent `z`, the released interpolation initializes
`pred = clone(z)` and `u = zeros_like(z)`, defines `delta = pred - z` and
`delta_sigma = sigma_last - sigma_i`, then computes:

```text
state    = (z + delta) - delta_sigma * u
velocity = v_theta(state, timestep_i, conditioning)
z_hat    = state + delta_sigma * velocity
```

The released code minimizes the global unsquared
`torch.norm(z_hat - z)`. The paper writes squared L2 with optional `delta` and
`u` regularizers, while the released defaults set those explicit regularizers
to zero. Released AdamW leaves weight decay implicit, giving PyTorch's 0.01
default. Other released-code defaults are 512x512, 100 scheduler points,
start index 35, 100 optimizer steps per endpoint, `pred` LR 0.04, `u` LR
0.01, 20 inclusive alphas, and render indices 35/55/75/95.

Important source findings are that the paper reports FLUX.1-Depth-dev while
the repository defaults to FLUX.1-schnell; the released VAE path samples its
posterior; schedule shifting hard-codes sequence length 16; source and target
use separate prompt wrappers; rendering uses the source wrapper; and
`sampling_count=10` belongs to Flow-Optimizer rather than direct
Flow-Interpolation. The main interpolation path does not use
`prompt_interpolator.py`.

## 8. Required FLUX.2 adaptations

This project is not bit-exact paper reproduction. The deviations from the
original FlowMorph release are explicit:

| Deviation | Reason | Fidelity classification |
|---|---|---|
| FLUX.1 model/transformer replaced by undistilled FLUX.2 Klein Base 9B | requested target architecture | `required_flux2_adaptation` |
| FLUX.2 Qwen chat conditioning and four-axis text/image IDs | required by loaded Klein pipeline | `required_flux2_adaptation` |
| External conditional/unconditional CFG, default scale 4.0 | Base 9B is undistilled; Schnell path differs | `required_flux2_adaptation` |
| Deterministic VAE posterior mode, 2x2 patchification, loaded BN normalization, token packing, inverse decode | FLUX.2 latent contract differs from FLUX.1 | `required_flux2_adaptation` |
| Empirical scheduler shift from actual packed token count | released hard-coded FLUX.1 length 16 is invalid for Klein | `required_flux2_adaptation` |
| One optional native unfused transformer LoRA | requested extension; absent upstream | `required_flux2_adaptation` |
| FP32 `pred`/`u` masters with differentiable cast to model dtype | stability and auditability on Colab | `colab_execution_adaptation` |
| One model reused; source and target fit sequentially | avoids two 9B copies without changing endpoint equations | `colab_execution_adaptation` |
| Cached CPU conditioning; text encoder/VAE phase offload | peak-memory control | `colab_execution_adaptation` |
| Transparent safetensors checkpoints, exact-resume metadata, manifests, metrics, and compact archive | transient Colab/reproducibility requirements | `colab_execution_adaptation` |
| Resume validates current and staged input integrity before phase restoration or file mutation, then reuses verified preprocessing | prevents mixed-input checkpoint directories and evidence overwrite | `colab_execution_adaptation` |
| One same-profile retry after a production-probe CUDA OOM | permits graph/cache cleanup without silently changing scientific controls | `colab_execution_adaptation` |
| Separate deterministic velocity-only input-Jacobian/backward check before reconstruction backward | ordinary FlowMorph reconstruction has a direct state term, so `pred`/`u` gradients alone cannot prove transformer input differentiability; this is especially critical for FP8 | `colab_execution_adaptation` |
| Exact Diffusers 0.39.0 version, Git provenance, component classes, and 9B configuration must validate | prevents an API-compatible but uninspected runtime from being represented as the pinned implementation | `colab_execution_adaptation` |
| Required raw/display frames, endpoint state, loss history, archive, environment, and checksums cannot be disabled | required evidence must exist for acceptance | `colab_execution_adaptation` |
| Raw generated endpoints retained while display endpoints use originals | separates algorithmic evaluation from presentation | `colab_execution_adaptation` |
| Robust finite SLERP fallback for zero/opposite vectors | released numerical edge cases need deterministic handling | `colab_execution_adaptation` |
| Interpolated-embedding rendering also produces a full source-conditioned comparison render, paired sheet, and manifest | makes the prompt-conditioning extension auditable against code-parity rendering | `experimental_extension` |
| BFL FP8 single-file transformer construction, layerwise float8 storage, and injection into the full pinned Base pipeline | explicit memory experiment; never an automatic substitute | `experimental_extension` |

The default unsquared loss, learning rates, weight decay, schedule count/start,
render indices, and decoupled interpolation follow released-code parity. The
paper's mean-squared alternative exists only behind an explicit option.

## 9. CFG implementation

External guidance is implemented as:

```text
v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
```

The default is enabled at scale 4.0 with an empty negative prompt. Sequential
branch execution is the memory-oriented reference mode; a batched mode is
available only to explicitly experimental full-shape runs. Neither branch is
detached, and the low-level transformer call
uses `guidance=None` because Base 9B has `guidance_embeds=False`. The configured
`sdpa` backend maps to Diffusers v0.39's `native` backend name.

Synthetic CPU tests verify differentiability through the state/cast path. A
gated stock-step test compares both conditional-only and external-CFG custom
velocity/update behavior with the pinned high-level pipeline, and the gated
backward test exercises real CFG input gradients. Neither has **run** locally.
The BF16 comparison uses declared absolute/relative tolerance `2e-2`. Scale
1.0 is an explicit diagnostic choice only and is never selected
automatically after OOM.

## 10. Prompt-conditioning policy

Source fitting uses `source_prompt`, then `bridge_prompt`, then the neutral
default `"an image"`. Target fitting applies the corresponding target/bridge/
neutral order. `negative_prompt` supplies the unconditional CFG package.
Reference rendering uses the source conditioning package, matching the
released rendering wrapper; bridge and embedding-interpolation modes are
explicit extensions.

`interpolated_embeddings` is restricted to experimental mode and is not
accepted as a bare primary render. The runner also executes a complete
source-conditioned baseline chain, saves one baseline PNG per frame under
`conditioning_comparison/source_conditioning_frames`, writes
`comparison.json`, and builds the paired
`interpolated_vs_source.png` contact sheet. Acceptance requires all three
comparison outputs at the configured frame count, and the archive allowlist
includes them. The comparison pipeline is offline contract-tested; no real
comparison images have been generated.

Encoding delegates to `Flux2KleinPipeline.encode_prompt`, so Qwen chat
templating, hidden layers, sequence length, and feature width come from the
loaded pinned pipeline rather than duplicated constants. The inspected 9B
defaults concatenate layers 9, 18, and 27 into width 12,288 with maximum
sequence length 512. Source, target, unconditional, and optional bridge
packages contain prompt embeddings and four-axis text IDs, are hashed by
resolved prompt text, and are cached on CPU when inactive. Token IDs are never
interpolated. Actual Qwen encoding and post-cache transformer execution remain
pending gated integration.

## 11. Latent representation details

The implemented 512x512 reference path is expected to produce:

```text
VAE posterior mode   [B, 32, 64, 64]
2x2 patchification   [B, 128, 32, 32]
BN normalization     [B, 128, 32, 32]
packed image tokens  [B, 1024, 128]
image IDs            [B, 1024, 4]  # (0, h, w, 0)
```

Encoding uses posterior `mode()`, not sampling. Normalization reads the loaded
VAE's `bn.running_mean`, `bn.running_var`, and configured epsilon. Decode
reverses packing, BN normalization, and patchification before `vae.decode`;
no FLUX.1 scale/shift constants are reused. CPU tests verify patchify/pack and
normalization round trips, shapes, IDs, and deterministic helper behavior. A
gated actual-model test compares preprocessing, posterior-mode encoding,
packing/IDs, inverse normalization/unpatchification, VAE decode, and
postprocessing against the pinned pipeline helpers. That test is implemented
with declared BF16 tolerance `2e-2` but **not run** locally.

Endpoint preprocessing is deterministic RGB/EXIF handling with the resolved
resize policy and checksums. Reference configuration uses `stretch`, matching
released-code aspect-ratio behavior; `center_crop` and `contain_and_pad` are
explicit alternatives.

## 12. Scheduler and sigma behavior

The Klein schedule starts from 100 unshifted values linearly spaced from 1.0
to 0.01, then applies `compute_empirical_mu(image_seq_len, num_steps)` through
the loaded scheduler and appends terminal sigma 0. For 1,024 image tokens and
100 points, inspected `mu` is approximately `1.3446371848724212`; approximate
sigmas are 0.8769303 at index 35, 0.7584072 at 55, 0.5611979 at 75,
0.1680093 at 95, and 0 terminal.

The render chain is 35->55->75->95->terminal. Each step is:

```text
state_next = state + (sigma_next - sigma_current) * velocity
```

The low-level transformer receives `timestep / 1000`; the transformer restores
the internal scale. Pure schedule construction, monotonicity, terminal sigma,
sparse-chain indexing, constant-field Euler arithmetic, and sign conventions
are offline verified. Agreement with one stock scheduler step on a real Base-9B
latent is gated and **not run**. A production run writes actual timesteps,
sigmas, deltas, empirical shift, and scheduler configuration to `schedule.json`
and `attention_and_schedule.json`; no such production files exist yet.

## 13. Loss implementation

`code_l2_norm`, the reference default, computes one FP32 global unsquared
vector norm of `z_hat - z`, matching the released interpolation code.
`paper_l2_squared` computes `mean((z_hat - z) ** 2)`, exactly as requested for
the opt-in scale-normalized paper alternative. Optional `lambda_delta` and
`lambda_u` terms exist in the optimizer core but default to zero. No gradient
clipping is applied. The state, reconstruction, loss alternatives, signs, and
finite-gradient behavior are offline tested. No real endpoint loss curve has
been produced.

## 14. Optimizer implementation

Endpoint initialization is exactly `pred=clone(z)` and `u=zeros_like(z)`.
Both are FP32 leaf parameters in distinct AdamW groups: LR 0.04 for `pred`, LR
0.01 for `u`, and explicit weight decay 0.01 (the implicit released default).
The VAE latent target, transformer, LoRA, text encoder, conditioning, and
scheduler are frozen. The constructed state is cast to transformer compute
dtype without detaching, and the residual returns to FP32 for loss.
The optimizer validates every predicted velocity: when the optimizable state
requires gradients, a velocity tensor with `requires_grad=False` is rejected
immediately as detached from that state. This prevents a direct `state` term in
the reconstruction from masking a broken transformer gradient path.

Per-step records include total/reconstruction/regularization loss, `pred` and
`u` gradient/parameter norms, `delta` norm, elapsed time, and CUDA peaks when
available. Checkpoints default to every 25 steps plus the final step. Offline
tests confirm separate groups, FP32 leaves, BF16 cast gradients, rejection of
trainable model/conditioning tensors and detached velocities, finite gradients,
checkpoint round trip, compatibility rejection, and optimizer-state resume.
Real 9B optimization is pending.

## 15. Sequential endpoint fitting design

One frozen transformer instance is loaded. After the production backward
probe, the runner applies the fitting LoRA scale, fits and checkpoints the
source, releases its live optimizer state, then constructs the target optimizer
and fits/checkpoints the target. Target fitting cannot be represented as
completed before the source checkpoint phase. Inactive endpoint states and
conditioning remain on CPU.

Checkpoints store explicit `z`, `delta`, and `u`, schedule/start values, prompt,
input and preprocessing hashes, endpoint label, model revision, exact
Diffusers commit, FlowMorph commit, BFL FLUX.2 commit, LoRA provenance and
scale, precision, loss, and optimizer configuration. Endpoint label plus all
three code-source commits are compatibility fields, so a source checkpoint
cannot be resumed as a target and a changed implementation provenance cannot
silently reuse old optimizer state. Interrupted checkpoints may retain AdamW
moments; resume refuses any compatibility mismatch. Each save first commits a
complete tensor/metadata generation, atomically advances a `LATEST` pointer,
publishes the requested canonical files, and retains the previous complete
generation. Resume can recover from a torn newest pair without accepting
mismatched transaction IDs. Before a
resume restores phase state, collects a new environment, copies a file, or
rewrites preprocessing, it compares current source/target checksums with the
manifest, verifies staged originals and preprocessed PNGs, and checks a local
LoRA fingerprint when applicable. A valid resume loads and reuses the verified
preprocessed images without rewriting them. Offline tests demonstrate both
pre-mutation rejection of a changed input while preserving staged evidence and
mtime-stable reuse of valid persisted preprocessing. Sequential ordering is
offline tested with synthetic predictors. A persisted successful backward
probe is retained only as history: every newly prepared runner process must
execute the real input-Jacobian/VRAM probe once on its current runtime before
incomplete fitting can continue. The real 100+100 sequence is **not run**.

## 16. Gradient checkpointing behavior

Reference configuration requests transformer gradient checkpointing. The
runner enables the transformer's native checkpointing before the production
probe, leaves the transformer/adapter frozen, and keeps `pred`/`u` outside the
checkpointed model as FP32 leaves. It does not use `torch.compile`, KV caching,
or retained attention maps.

No real Base-9B checkpointing A/B was executed. In particular, numerical
closeness with checkpointing off/on, retained real LoRA effect, retained input
gradients, and reduced peak VRAM have **not been demonstrated**. Enabling this
default is therefore an implemented candidate behavior, not a validated memory
claim. An opt-in real-model A/B integration gate checks all four properties,
including lower measured peak allocation; it remained skipped here. It is a
required production gate before the profile can be called supported.

## 17. Production backward-probe results

**Not run.** The local runtime had no CUDA device or gated Base-9B weights, so
there is no production `memory_report.json`, real loss, gradient norm, elapsed
time, or VRAM peak.

The preflight no longer treats non-zero `pred`/`u` gradients as sufficient:
the reconstruction contains a direct `state` term, which could produce those
gradients even if transformer velocity were detached. It first forms a
deterministic, seed-fixed Rademacher projection of velocity alone and calls
backward on that scalar. It requires a present, finite, strictly positive
gradient at the input state and records its norm as
`velocity_input_gradient_norm`. Detached velocity, no state gradient, a
non-finite gradient, or a zero velocity-input Jacobian fails the preflight.
It then clears probe gradients, recomputes velocity/state, and performs the
ordinary FlowMorph reconstruction backward. Using backward rather than
`autograd.grad` keeps this check compatible with reentrant activation
checkpointing. This independent check is particularly important for the
experimental FP8 path.

Offline tests exercise successful, detached, and zero-Jacobian cases. A
full-shape `(1, 1024, 128)` synthetic field also produces positive
`velocity_input_gradient_norm` plus non-zero endpoint gradients with no model
gradient. These results validate the diagnostic plumbing only; they are not
evidence that the nine-billion-parameter BF16 or FP8 transformer
backpropagates to its packed input.

The renderer also fails before decode on any non-finite interpolated state,
transformer velocity, Euler-updated state, or final latent, with frame and
scheduler-step context. Endpoint/checkpoint tensors and decoded VAE pixels
have independent finite-value gates, and acceptance rejects serialized NaN or
infinite metric evidence except the mathematically valid positive-infinite
PSNR of an exactly identical image. These guards are offline tested.

The implemented preflight OOM policy permits exactly one retry after a
recognized CUDA OOM. It clears failed graph references and CUDA caches, records
both attempts and memory snapshots, and reruns the same explicitly selected
profile. The policy records and holds fixed the 9B model/revision, resolution,
scheduler points, source/target step counts, and CFG settings; it cannot switch
profile, select FP8, lower precision, or reduce semantics. A second OOM (or any
non-OOM error) fails closed and retains diagnostic evidence. This control flow
is implemented and its OOM classifier is offline tested, but an actual OOM
retry was not exercised on CUDA.

## 18. Full source fitting results

**Not run.** No 100-step source fit, production source loss CSV/plot, endpoint
checkpoint, convergence result, or source runtime exists. A 12-step synthetic
offline optimizer/checkpoint test reduced loss, but it is not a substitute for
the required Base-9B source fit.

## 19. Full target fitting results

**Not run.** No 100-step target fit, production target loss CSV/plot, endpoint
checkpoint, convergence result, or target runtime exists. Pure tests verify
that target fitting is ordered after source fitting, but do not validate the
real target endpoint.

## 20. Endpoint reconstruction metrics

**Not run.** There are no generated endpoint reconstructions and therefore no
PSNR, SSIM, LPIPS, latent error, or difference-image values for a production
run. Metric functions and serialization are offline tested on synthetic data.
The implementation preserves raw generated endpoint frames for authoritative
measurement while separately replacing the first/last display frames with the
preprocessed originals.

## 21. Transition metrics

**Not run.** There are no production adjacent-frame LPIPS/pixel/latent change
values, smoothness summaries, monotonicity observations, or transition plots.
The transition metric API and a pure twenty-frame sparse render contract are
offline tested only.

## 22. Runtime

No model-load, conditioning, VAE, backward-probe, endpoint-fit, render, decode,
metric, video, or packaging runtime has been measured on production hardware.
The only reportable timing is the final local full suite: 188 passed and 7
opt-in integration tests skipped in 1.54 seconds. `compileall` passed for
`src/` and `scripts/`; the notebook JSON and all 24 code cells parsed
successfully. These validation timings have no predictive value for the 9B
workflow.

## 23. Peak VRAM

**Not measured.** CUDA was unavailable, so allocated/reserved peaks and
headroom are unknown. No claim is made for A100 80 GB, A100 40 GB, or FP8
memory support. None of those memory profiles was attempted; therefore none is
recorded as either a successful or failed production profile. The implemented
same-profile one-retry OOM policy is a safety mechanism, not memory evidence.

## 24. GPU model

No GPU was present in the local validation environment. Recorded facts are:

```text
PyTorch: 2.9.1
torch.cuda.is_available(): False
torch.version.cuda: None
GPU model / compute capability / driver / VRAM: unavailable
```

The code can record NVIDIA device name, total/free memory, driver, compute
capability, BF16 support, CUDA and PyTorch versions on the eventual runtime,
but no such environment artifact has been produced.

## 25. LoRA activation evidence

There is no real-LoRA activation evidence because no adapter was supplied.
Offline tests verify source parsing, safetensors key/provenance/shape rejection,
active/list-adapter checks, parameter freezing, and a synthetic adapter whose
enabled output differs from its disabled output while preserving an input
gradient. This validates the test mechanism, not compatibility with any user
LoRA.

A production pass requires all of: an immutable adapter source/revision and
SHA-256, at least one mapped transformer A/B pair, native unfused load, the
expected active adapter name, frozen adapter parameters, and a finite non-zero
velocity difference for the same real latent/prompt/timestep with the adapter
disabled and enabled. None has yet been demonstrated with real weights.

## 26. Colab profile used

No Colab session or GPU profile was used. `auto` is configured in the full
template; at runtime it is designed to choose only a named Base-9B profile
from observed CUDA memory and BF16 support. Candidate profiles are:

- `a100_80gb_full`: reference BF16 candidate, untested.
- `a100_40gb_checkpointed`: unsupported until probe and full-endpoint gates.
- `fp8_9b_experimental`: explicit non-reference experiment, untested.
- `unsupported_low_vram`: diagnostic failure path, never success.

The notebook's structural contract is included in the offline 184-test pass;
its JSON and all 24 code cells also parsed successfully. It has not been
executed in standard Colab or through VS Code Colab, and its final download
action has not been exercised.

## 27. Output archive location

No production run archive exists, so there is no archive path, byte size,
SHA-256, or download result to report. The intended path is:

```text
<result_root>/<run_id>/artifacts/<run_id>.flowmorph-klein.zip
```

Offline tests build temporary synthetic archives and verify ZIP64/CRC,
checksums, member allowlisting, path safety, token scanning, exclusion of model
cache and adapter weights, 20 raw plus 20 display frame members, and inclusion
of the required experimental conditioning-comparison directory when that mode
is selected. Those temporary pytest artifacts are not deliverable research
archives.

## 28. Known artifacts

No generated production images exist, so no visual artifact was empirically
observed. The following design-visible effects should be expected and assessed:

- The four-step sparse render chain may expose discretization or abrupt
  semantic changes even when endpoint parameters fit well.
- Reference rendering uses source conditioning for all alphas, matching the
  release but potentially biasing later frames toward the source prompt.
- Experimental interpolated embeddings can introduce a different semantic
  trajectory; the mandatory source-conditioned comparison frames, report, and
  paired sheet are intended to expose that change rather than conceal it.
- Raw generated endpoint frames can differ from input images; display frames
  intentionally replace only the visible endpoints, while metrics use raw
  reconstructions.
- Linear `z`/`delta` and decoupled spherical-direction/linear-magnitude `u`
  interpolation can pass through weak or ambiguous semantics.
- Zero and antipodal `u` directions use deterministic finite fallbacks; these
  are robustness deviations from an undefined naive SLERP edge case.
- Stretch preprocessing can distort non-square sources; alternate resize modes
  are explicit and checkpointed rather than silently selected.
- A source/target LoRA scale mismatch is permitted only as an experimental
  choice and changes the fitted vector field at render time.

These are anticipated mechanisms, not claims about an unrendered result.

## 29. Failed experiments

No gated CUDA/model/LoRA experiment was attempted, so there are no production
failures to hide and no memory profile can honestly be labeled failed or
passed. The final full suite had no test failures: 188 passed and 7 gated
integration tests skipped in 1.54 seconds. With the declared `ruff==0.14.14`
installed in the project virtual environment, `.venv/bin/ruff check .`
reported `All checks passed!`. `compileall` passed for `src/` and `scripts/`,
and the notebook JSON plus every one of its 24 code cells parsed. The exact
Colab requirements also completed a clean `--ignore-installed` resolver
dry-run; they were not installed or executed. The local shell emitted a
non-fatal pyenv rehash warning because the shim directory was not writable.

Crucially, “not attempted” is not “passed”: model access, A100 80 GB, A100 40
GB, FP8, real LoRA, full fits, morph rendering, notebook execution, and final
download all remain unvalidated.

## 30. Unresolved uncertainties

1. The clean resolver dry-run succeeded, but whether that resolved dependency
   set installs/imports together on the selected current Colab image without
   changing its compatible CUDA PyTorch build remains unknown.
2. Whether the gated revision loads as a complete undistilled Base-9B pipeline
   and satisfies the structural guards.
3. Exact parity of preprocessing, VAE posterior mode, BN normalization,
   pack/unpack, and decode against the real loaded pipeline.
4. Conditional, external-CFG, timestep, dtype, and scheduler-step parity
   against the pinned high-level pipeline behavior.
5. Whether sequential and batched CFG remain numerically close by compute
   dtype and both preserve real input gradients.
6. Whether a selected user LoRA has provable Base-9B rather than distilled-9B
   provenance; architecture shapes alone cannot distinguish those variants.
7. Whether the real adapter maps non-zero tensors, remains active/frozen, and
   changes velocity at the requested scale during both fitting and rendering.
8. Whether gradient checkpointing preserves output/LoRA/input gradients and
   actually lowers peak VRAM.
9. Whether A100 80 GB can complete the 512x512 CFG backward probe with a
   positive real `velocity_input_gradient_norm` and then complete 100-step
   endpoint fits; A100 40 GB support is still more uncertain.
10. The FP8 download/construction/layerwise-storage/injection path is
    implemented, but whether it executes successfully and preserves input
    gradients, PEFT hooks, checkpointing, and real memory savings remains
    unknown.
11. Full-fit convergence, reconstruction quality, transition smoothness,
    runtime, video encoding, archive size, Colab/VS Code Colab behavior, and
    Drive/download reliability.
12. The pinned Diffusers SHA is inspected and immutable but cannot be promoted
    to “tested” until all mandatory integration gates pass.

## 31. Reproduction commands

From the project root, create the intended environment and run the offline
static/full collection checks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --dry-run --ignore-installed -r requirements-colab.txt
python -m pip install -r requirements-colab.txt -r requirements-dev.txt
pytest -q
python -m compileall -q src scripts
.venv/bin/ruff check .
```

Without the explicit integration environment gates, the seven real CUDA/model
tests skip rather than masquerading as passes. The successful resolver dry-run
reported above used the exact `requirements-colab.txt`; the actual install line
is a reproduction instruction, not a claim that the pinned stack was installed
in the local CPU harness.

On a licensed CUDA runtime, first accept the model terms and authenticate
through environment/Colab secrets or interactive Hub login; never paste or
print a token. Then run the mandatory production-shape probe with actual input
paths:

```bash
python scripts/validate_colab.py \
  --config configs/full_9b_lora.yaml \
  --source /content/flowmorph_klein_images/max_v1/images/source.png \
  --target /content/flowmorph_klein_images/max_v1/images/target.png \
  --profile a100_80gb_full
```

Run the reference workflow with a compatible adapter:

```bash
python scripts/run_flowmorph.py \
  --config configs/full_9b_lora.yaml \
  --source /content/flowmorph_klein_images/max_v1/images/source.png \
  --target /content/flowmorph_klein_images/max_v1/images/target.png \
  --lora-source org/repository \
  --lora-scale 1.0 \
  --profile a100_80gb_full
```

Resume only an exactly compatible run/configuration:

```bash
python scripts/resume_flowmorph.py \
  --config configs/full_9b_lora.yaml \
  --source /content/flowmorph_klein_images/max_v1/images/source.png \
  --target /content/flowmorph_klein_images/max_v1/images/target.png \
  --lora-source org/repository \
  --lora-scale 1.0 \
  --profile a100_80gb_full
```

The opt-in integration tests require the global gate plus each costly gate;
for example, after supplying a compatible LoRA source through a secure
environment configuration:

```bash
export FLOWMORPH_RUN_FLUX2_9B_INTEGRATION=1
export FLOWMORPH_RUN_PRODUCTION_BACKWARD_INTEGRATION=1
export FLOWMORPH_RUN_LORA_INTEGRATION=1
export FLOWMORPH_RUN_CHECKPOINTING_AB_INTEGRATION=1
export FLOWMORPH_RUN_FULL_ENDPOINT_INTEGRATION=1
export FLOWMORPH_RUN_THREE_FRAME_INTEGRATION=1
export FLOWMORPH_RUN_FULL_MORPH_INTEGRATION=1
export FLOWMORPH_TEST_LORA_SOURCE=org/repository
pytest -m integration -ra
```

These commands describe the intended reproduction; this report does not imply
that the CUDA commands were executed.

## 32. Recommended next experiments

1. Build a clean Colab environment from the declared pins, record `pip freeze`
   and `pip check`, and resolve any dependency conflict before model download.
2. On A100 80 GB, verify gated Base-9B load, exact revision/class/configuration,
   a stock 512x512 generation, and environment recording.
3. Run real preprocessing/VAE encode-decode, schedule sign, and no-CFG/CFG
   low-level velocity parity by FP32/BF16 tolerance.
4. Supply one immutable Base-9B LoRA revision; inspect provenance and shapes,
   load it unfused, record its fingerprint, prove a non-zero velocity effect,
   and prove adapter parameters remain frozen.
5. Compare gradient checkpointing off/on for output closeness, `pred`/`u`
   gradients, LoRA activation, runtime, and peak allocated/reserved VRAM.
6. Pass the real 512x512 sequential-CFG production backward probe on A100 80
   GB before starting optimization, including a finite positive
   `velocity_input_gradient_norm` from the independent velocity-only backward.
7. Complete and inspect one 100-step source fit, including resume from a
   25-step checkpoint, gradients, loss curve, memory, and reconstruction.
8. Complete the target fit, then the three-frame smoke render, before the full
   20-frame render and metric/video/archive phases.
9. Execute the notebook end-to-end in both standard Colab and VS Code Colab;
   validate the final archive, token scan, checksum, compactness, Drive copy,
   and download action.
10. Only after the BF16 reference passes, test A100 40 GB. Treat FP8 as a
    separate experimental study requiring fresh gradient, LoRA, parity, and
    memory evidence rather than as an automatic fallback.
