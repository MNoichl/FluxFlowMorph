# Research notes

Status date: 2026-07-20. These notes distinguish source inspection from runtime
validation. The gated 9B model was not available in the local CPU-only
workspace, so no statement below presents an unexecuted CUDA check as passed.

## Revisions and dependency envelope

| Source or package | Revision/version | Status |
|---|---:|---|
| VITA-Group/FlowMorph | `0db52344ad0ec6963f74a508db831e506058b2f7` | inspected |
| Diffusers | `a3608b512ed7248499a44c61d954965ed9bdae4d` (v0.39.0 commit) | inspected; selected reproducibility pin |
| Diffusers main during research | `86e6dac5360703ddf09fe250db50be667eb93662` | compared, not selected |
| BFL FLUX.2 inference repository | `50fe5162777813d869182b139e83b10743caef15` | inspected |
| FLUX.2 Klein Base 9B model | `32773329fbe7e81a90ef971740e8ba4b0364ecf3` | repository metadata inspected; weights gated |
| FLUX.2 Klein Base 9B FP8 | `9ecf2143d71542449960c5584340269c6d401449` | repository metadata inspected; execution unverified |
| Transformers | `4.56.1` | candidate pin from official BFL environment; GPU run pending |
| Accelerate | `1.12.0` | candidate pin from official BFL environment; GPU run pending |
| PEFT | `0.19.1` | candidate pin; exact-model adapter run pending |
| Safetensors | `0.8.0` | Diffusers v0.39 minimum and project pin |
| Hugging Face Hub | `0.36.2` | compatible project pin (`transformers==4.56.1` requires `<1.0`) |
| PyTorch | local `2.9.1` CPU; official BFL `2.8.0` | Colab-provided compatible build is preferred |
| CUDA | unavailable locally | production validation pending |

The full Diffusers commit is pinned even though project-level 9B validation is
pending. It must only be promoted from “inspected candidate” to “tested” in the
implementation report after all four gates pass: 9B load, VAE roundtrip,
scheduler/velocity parity, compatible LoRA activation, and an input-gradient
backward probe.

Primary sources:

- [Final FlowMorph WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Zheng_FlowMorph_Revealing_an_Optimizable_Flow_Latent_Space_for_Controlled_Image_WACV_2026_paper.pdf)
- [Pinned FlowMorph repository](https://github.com/VITA-Group/FlowMorph/tree/0db52344ad0ec6963f74a508db831e506058b2f7)
- [FLUX.2 Klein Base 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
- [Pinned BFL FLUX.2 repository](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15)
- [Pinned Diffusers repository](https://github.com/huggingface/diffusers/tree/a3608b512ed7248499a44c61d954965ed9bdae4d)

## FlowMorph files inspected

The audit covered `README.md`, `docs/METHOD.md`, `configs/default.yaml`,
`flowmorph/flow_interpolation.py`, `flowmorph/flux_optim.py`,
`flowmorph/utils.py`, `flowmorph/prompt_interpolator.py`,
`scripts/run_flow_interpolation.sh`, and
`scripts/ablations/run_prompt_separation.sh`.

For a clean packed endpoint latent `z`, optimizable prediction `pred`, and
optimizable one-step vector `u`:

```text
delta       = pred - z
delta_sigma = sigma_last - sigma_i
state       = (z + delta) - delta_sigma * u
            = pred - delta_sigma * u
velocity    = v_theta(state, timestep_i, conditioning)
z_hat       = state + delta_sigma * velocity
```

The paper minimizes squared L2 with optional regularizers:

```text
||z_hat - z||_2^2 + lambda_delta ||delta||_2^2 + lambda_u ||u||_2^2
```

The released interpolation path instead uses the unsquared global
`torch.norm(z_hat - z)`. Both explicit regularizers default to zero. This
project names the alternatives `code_l2_norm` and `paper_l2_squared`, with the
released-code behavior as the reference default.

Released-code defaults are 512×512, 100 scheduler points, start index 35, 100
optimization steps per endpoint, AdamW with learning rates 0.04 (`pred`) and
0.01 (`u`), 20 inclusive linear alpha values, and sparse render indices
35/55/75/95 followed by a terminal update. AdamW is called without an explicit
weight decay, so PyTorch's default `0.01` applies. The interpolation initializer
is exactly `pred=clone(z)`, `u=zeros_like(z)`. `sampling_count=10` is used by
Flow-Optimizer, not by Flow-Interpolation.

Paper/release differences and release quirks:

1. The paper uses FLUX.1-Depth-dev for reported experiments; the repository
   defaults to FLUX.1-schnell.
2. The paper says squared L2; the released interpolation uses unsquared L2.
3. The paper shows explicit regularizers; released defaults set them to zero,
   while AdamW still applies implicit weight decay 0.01.
4. The paper calls the representation `z_{t_i}`, but released interpolation
   directly VAE-encodes the clean image and does not forward-noise it.
5. Released FLUX.1 VAE encoding samples the posterior without a generator;
   this adaptation deliberately uses the deterministic FLUX.2 pipeline mode.
6. Released optimization tensors inherit BF16; this adaptation uses FP32
   master tensors and casts the transformer input without detaching it.
7. Released schedule shifting uses a hard-coded sequence length 16. This
   adaptation uses the actual packed FLUX.2 token count.
8. Source and target are fit with separate prompts/wrappers. Rendering always
   calls the source wrapper, so the code-parity render condition is `source`.
9. `prompt_interpolator.py` is unused by the main interpolation path.
10. The prompt-separation ablation changes prompts and wrappers together. Its
    shared branch tries to re-encode after text encoders were deleted and still
    creates a new solver/AdamW, so it neither isolates nor truly shares the
    optimizer state.

## FLUX.2 and Diffusers files inspected

The audit covered:

- `pipelines/flux2/pipeline_flux2_klein.py`
- `pipelines/flux2/image_processor.py`
- `models/transformers/transformer_flux2.py`
- `models/attention_processor.py`
- `models/autoencoders/autoencoder_kl_flux2.py`
- `schedulers/scheduling_flow_match_euler_discrete.py`
- `loaders/lora_pipeline.py`
- `loaders/lora_conversion_utils.py`
- the official Klein DreamBooth LoRA example

Exact classes/helpers used are `Flux2KleinPipeline`,
`Flux2Transformer2DModel`, `AutoencoderKLFlux2`,
`FlowMatchEulerDiscreteScheduler`, `Flux2LoraLoaderMixin`,
`retrieve_latents`, `compute_empirical_mu`, `retrieve_timesteps`,
`_prepare_text_ids`, `_prepare_latent_ids`, `_patchify_latents`,
`_unpatchify_latents`, `_pack_latents`, `_unpack_latents_with_ids`,
`_encode_vae_image`, `encode_prompt`, `load_lora_weights`, `set_adapters`,
`get_active_adapters`, and `get_list_adapters`.

At 512×512, the expected Base-9B latent path is dynamic but normally has these
shapes:

```text
VAE posterior mode       [B, 32, 64, 64]
2x2 patchification       [B, 128, 32, 32]
BN normalization         [B, 128, 32, 32]
token packing            [B, 1024, 128]
image IDs                [B, 1024, 4] = (0, h, w, 0)
```

Normalization uses the loaded VAE's `bn.running_mean`,
`sqrt(bn.running_var + vae.config.batch_norm_eps)`. Decode reverses packing,
normalization, and patchification before `vae.decode`. FLUX.1 scaling and shift
constants are not applicable.

Prompt encoding uses Qwen chat formatting with thinking disabled, sequence
length 512, and hidden layers 9, 18, and 27 concatenated into a 12,288-wide
embedding for the 9B model. These are pipeline defaults read through the
loaded API rather than constants in the numerical core. Text IDs use
`(0,0,0,token_index)`.

The low-level Base-9B call passes `timestep / 1000` into the transformer,
which multiplies by 1000 internally. `guidance=None` is mandatory because Base
9B has `guidance_embeds=False`. External classifier-free guidance is:

```text
v_cfg = v_uncond + scale * (v_cond - v_uncond)
```

The stock Klein pipeline evaluates the branches separately. Its public call
uses the empty prompt for the unconditional branch. Supporting a configurable
negative prompt is a documented project extension using the same encoder.

The pipeline supplies custom sigmas, not the scheduler's implicit defaults:

```python
sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
mu = compute_empirical_mu(image_seq_len, num_steps)
scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
```

For 1024 image tokens and 100 points, `mu` is approximately
`1.3446371848724212`. Inspected approximate sigmas are 0.8769303 (35),
0.7584072 (55), 0.5611979 (75), 0.1680093 (95), and 0 terminal. The
deterministic scheduler update is exactly
`sample + (sigma_next - sigma) * model_output`.

Diffusers v0.39 names its PyTorch SDPA attention backend `native`. The user
configuration value `sdpa` therefore maps to `native`; passing `sdpa` directly
to the backend selector is invalid at this pin.

## FLUX.1 Schnell versus FLUX.2 Klein Base 9B

| Topic | Released FlowMorph / FLUX.1 Schnell | This adaptation / FLUX.2 Klein Base 9B |
|---|---|---|
| Model type | distilled Schnell | undistilled Base 9B |
| Prompt encoders | FLUX.1 pipeline encoders | Qwen3 causal LM hidden-state stack |
| Guidance | no external CFG in released path | external conditional/unconditional CFG, default 4.0 |
| Transformer | `FluxTransformer2DModel` | `Flux2Transformer2DModel` |
| VAE | FLUX.1 VAE and scale/shift | `AutoencoderKLFlux2`, 32 channels, VAE BN statistics |
| Packing | FLUX.1-specific packing | 2×2 patchify, BN normalize, spatial token pack |
| Position IDs | three-axis FLUX.1 layout | four-axis `(T,H,W,L)` layout |
| Endpoint posterior | stochastic `.sample()` in released code | deterministic `.mode()` matching Klein pipeline |
| Schedule shift | released hard-coded sequence length 16 | actual FLUX.2 packed token count and empirical mu |
| Timestep | divided by 1000 at call | divided by 1000 at call; FLUX.2 transformer restores scale internally |
| LoRA | not part of released core | native unfused Flux2 transformer LoRA, frozen and verified numerically |

## LoRA source and format findings

`Flux2LoraLoaderMixin` loads transformer adapters only. Native Diffusers,
Kohya, ai-toolkit/FAL `diffusion_model.*`, and PEFT
`base_model.model.*` paths are recognized upstream. Version 1 rejects LoHa,
LoKr, and unrelated formats. Base and distilled 9B share their architecture,
so tensor shapes alone cannot prove Base provenance; metadata/repository
provenance is required and ambiguous distilled provenance needs an explicit
override. A load exception-free result is insufficient: active adapter names
and a non-zero deterministic velocity difference are mandatory.

The FP8 repository is a transformer artifact rather than a complete pipeline.
It requires a Base-9B configuration and injection into the full pipeline.
Input gradients, retained FP8 storage, gradient checkpointing, and PEFT
compatibility are all unverified.

## Memory choices and compromises

These choices change execution, not the FlowMorph endpoint equations:

- One transformer instance is reused; endpoints fit sequentially.
- Source, target, bridge, and negative prompt packages are cached once.
- The text encoder leaves GPU memory after prompt caching.
- The VAE leaves GPU memory after endpoint encoding and returns for decoding.
- CFG runs sequentially by default to reduce concurrent activation memory.
- Transformer gradient checkpointing is enabled only after its parity and
  input-gradient tests pass.
- `pred` and `u` remain FP32 master tensors; transformer inputs are cast.
- Model/sequential CPU offload is experimental during fitting.
- No resolution, step, frame, CFG, architecture, or model-precision reduction
  is performed automatically.

## Unverified assumptions and required gates

The following remain unverified until run on a licensed CUDA Colab runtime:

1. Gated Base-9B weight access and complete pipeline loading.
2. Custom preprocessing, VAE mode encode, packed representation, and decode
   parity against the pinned pipeline with the actual VAE configuration.
3. Sparse scheduler update parity using the actual model scheduler config.
4. Conditional and CFG low-level velocity parity by compute dtype.
5. A real user LoRA's Base-9B provenance, mapped tensor count, active scale,
   and non-zero velocity effect.
6. Backpropagation from the 9B transformer output to a 512×512 packed input.
7. Numerical closeness and VRAM benefit of gradient checkpointing.
8. Complete 100-step source and target fits and twenty-frame render.
9. A100 80 GB and 40 GB memory support.
10. FP8 input gradients, adapter support, and actual memory savings.

