# CHIMERA port for FLUX.2 Klein

This repository includes a prompt-only cyclic implementation inspired by
[CHIMERA: Adaptive Cache Injection and Semantic Anchor Prompting for Zero-shot
Image Morphing with Morphing-oriented Metrics](https://arxiv.org/abs/2512.07155)
(Kye et al., ECCV 2026). The paper's
[official repository](https://github.com/CMLab-Korea/ECCV26-CHIMERA) still
listed the main implementation and GLCS release as pending when this port was
written. This is therefore an inspectable port of the published algorithm, not
a claim of code-level reproduction.

## What is preserved

The implementation follows the paper's algorithm:

1. Encode each endpoint image and reverse the native deterministic sampler to
   obtain an inverted endpoint latent.
2. Calibrate radial FFT prototypes for the representative FLUX layer groups
   and every inversion timestep, then select the minimum-L1 group per step.
3. Cache the frequency-matched diffusion feature during pair inversion.
4. Spherically interpolate both endpoint latents and endpoint caches.
5. Use a linear inversion-denoising timestep mapping (IDM) when retrieving a
   cache during forward denoising.
6. Add the interpolated feature as an ACI residual with the published default
   weight `0.4`.
7. Construct an anchor-correlated prompt triplet with a VLM, require a minimum
   endpoint-anchor cosine similarity of `0.45`, and append the shared anchor
   tokens during the first `20%` of denoising steps only.

The notebook uses the paper's FLUX setting: the native Euler flow-matching
sampler with 50 inversion and 50 denoising steps. LoRA activation, model
revision pinning, image encoding/decoding, Google Drive manifests, flat
cyclic assembly, flicker diagnostics, and RIFE finishing reuse the existing
repository infrastructure.

The checked-in CHIMERA notebook is the source of truth for its prompts,
settings, markdown, outputs, and experimental cells. Its companion builder no
longer composes it from another notebook or a parallel prompt file: by default
the command only validates the authored notebook, and it can export an exact
copy only to a path that does not already exist. There is no force-overwrite
escape hatch. Future repository changes should use explicit marker-based
migrations that edit only their declared cells.

A new run seed is drawn from OS entropy, saved as `metadata/run_seed.json`, and
reused when that run is resumed; rerunning into a new run directory therefore
produces new anchors.

The Colab setup cell always fetches `REPOSITORY_REF`, checks out the freshly
fetched commit, clears previously imported `flowmorph_klein` modules, and then
imports the package from the checked-out `src` directory. Rerunning that cell
therefore refreshes implementation code even when the repository is already
present in the runtime.

## FLUX architecture mapping

The main CHIMERA derivation describes U-Net down, mid, and up features. FLUX.2
Klein is a diffusion transformer. The paper's supplement reports the analogous
coarse-to-fine pattern in FLUX: earlier transformer layers emphasize lower
frequencies, later layers emphasize higher-frequency detail, and denoising
steps exhibit the same coarse-to-fine progression.

`select_flux_feature_groups()` maps the centers of the first, middle, and last
transformer-depth thirds to `early`, `middle`, and `late` representatives.
`radial_frequency_descriptor()` computes the channel-mean, radially pooled FFT
magnitude in exact channel chunks. `calibrate_flux_ltm()` averages descriptors
over a calibration anchor set to form layer and timestep prototypes, then uses
the paper's per-timestep minimum-L1 rule. Hooks cache and inject image tokens
only; SAP may change the number of text tokens without invalidating the
image-feature boundary.

The production notebook calibrates on four evenly spaced anchors from the
active cycle. This is model-, LoRA-, resolution-, prompt-, and image-specific,
is saved to Google Drive, and is reused across every pair. Only the small FFT
statistics and final 50-step mapping are persisted. Setting
`CHIMERA_LTM_MODE="linear"` explicitly restores the old fixed-third fallback.

## Memory contract

Uncompressed 1024px features from a 9B transformer are very large. Calibration
therefore stores descriptors rather than full features from all three groups.
After calibration, only the selected group is cached for each timestep. The
notebook also exposes two consequential settings:

- `CHIMERA_CACHE_STORAGE="int8"` stores symmetric per-feature int8 caches on
  CPU and dequantizes only the active feature to the transformer's dtype.
- `CHIMERA_CACHE_STRIDE=2` captures every other inversion step and uses
  deterministic nearest-step retrieval.

Base anchors cannot be batched because each painting conditions the next one.
Midpoint rendering instead begins with batch 2 and measures the first successful
CUDA peak. It preserves the larger of 10% or 2 GiB as free memory, pads the
observed per-item requirement by 25%, and probes upward only to that guarded
ceiling. OOM results establish an upper bound and subsequent attempts use binary
backoff; the learned successful batch is retained across gaps. The VAE-only
decode phase starts with all ten interiors and persists its existing OOM
backoff result.

Set stride `1` and storage `float32` for the closest published-algorithm
contract. Pair caches are never retained for the whole flat sequence:
after each pair's interiors are decoded and its completion manifest is safely
written, both endpoint caches are deleted.

## Semantic Anchor Prompting

The original prompt-only notebook already used a signed API call to create
image-aware midpoint text. The CHIMERA notebook replaces that contract with
three mutually correlated prompts: a shared semantic/structural anchor and two
anchor-conditioned endpoint descriptions. Reliability is measured with pooled
FLUX text embeddings rather than the paper's CLIP text encoder, avoiding a
second text model and keeping the gate aligned with the generator actually in
use. Weak triplets trigger at most three bounded re-queries.

SAP is implemented by concatenating up to 64 anchor embedding tokens to the
conditional text context during the first 20% of Euler steps. Early SAP steps
use sequential external CFG because conditional and unconditional token counts
differ; later steps return to the configured batched CFG path.

Endpoint prompt embeddings follow a norm-preserving spherical interpolation
by default. This avoids the midpoint norm loss possible under direct linear
averaging of semantically different endpoint embeddings. Every render records
the active and hypothetical linear embedding norms plus mean CFG-residual RMS
for the SAP and post-SAP phases. These diagnostics are persisted in each pair's
cache report, and the interpolation mode is part of the pair fingerprint so
linear-conditioning results cannot be silently reused as SLERP results.

## GLCS

`compute_glcs_from_similarities()` implements the paper's GCS, LCS, and
geometric-mean GLCS aggregation for any bounded similarity function. The
notebook offers an optional DINOv2 similarity audit after releasing FLUX. The
paper's supplement found DINO had the highest agreement with its user study
among the tested DiffSim alternatives. GLCS is computed per completed pair,
not across the whole multi-anchor cycle.

## Known limitations

- The paper's dataset-level FFT prototypes were not released. This port derives
  run-level prototypes from four evenly spaced cyclic anchors; increase
  `CHIMERA_LTM_CALIBRATION_ANCHORS` for a broader but slower calibration set.
- This port cannot reproduce unpublished feature-layer choices, VLM instruction
  details, or reference-code numerical behavior.
- Representative depth hooks are less comprehensive than caching every FLUX
  block. Caching all 32 9B blocks at all 50 steps is not a practical Colab
  default.
- Int8 cache quantization and stride 2 are engineering approximations. They can
  be disabled explicitly.
- Reverse Euler inversion is deterministic but not guaranteed to reconstruct
  endpoints exactly under a finite-step learned vector field. The final cyclic
  sequence uses the original endpoint PNGs, as image morphing normally does.
- As reported in the paper, typography and prominent text can break or change
  abruptly in diffusion morphs.

## Entry points

- Notebook:
  `notebooks/StillLife_Recursive_CHIMERA_Prompt_Only.ipynb`
- Non-mutating notebook validator/safe exporter (legacy builder entry point):
  `scripts/build_recursive_chimera_prompt_only_notebook.py`
- Core implementation: `src/flowmorph_klein/chimera.py`
- CPU contract tests: `tests/test_chimera.py` and
  `tests/test_chimera_prompt_only_notebook.py`
