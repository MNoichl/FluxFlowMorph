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

The implementation follows the paper's six-stage algorithm:

1. Encode each endpoint image and reverse the native deterministic sampler to
   obtain an inverted endpoint latent.
2. Cache multi-depth diffusion features during inversion.
3. Spherically interpolate both endpoint latents and endpoint caches.
4. Use a linear inversion-denoising timestep mapping (IDM) when retrieving a
   cache during forward denoising.
5. Add the interpolated feature as an ACI residual with the published default
   weight `0.4`.
6. Construct an anchor-correlated prompt triplet with a VLM, require a minimum
   endpoint-anchor cosine similarity of `0.45`, and append the shared anchor
   tokens during the first `20%` of denoising steps only.

The notebook uses the paper's FLUX setting: the native Euler flow-matching
sampler with 50 inversion and 50 denoising steps. LoRA activation, model
revision pinning, image encoding/decoding, Google Drive manifests, flat
cyclic assembly, flicker diagnostics, and RIFE finishing reuse the existing
repository infrastructure.

## FLUX architecture mapping

The main CHIMERA derivation describes U-Net down, mid, and up features. FLUX.2
Klein is a diffusion transformer. The paper's supplement reports the analogous
coarse-to-fine pattern in FLUX: earlier transformer layers emphasize lower
frequencies, later layers emphasize higher-frequency detail, and denoising
steps exhibit the same coarse-to-fine progression.

`select_flux_feature_groups()` maps the centers of the first, middle, and last
transformer-depth thirds to `early`, `middle`, and `late` representatives.
`flux_depth_ltm()` matches the corresponding denoising thirds. Hooks cache and
inject image tokens only; SAP may change the number of text tokens without
invalidating the image-feature boundary.

This depth prior is the port's Layer- and Timestep-wise Frequency Matching
(LTM) approximation. The paper's reference LTM uses offline, dataset-averaged
radial FFT prototypes, but those prototypes and the code used to produce them
were not released. `radial_frequency_descriptor()` implements the paper's
descriptor for calibration experiments without presenting pair-specific
statistics as the missing dataset-level reference calibration.

## Memory contract

Uncompressed 1024px features from a 9B transformer are very large. The notebook
therefore exposes two consequential settings:

- `CHIMERA_CACHE_STORAGE="int8"` stores symmetric per-feature int8 caches on
  CPU and dequantizes only the active feature to the transformer's dtype.
- `CHIMERA_CACHE_STRIDE=2` captures every other inversion step and uses
  deterministic nearest-step retrieval.

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

## GLCS

`compute_glcs_from_similarities()` implements the paper's GCS, LCS, and
geometric-mean GLCS aggregation for any bounded similarity function. The
notebook offers an optional DINOv2 similarity audit after releasing FLUX. The
paper's supplement found DINO had the highest agreement with its user study
among the tested DiffSim alternatives. GLCS is computed per completed pair,
not across the whole multi-anchor cycle.

## Known limitations

- This port cannot reproduce unpublished feature-layer choices, offline FFT
  prototypes, VLM instruction details, or reference-code numerical behavior.
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
- Notebook builder:
  `scripts/build_recursive_chimera_prompt_only_notebook.py`
- Core implementation: `src/flowmorph_klein/chimera.py`
- CPU contract tests: `tests/test_chimera.py` and
  `tests/test_chimera_prompt_only_notebook.py`
