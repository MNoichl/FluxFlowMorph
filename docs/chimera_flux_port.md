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
2. Calibrate normalized radial FFT prototypes for representative FLUX layer
   groups and dedicated per-timestep velocity outputs, then fit a robust
   coarse-to-fine layer schedule from their L1 distances.
3. Cache the frequency-matched diffusion feature during pair inversion.
4. Spherically interpolate both endpoint latents and endpoint caches.
5. Use a linear inversion-denoising timestep mapping (IDM) when retrieving a
   cache during forward denoising.
6. Add the LTM-selected interpolated feature as an ACI residual at all three
   representative FLUX depth groups with the published default weight `0.4`.
7. Generate one image-aware intermediate/SAP prompt per pair with a VLM,
   require a normalized endpoint-relative bridge score of `0.45`, and append
   its valid anchor tokens during the first `20%` of denoising steps only.

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
magnitude in exact channel chunks. LTM-v2 normalizes these descriptors by total
band energy so differences in activation scale between transformer depths do
not dominate the comparison. `calibrate_flux_ltm()` forms layer prototypes
from the three representative hooks and timestep prototypes from the model's
conditional velocity output rather than averaging the candidate layers back
together. A radius-one temporal mean suppresses single-step spectral jitter.
The independent minimum-L1 matches remain in the calibration report; a
constrained fit turns healthy matches into one monotonic early-to-middle-to-
late schedule. A collapsed one-group argmin is rejected and transparently
falls back to fixed coarse-to-fine thirds. Hooks cache image tokens only. At
render time, the LTM-selected cache is injected at the early, middle, and late
representatives, matching CHIMERA's all-group ACI structure while retaining a
bounded FLUX hook set. SAP may change the number of text tokens without
invalidating the image-feature boundary.

The production notebook calibrates on a configurable set of unique, evenly spaced anchors
from the active cycle; requesting more anchors than exist never duplicates a
sample. This is model-, LoRA-, resolution-, prompt-, and image-specific, is
saved to Google Drive, and is reused across every pair. Only the small FFT
statistics, mapping diagnostics, and final 50-step mapping are persisted. Setting
`CHIMERA_LTM_MODE="linear"` explicitly restores the old fixed-third fallback.

## Memory contract

Standalone base-anchor generation defaults to
`BASE_PIPELINE_CPU_OFFLOAD=False`. On a high-VRAM GPU this keeps the fused
text encoder, transformer, and VAE resident and avoids copying the 9B model
between CPU and CUDA for every anchor. Set the switch to `True` for smaller
GPUs; rerunning the load cell detects a residency-mode change and rebuilds the
pipeline instead of reusing an incompatible instance. Regardless of this
choice, the base pipeline is deleted, hooks are released, garbage collection
is run, and the CUDA allocator cache is emptied immediately before the
separate CHIMERA runner is prepared. Base and intermediate-generation model
instances therefore never intentionally coexist on the GPU.

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
decode phase starts with all configured interiors and persists its existing
OOM backoff result.

Set stride `1` and storage `float32` for the closest published-algorithm
contract. Pair caches are never retained for the whole flat sequence:
after each pair's interiors are decoded and its completion manifest is safely
written, both endpoint caches are deleted.

The active trajectory-fidelity experiment raises the notebook calibration set
to eight anchors and uses stride `1` with `float16` storage. At 1024px this is a
practical middle ground: every inversion step is retained and quantization is
removed, while CPU cache storage remains about half of float32. These settings
do not keep pair caches on the GPU and therefore do not directly reduce the
learned transformer microbatch size.

## Symmetric velocity smoothing

CHIMERA's endpoint interpolation does not itself couple adjacent midpoint
denoising trajectories. The notebook therefore exposes a weak, explicitly
non-paper extension through `CHIMERA_VELOCITY_SMOOTHING_STRENGTH`. At every
Euler step, the renderer first predicts velocities for the complete ordered
alpha trajectory in memory-bounded microbatches. It then pulls each interior
velocity toward the alpha-aware linear interpolation of both neighbours before
performing the Euler update. The first and last interior velocities remain
unchanged, and reversing the endpoint direction produces the reversed
smoothing operation. Neighbour interpolation and the resulting Euler velocity
remain float32, so weak corrections are not repeatedly rounded away in
bfloat16 even though transformer inference retains its native model dtype.

This loop ordering is important: microbatches limit transformer activation
memory but are not allowed to define independent trajectory segments. A
strength of `0.10` is deliberately conservative. The applied per-frame
velocity RMS delta is included in conditioning diagnostics so the intervention
can be audited rather than inferred from the output images.

## Semantic Anchor Prompting

The original prompt-only notebook already used a signed API call to create
image-aware midpoint text. The CHIMERA notebook sends both endpoint prompts
and images to the VLM and requests one stable intermediate/SAP prompt per
pair; authored endpoint prompts are not rewritten. Reliability is measured
from padding-masked FLUX text embeddings rather than the paper's CLIP text
encoder. The score is normalized relative to the endpoint-to-endpoint cosine:
zero is the direct endpoint baseline and one is identity with both endpoints.
This gives the `0.45` threshold a generator-native meaning instead of copying
a CLIP-specific cosine scale. Weak proposals trigger at most three bounded
re-queries.

SAP is implemented by concatenating up to 64 valid anchor embedding tokens to
the conditional text context during the first 20% of Euler steps. Padding is
excluded, and capped prompts are sampled across the complete valid Qwen chat
sequence rather than truncated at the first 64 positions. Early SAP steps use
sequential external CFG because conditional and unconditional token counts
differ; later steps return to the configured batched CFG path.

Endpoint prompt embeddings follow a norm-preserving spherical interpolation
by default. This avoids the midpoint norm loss possible under direct linear
averaging of semantically different endpoint embeddings. Every render records
the active and hypothetical linear embedding norms plus mean CFG-residual RMS
for the SAP and post-SAP phases. These diagnostics are persisted in each pair's
cache report, and the interpolation mode is part of the pair fingerprint so
linear-conditioning results cannot be silently reused as SLERP results.

## Perceptual spacing and video timing

The notebook supports an endpoint-preserving sinusoidal experimental schedule:

`alpha(u) = u + s * sin(2*pi*u) / (2*pi)`

Positive `CHIMERA_ALPHA_WARP_STRENGTH` moves samples on either side toward the
midpoint while retaining the same number of FLUX renders. The active setting
is `0`, restoring uniform CHIMERA coefficients and avoiding enlarged gaps next
to the real endpoint anchors. The strength is stored in pair fingerprints and
manifests, so cached uniform results cannot be mistaken for warped results.

RIFE then measures reduced-resolution mean pixelwise CIE76 distance for every
cyclic source-frame edge. `allocate_perceptual_subdivisions()` gives larger
edges more interpolation subdivisions and smaller edges fewer, subject to a
minimum, maximum, and robust median-relative distance cap. Its integer
allocation preserves exactly `pair_count * RIFE_MULTIPLIER` subdivisions, so
adaptive allocation does not increase the total RIFE image budget. The
existing dense circular SSIM resampling remains as a final timing pass. The
active notebook renders 24 dense RIFE candidates per source edge, then selects
an average of eight at 24 fps for an exact fourfold slowdown from the nominal
12 fps source sequence. Dense candidate count and playback duration are kept
separate so additional RIFE work improves temporal selection without silently
making the output slower.

The optional temporal correction stage also provides a chroma-only mode,
enabled in the CHIMERA notebook independently of its luminance/contrast
outlier correction. It measures mean OKLab chroma, linearly interpolates a
reference between every pair of final-round endpoint anchors, and pulls the
measured trajectory 70% toward that line. A Whittaker second-difference solve
then smooths the desired output trajectory itself while fixing every endpoint;
it does not smooth the correction curve. Signed OKLab scaling can therefore
reduce early overshoots as well as lift later deficits. Adjustments below 1%
are ignored, increases are capped at 12%, and decreases at 8%. OKLab lightness
and hue are retained except for unavoidable output-gamut clipping.

The stage keeps the raw CHIMERA PNGs, writes corrected copies only for frames
that receive a change, and saves both a JSON audit and a two-panel plot of raw,
endpoint-line, solved-target, and corrected chroma plus the signed adjustment
trajectory. The report includes linear-target error and curvature RMS before
and after correction. RIFE consumes the corrected paths when chroma
stabilization is enabled.

The one-gap quality gate now mirrors the first production round rather than
using a separate five-alpha sample. It renders the configured midpoint count
with the configured alpha warp, then runs the same endpoint-anchored temporal
tone/chroma correction used by the full sequence. The gate preserves a raw
contact sheet, a representative corrected sheet, the correction JSON, and the
before/after chroma plot. Its printed summary includes the repository commit,
warp strength, and exact alpha values so a stale or nonrepresentative run is
immediately visible.

## GLCS

`compute_glcs_from_similarities()` implements the paper's GCS, LCS, and
geometric-mean GLCS aggregation for any bounded similarity function. The
notebook offers an optional DINOv2 similarity audit after releasing FLUX. The
paper's supplement found DINO had the highest agreement with its user study
among the tested DiffSim alternatives. GLCS is computed per completed pair,
not across the whole multi-anchor cycle.

## Known limitations

- The paper's dataset-level FFT prototypes were not released. This port derives
  run-level prototypes from up to four unique evenly spaced cyclic anchors; increase
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
