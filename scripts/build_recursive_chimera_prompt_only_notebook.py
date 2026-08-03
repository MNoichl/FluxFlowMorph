"""Build the prompt-only flat CHIMERA/FLUX production notebook.

The generated notebook retains the prompt-only anchor generator, Google Drive
layout, pinned FLUX.2/LoRA loader, temporal diagnostics, and RIFE finishing
stack from ``StillLife_Recursive_FlowMorph_Prompt_Only``.  Its morphing core is
replaced with the repository's CHIMERA port (reverse Euler inversion, ACI,
IDM, SAP, and optional DINO-backed GLCS evaluation).
"""

from __future__ import annotations

import json
import os
import runpy
import tempfile
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_recursive_flowmorph_prompt_only_notebook.py"
OUTPUT = Path(
    os.environ.get(
        "CHIMERA_PROMPT_ONLY_NOTEBOOK_OUTPUT",
        ROOT / "notebooks" / "StillLife_Recursive_CHIMERA_Prompt_Only.ipynb",
    )
)
if (
    "CHIMERA_PROMPT_ONLY_NOTEBOOK_OUTPUT" not in os.environ
    and OUTPUT.exists()
    and os.environ.get("FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE") != "1"
):
    raise RuntimeError(
        "Refusing to overwrite the tracked CHIMERA notebook. Set "
        "CHIMERA_PROMPT_ONLY_NOTEBOOK_OUTPUT or explicitly set "
        "FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE=1."
    )


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Expected one settings block bounded by {start!r} and {end!r}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + dedent(replacement).strip("\n") + "\n\n" + end + after


@contextmanager
def temporary_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


with tempfile.TemporaryDirectory(prefix="chimera_prompt_notebook_") as temp:
    base_path = Path(temp) / "base.ipynb"
    with temporary_environment({"FLOWMORPH_PROMPT_ONLY_NOTEBOOK_OUTPUT": str(base_path)}):
        runpy.run_path(str(BASE_BUILDER), run_name="__main__")
    notebook = json.loads(base_path.read_text(encoding="utf-8"))


notebook["cells"][0]["source"] = lines(
    """
    # Prompt-only recursive still-life loop - CHIMERA for FLUX.2 + LoRA

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MNoichl/FluxFlowMorph/blob/main/notebooks/StillLife_Recursive_CHIMERA_Prompt_Only.ipynb)

    This version keeps the prompt-only still-life anchor workflow and replaces
    endpoint fitting with a FLUX-native port of CHIMERA (Kye et al., ECCV 2026):

    1. Generate the editable cyclic `BASE_STAGES` as RIJKSOIL anchors.
    2. Ask a vision-language model for one anchor-correlated prompt triplet per gap.
    3. Calibrate Layer- and Timestep-wise Frequency Matching (LTM) once from a
       representative anchor subset using the paper's radial FFT descriptors.
    4. Reverse the native 50-step FLUX Euler ODE for both endpoint images while
       caching the frequency-matched transformer feature at each timestep.
    5. Slerp the inverted latents and caches, map inversion to denoising steps
       with IDM, and inject the matched cache through ACI (default weight 0.4).
    6. Append the shared semantic-anchor tokens only during the first 20% of
       denoising, then release that constraint for fine-detail formation.
    7. Render one flat ten-interior pass, save every image directly to Google
       Drive, optionally score completed pairs with DINO-backed GLCS, and finish
       the exact cyclic PNG sequence with the existing RIFE/SSIM video stack.

    The official CHIMERA code was not public when this port was written.  FLUX
    has no U-Net down/mid/up blocks, so this implementation follows the paper's
    FLUX appendix and maps them to representative transformer depths, then
    measures their timestep correspondence instead of assuming fixed thirds.
    Int8 CPU caches and cache stride 2 are exposed memory controls; use
    float32/stride 1 for the closest (and much larger) cache contract.
    """
)
notebook["cells"][1]["source"] = lines(
    """
    ## 1. Editable prompt-only run, generation, CHIMERA, and video settings

    `BASE_STAGES` remains the only creative input.  The CHIMERA defaults mirror
    the paper where a value is published (`lambda=0.4`, SAP ratio `0.2`, anchor
    reliability `0.45`, and 50 FLUX Euler steps).  The cache compression and
    stride are explicit Colab memory adaptations.
    """
)

settings = source(notebook["cells"][2])
settings = settings.replace(
    'PROJECT_NAME = "science_path_prompt_only_flowmorph"',
    'PROJECT_NAME = "science_path_prompt_only_chimera"',
)
settings = settings.replace(
    "# Optional post-FlowMorph tonal correction; raw PNGs are never overwritten.",
    "# Optional post-CHIMERA tonal correction; raw PNGs are never overwritten.",
)
settings = settings.replace(
    "# Read-only diagnosis of raw cyclic FlowMorph output.",
    "# Read-only diagnosis of raw cyclic CHIMERA output.",
)
settings = settings.replace("RUN_FLOWMORPH_ONE_GAP_TEST", "RUN_CHIMERA_ONE_GAP_TEST")
settings = settings.replace("FLOWMORPH_ONE_GAP_TEST_INDEX", "CHIMERA_ONE_GAP_TEST_INDEX")
settings = settings.replace("FLOWMORPH_ONE_GAP_TEST_ALPHAS", "CHIMERA_ONE_GAP_TEST_ALPHAS")
settings = replace_between(
    settings,
    "# Editable anchor selection and recursive insertion.\n",
    "# FLUX.2 Klein Base 9B + RIJKSOIL LoRA.\n",
    """
    # Editable anchor selection and one flat CHIMERA pass.
    BASE_PROMPT_COUNT = None  # None uses every entry in BASE_STAGES.
    CHIMERA_ROUND_SPECS = [
        {"midpoint_count": 10},
    ]
    INTERPOLATION_ROUNDS = len(CHIMERA_ROUND_SPECS)
    REGENERATE_BASE_FRAMES = True
    REUSE_EXISTING_MIDPOINTS = True
    RESUME_CHIMERA_SEQUENCE = True

    # CHIMERA for the native FLUX.2 Euler flow-matching sampler.
    CHIMERA_INVERSION_STEPS = 50
    CHIMERA_DENOISING_STEPS = 50
    CHIMERA_ACI_WEIGHT = 0.4
    CHIMERA_SAP_ACTIVE_RATIO = 0.2
    CHIMERA_ANCHOR_RELIABILITY_THRESHOLD = 0.45
    CHIMERA_SAP_MAX_REQUERIES = 3
    CHIMERA_ANCHOR_MAX_TOKENS = 64
    CHIMERA_LTM_MODE = "fft"  # "fft" (paper method) or explicit "linear" fallback.
    CHIMERA_LTM_BANDS = 16
    CHIMERA_LTM_CHANNEL_CHUNK_SIZE = 128  # Exact chunked FFT; affects memory, not results.
    CHIMERA_LTM_CALIBRATION_ANCHORS = 4  # Evenly sampled from the active cyclic anchors.
    REUSE_CHIMERA_LTM_CALIBRATION = True
    CHIMERA_CACHE_STRIDE = 2  # 1 caches every inversion step.
    CHIMERA_CACHE_STORAGE = "int8"  # int8, float16, bfloat16, or float32.
    CHIMERA_GUIDANCE_SCALE = 7.0
    CHIMERA_LORA_SCALE = 1.2
    CHIMERA_RENDER_BATCH_SIZE = 2
    CHIMERA_DECODE_BATCH_SIZE = 4
    CHIMERA_CFG_EXECUTION = "batched"
    CHIMERA_BATCH_OOM_BACKOFF = True
    CHIMERA_STREAM_PAIRS_PER_CHUNK = 1
    CHIMERA_STREAM_DISPLAY_PROGRESS = True
    OPENAI_CONCURRENCY = 6

    # Optional pairwise paper metric. DINO was the best-performing alternative
    # similarity in the paper's supplemental comparison; disabled by default to
    # avoid another model download.
    RUN_CHIMERA_DINO_GLCS = False
    CHIMERA_DINO_MODEL_ID = "facebook/dinov2-base"
    CHIMERA_GLCS_GAMMA = 2.0
    """,
)
settings = settings.replace(
    "BASE_REFERENCE_BLUR = 16.0\n"
    "BASE_REFERENCE_GRAIN_STRENGTH = 0.035  # Normalized monochrome noise sigma; 0 disables.\n"
    "BASE_REFERENCE_DENOISE_STRENGTH = 0.75\n",
    "BASE_REFERENCE_STRENGTH = 0.3  # Blend of the blurred previous anchor into the warm field.\n"
    "BASE_REFERENCE_BLUR = 16.0\n"
    "BASE_REFERENCE_GRAIN_STRENGTH = 0.035  # Normalized monochrome noise sigma; 0 disables.\n"
    "REFERENCE_BACKGROUND = (116, 105, 91)\n",
)
notebook["cells"][2]["source"] = lines(settings)

notebook["cells"][9]["source"] = lines(
    """
    ## 5. Validate settings and preview the flat-run cost

    The estimate reports pairwise inversions and denoising batches because they
    dominate CHIMERA runtime.  It also validates every exposed memory control
    before the 9B model is downloaded.
    """
)
notebook["cells"][10]["source"] = lines(
    """
    import math

    if BASE_PROMPT_COUNT is None:
        BASE_PROMPT_COUNT = len(BASE_STAGES)
    elif not 3 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):
        raise ValueError(f"BASE_PROMPT_COUNT must be between 3 and {len(BASE_STAGES)}")
    if len(CHIMERA_ROUND_SPECS) != 1:
        raise ValueError("This flat notebook expects exactly one CHIMERA round")
    for index, spec in enumerate(CHIMERA_ROUND_SPECS, start=1):
        if not 1 <= int(spec.get("midpoint_count", 0)) <= 20:
            raise ValueError(f"Round {index} midpoint_count must be between 1 and 20")
    if CHIMERA_ROUND_SPECS != [{"midpoint_count": 10}]:
        raise ValueError("The production contract is one flat pass with ten interiors per gap")
    if not (256 <= IMAGE_WIDTH <= 2048 and IMAGE_WIDTH % 16 == 0):
        raise ValueError("IMAGE_WIDTH must be 256-2048 and divisible by 16")
    if not (256 <= IMAGE_HEIGHT <= 2048 and IMAGE_HEIGHT % 16 == 0):
        raise ValueError("IMAGE_HEIGHT must be 256-2048 and divisible by 16")
    if CHIMERA_INVERSION_STEPS != CHIMERA_DENOISING_STEPS:
        raise ValueError("This port currently requires equal inversion and denoising step counts")
    if CHIMERA_DENOISING_STEPS != 50:
        raise ValueError("The paper's FLUX setting uses 50 Euler steps")
    if not 0 <= CHIMERA_ACI_WEIGHT <= 2:
        raise ValueError("CHIMERA_ACI_WEIGHT must lie in [0, 2]")
    if not 0 <= CHIMERA_SAP_ACTIVE_RATIO <= 1:
        raise ValueError("CHIMERA_SAP_ACTIVE_RATIO must lie in [0, 1]")
    if not -1 <= CHIMERA_ANCHOR_RELIABILITY_THRESHOLD <= 1:
        raise ValueError("CHIMERA_ANCHOR_RELIABILITY_THRESHOLD must lie in [-1, 1]")
    if CHIMERA_SAP_MAX_REQUERIES < 0 or CHIMERA_CACHE_STRIDE < 1:
        raise ValueError("CHIMERA requery count and cache stride are invalid")
    if CHIMERA_LTM_MODE not in {"fft", "linear"}:
        raise ValueError("CHIMERA_LTM_MODE must be fft or linear")
    if CHIMERA_LTM_BANDS < 2 or CHIMERA_LTM_CHANNEL_CHUNK_SIZE < 1:
        raise ValueError("CHIMERA LTM bands/chunk size are invalid")
    if not 1 <= CHIMERA_LTM_CALIBRATION_ANCHORS <= BASE_PROMPT_COUNT:
        raise ValueError("CHIMERA_LTM_CALIBRATION_ANCHORS is outside the active anchor range")
    if CHIMERA_CACHE_STORAGE not in {"int8", "float16", "bfloat16", "float32"}:
        raise ValueError("Unsupported CHIMERA_CACHE_STORAGE")
    if CHIMERA_CFG_EXECUTION not in {"sequential", "batched"}:
        raise ValueError("CHIMERA_CFG_EXECUTION must be sequential or batched")
    if CHIMERA_LORA_SCALE != IMAGE_LORA_SCALE:
        raise ValueError("CHIMERA and anchor-generation LoRA scales must match")
    if CHIMERA_GUIDANCE_SCALE != IMAGE_GUIDANCE_SCALE:
        raise ValueError("CHIMERA and anchor-generation guidance scales must match")
    for name, value in {
        "CHIMERA_RENDER_BATCH_SIZE": CHIMERA_RENDER_BATCH_SIZE,
        "CHIMERA_DECODE_BATCH_SIZE": CHIMERA_DECODE_BATCH_SIZE,
        "CHIMERA_STREAM_PAIRS_PER_CHUNK": CHIMERA_STREAM_PAIRS_PER_CHUNK,
        "OPENAI_CONCURRENCY": OPENAI_CONCURRENCY,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= CHIMERA_ONE_GAP_TEST_INDEX < BASE_PROMPT_COUNT:
        raise ValueError("CHIMERA_ONE_GAP_TEST_INDEX is outside the active anchor range")
    if (
        not CHIMERA_ONE_GAP_TEST_ALPHAS
        or CHIMERA_ONE_GAP_TEST_ALPHAS != sorted(set(CHIMERA_ONE_GAP_TEST_ALPHAS))
        or any(not 0.0 < alpha < 1.0 for alpha in CHIMERA_ONE_GAP_TEST_ALPHAS)
    ):
        raise ValueError("CHIMERA_ONE_GAP_TEST_ALPHAS must be unique sorted interior values")
    if not 1 <= IMAGE_INFERENCE_STEPS <= 100 or not 0 <= IMAGE_GUIDANCE_SCALE <= 20:
        raise ValueError("Anchor inference settings are invalid")
    if not 0 < IMAGE_LORA_SCALE <= 4:
        raise ValueError("IMAGE_LORA_SCALE must lie in (0, 4]")
    if not 0 <= BASE_REFERENCE_GRAIN_STRENGTH <= 0.25:
        raise ValueError("BASE_REFERENCE_GRAIN_STRENGTH must lie in [0, 0.25]")
    if not 0 < BASE_REFERENCE_STRENGTH <= 1:
        raise ValueError("BASE_REFERENCE_STRENGTH must lie in (0, 1]")
    if not 32 <= FLUX_PROMPT_MAX_SEQUENCE_LENGTH <= 512:
        raise ValueError("FLUX_PROMPT_MAX_SEQUENCE_LENGTH must lie in [32, 512]")
    if OPENAI_IMAGE_DETAIL not in {"low", "high", "original", "auto"}:
        raise ValueError("OPENAI_IMAGE_DETAIL must be low, high, original, or auto")
    if VIDEO_SLOWDOWN_FACTOR < 1:
        raise ValueError("VIDEO_SLOWDOWN_FACTOR must be at least 1")

    ACTIVE_BASE_STAGES = BASE_STAGES[:BASE_PROMPT_COUNT]
    ids = [item["id"] for item in ACTIVE_BASE_STAGES]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_]+", item) for item in ids):
        raise ValueError("Anchor IDs must be unique lowercase snake_case values")
    for item in ACTIVE_BASE_STAGES:
        if not item["science"].strip() or not item["prompt"].strip():
            raise ValueError(f"Blank science or prompt in {item['id']}")
        if item["prompt"].casefold().count(LORA_TRIGGER.casefold()) != 1:
            raise ValueError(f"{item['id']} must contain the LoRA trigger exactly once")

    round_counts = [BASE_PROMPT_COUNT]
    for spec in CHIMERA_ROUND_SPECS:
        round_counts.append(round_counts[-1] * (int(spec["midpoint_count"]) + 1))
    pair_count = sum(round_counts[:-1])
    denoise_batches = sum(
        round_counts[index] * math.ceil(
            int(CHIMERA_ROUND_SPECS[index]["midpoint_count"]) / CHIMERA_RENDER_BATCH_SIZE
        )
        for index in range(len(CHIMERA_ROUND_SPECS))
    )
    print({
        "anchor_images": BASE_PROMPT_COUNT,
        "sequence_counts": round_counts,
        "openai_anchor_triplets": pair_count,
        "endpoint_inversions": pair_count * 2,
        "inversion_transformer_calls": pair_count * 2 * CHIMERA_INVERSION_STEPS,
        "one_time_ltm_calibration_inversions": (
            CHIMERA_LTM_CALIBRATION_ANCHORS if CHIMERA_LTM_MODE == "fft" else 0
        ),
        "denoising_batches": denoise_batches,
        "denoising_transformer_calls_before_cfg": denoise_batches * CHIMERA_DENOISING_STEPS,
        "ltm_contract": (
            f"{CHIMERA_LTM_MODE}, bands={CHIMERA_LTM_BANDS}, "
            f"calibration_anchors={CHIMERA_LTM_CALIBRATION_ANCHORS}"
        ),
        "cache_contract": f"{CHIMERA_CACHE_STORAGE}, stride={CHIMERA_CACHE_STRIDE}",
        "final_generated_sequence_images": round_counts[-1],
        "rife_multiplier": RIFE_MULTIPLIER,
    })
    print("Anchor order:", " -> ".join(ids), "->", ids[0])
    """
)

notebook["cells"][11]["source"] = lines(
    """
    ## 6. Load RIJKSOIL and optionally render one anchor trial

    This is unchanged from the prompt-only source notebook: it validates the
    pinned FLUX.2 model, LoRA, guidance, prompt length, and image geometry before
    the full CHIMERA session is loaded.
    """
)

# Restore the original recursive-vision anchor initialization.  The previous
# painting is softened, mixed against the warm background, grained, and passed
# directly to the FLUX pipeline as ``image=``.  Do not manually construct the
# initial latent/sigma schedule here; that newer path produced poorer anchors.
model_source = source(notebook["cells"][12]).replace(
    "from flowmorph_klein.trajectory import prepare_flux2_klein_img2img_inputs\n",
    "",
)
notebook["cells"][12]["source"] = model_source.splitlines(keepends=True)
notebook["cells"][13]["source"] = lines(
    """
    ## 7. Generate prompt-only cyclic anchor paintings

    The first anchor is ordinary text-to-image.  For every later anchor, the
    original recursive-vision method blurs the previous painting, blends it into
    a warm neutral field, adds faint deterministic monochrome grain, and passes
    that PIL image directly to FLUX through `image=`.  This keeps broad continuity
    without manually replacing the pipeline's native noise or sigma initialization.
    """
)
notebook["cells"][14]["source"] = lines(
    """
    from flowmorph_klein.art_loop import make_soft_reference

    BASE_DIRECTORY = RUN_DIRECTORY / "base_frames"
    REFERENCE_DIRECTORY = BASE_DIRECTORY / "soft_references"
    BASE_MANIFEST_PATH = RUN_DIRECTORY / "metadata" / "base_manifest.json"
    BASE_RECORDS = []

    def generate_prompt_anchor(prompt, seed, reference=None):
        validate_flux_prompt_length(prompt, "Anchor generation prompt")
        kwargs = {
            "prompt": prompt,
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "num_inference_steps": IMAGE_INFERENCE_STEPS,
            "guidance_scale": IMAGE_GUIDANCE_SCALE,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "output_type": "pil",
            "max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
        }
        generation_mode = "text_to_image"
        if reference is not None:
            kwargs["image"] = reference
            generation_mode = "native_pipeline_image_conditioning"
        result = FLUX_PIPE(**kwargs)
        if not result.images:
            raise RuntimeError("FLUX returned no anchor image")
        return result.images[0].convert("RGB"), generation_mode

    if not REGENERATE_BASE_FRAMES and BASE_MANIFEST_PATH.is_file():
        BASE_RECORDS = json.loads(
            BASE_MANIFEST_PATH.read_text(encoding="utf-8")
        )["records"]
        missing = [
            item["path"]
            for item in BASE_RECORDS
            if not Path(item["path"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing resumed anchor images: " + ", ".join(missing)
            )
        resumed_contract = [
            (record["uid"], record["science"], record["prompt"])
            for record in BASE_RECORDS
        ]
        current_contract = [
            (f"base_{index:03d}", stage["science"], stage["prompt"])
            for index, stage in enumerate(ACTIVE_BASE_STAGES)
        ]
        if resumed_contract != current_contract:
            raise RuntimeError(
                "Editable anchor prompts differ from the saved anchors. "
                "Regenerate or resume the matching run."
            )
        print(f"Loaded {len(BASE_RECORDS)} existing anchor records.")
    else:
        BASE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        previous = None
        for index, stage in enumerate(ACTIVE_BASE_STAGES):
            seed = BASE_SEED + index
            reference = None
            reference_path = None
            if previous is not None and BASE_CONTINUITY_ENABLED:
                reference = make_soft_reference(
                    previous,
                    reference_blend=BASE_REFERENCE_STRENGTH,
                    blur_radius=BASE_REFERENCE_BLUR,
                    grain_strength=BASE_REFERENCE_GRAIN_STRENGTH,
                    grain_seed=seed,
                    background_rgb=REFERENCE_BACKGROUND,
                )
                if SAVE_SOFT_REFERENCES:
                    REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                    reference_path = (
                        REFERENCE_DIRECTORY / f"reference_{index:03d}.png"
                    )
                    reference.save(reference_path, format="PNG", compress_level=4)
            image, generation_mode = generate_prompt_anchor(
                stage["prompt"],
                seed,
                reference=reference,
            )
            output_path = BASE_DIRECTORY / f"{index:03d}_{stage['id']}.png"
            image.save(output_path, format="PNG", compress_level=4)
            record = {
                "uid": f"base_{index:03d}",
                "kind": "base",
                "round": 0,
                "science": stage["science"],
                "prompt": stage["prompt"],
                "generation_prompt": stage["prompt"],
                "generation_prompt_token_count": validate_flux_prompt_length(
                    stage["prompt"],
                    "Saved anchor prompt",
                ),
                "seed": seed,
                "path": str(output_path),
                "soft_reference_path": (
                    str(reference_path) if reference_path else None
                ),
                "base_continuity_used": reference is not None,
                "base_reference_source": "original_recursive_vision_soft_reference",
                "base_reference_strength": BASE_REFERENCE_STRENGTH,
                "base_reference_blur": BASE_REFERENCE_BLUR,
                "base_reference_grain_strength": BASE_REFERENCE_GRAIN_STRENGTH,
                "reference_background": list(REFERENCE_BACKGROUND),
                "generation_mode": generation_mode,
            }
            BASE_RECORDS.append(record)
            BASE_MANIFEST_PATH.write_text(json.dumps({
                "project": PROJECT_NAME,
                "complete": len(BASE_RECORDS) == len(ACTIVE_BASE_STAGES),
                "records": BASE_RECORDS,
            }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
            if previous is not None:
                previous.close()
            previous = image.copy()
            image.close()
            if reference is not None:
                reference.close()
            print(
                f"Anchor {index + 1}/{len(ACTIVE_BASE_STAGES)} saved: "
                f"{output_path.name}"
            )
        if previous is not None:
            previous.close()

    if len(BASE_RECORDS) != len(ACTIVE_BASE_STAGES):
        raise RuntimeError("The anchor manifest is incomplete.")
    print(f"Prepared {len(BASE_RECORDS)} cyclic anchors in {BASE_DIRECTORY}")
    """
)
notebook["cells"][16]["source"] = lines(
    """
    ## 8. Define CHIMERA Semantic Anchor Prompting (SAP)

    Each request receives the two actual endpoint paintings, their literal
    prompts, and their sciences.  It returns an anchor-correlated triplet:
    `anchor_prompt`, `prompt_a`, and `prompt_b`.  The shared anchor describes the
    semantic/structural intersection; both endpoint prompts describe their own
    image while explicitly sharing that concept.  FLUX pooled-embedding cosine
    similarity applies the paper's 0.45 reliability gate and triggers bounded
    VLM re-queries when necessary.
    """
)
notebook["cells"][17]["source"] = lines(
    r'''
    import base64
    import hashlib
    import io
    import time
    from pydantic import BaseModel, Field, ValidationError

    class ChimeraPromptTriplet(BaseModel):
        science_connection: str = Field(min_length=20, max_length=800)
        visual_correspondence: str = Field(min_length=20, max_length=1200)
        anchor_prompt: str = Field(min_length=120, max_length=1800)
        prompt_a: str = Field(min_length=180, max_length=2400)
        prompt_b: str = Field(min_length=180, max_length=2400)

    CHIMERA_SAP_SYSTEM_PROMPT = f"""
    Role: You are an art director constructing Semantic Anchor Prompting for a
    CHIMERA image morph rendered by FLUX.2 Klein with the {LORA_TRIGGER} LoRA.

    Inspect both attached still-life paintings. Return three mutually correlated,
    literal visual prompts:
    - anchor_prompt: the shared semantic or structural intersection of both images;
    - prompt_a: a faithful standalone description of painting A that naturally and
      explicitly contains the shared anchor concept;
    - prompt_b: a faithful standalone description of painting B that naturally and
      explicitly contains the same shared anchor concept.

    Requirements for all three prompts:
    - Begin exactly with "{LORA_TRIGGER}," and contain that trigger exactly once.
    - Describe a plausible seventeenth-century Dutch Baroque oil still life with
      concrete objects, positions, silhouette, material, color, lighting, negative
      space, support geometry, layered glazes, and restrained impasto.
    - No people and no readable text.
    - Be standalone literal image descriptions, never editing instructions.
    - Fit within {FLUX_PROMPT_MAX_SEQUENCE_LENGTH} tokens after FLUX's chat template.

    The anchor must be genuinely shared, not a third unrelated scene. The endpoint
    prompts must remain faithful to their respective attached image and must not
    collapse both images into the same description.

    Do not use production words such as source image, target image, endpoint,
    frame, interpolation, morph, transition, halfway, retain, preserve, unchanged,
    or same. Output only the structured fields.
    """.strip()

    FORBIDDEN_CHIMERA_PROMPT_TERMS = (
        "source image", "target image", "endpoint", "frame", "interpolation",
        "morph", "transition", "halfway", "retain", "preserve", "unchanged",
    )

    def image_data_url(path):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((VISION_IMAGE_MAX_SIDE, VISION_IMAGE_MAX_SIDE))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=VISION_JPEG_QUALITY, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_chimera_prompt(prompt, label):
        clean = " ".join(prompt.split())
        if not clean.startswith(f"{LORA_TRIGGER},"):
            raise ValueError(f"{label} must begin exactly with {LORA_TRIGGER},")
        if clean.casefold().count(LORA_TRIGGER.casefold()) != 1:
            raise ValueError(f"{label} must contain the LoRA trigger exactly once")
        found = [
            term for term in FORBIDDEN_CHIMERA_PROMPT_TERMS
            if re.search(rf"\b{re.escape(term)}\b", clean, re.I)
        ]
        if found:
            raise ValueError(f"{label} contains production terms: {', '.join(found)}")
        validate_flux_prompt_length(clean, label)
        return clean

    def validate_chimera_triplet(proposal):
        proposal.anchor_prompt = validate_chimera_prompt(proposal.anchor_prompt, "Anchor prompt")
        proposal.prompt_a = validate_chimera_prompt(proposal.prompt_a, "Endpoint-A prompt")
        proposal.prompt_b = validate_chimera_prompt(proposal.prompt_b, "Endpoint-B prompt")
        return proposal

    def extract_parsed_triplet(response):
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return parsed
        refusals = []
        for output in response.output:
            if output.type != "message":
                continue
            for item in output.content:
                if item.type == "refusal":
                    refusals.append(item.refusal)
                elif getattr(item, "parsed", None) is not None:
                    return item.parsed
        if refusals:
            raise RuntimeError("OpenAI refused the SAP request: " + " | ".join(refusals))
        raise RuntimeError("OpenAI response contained no parsed SAP triplet")

    def chimera_request_fingerprint(left, right):
        contract = {
            "model": OPENAI_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "image_detail": OPENAI_IMAGE_DETAIL,
            "system_prompt": CHIMERA_SAP_SYSTEM_PROMPT,
            "left_uid": left["uid"],
            "left_science": left["science"],
            "left_prompt": left["prompt"],
            "left_image_sha256": file_sha256(left["path"]),
            "right_uid": right["uid"],
            "right_science": right["science"],
            "right_prompt": right["prompt"],
            "right_image_sha256": file_sha256(right["path"]),
        }
        serialized = json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest(), contract

    def propose_chimera_triplet(left, right, reliability_feedback=None):
        feedback = ""
        if reliability_feedback:
            feedback = (
                "\n\nThe prior triplet failed the embedding reliability gate. "
                f"{reliability_feedback} Rewrite all three prompts around a clearer "
                "shared concrete semantic/structural concept."
            )
        request_text = f"""
        Painting A sciences: {left['science']}
        Painting A generation prompt: {left['prompt']}

        Painting B sciences: {right['science']}
        Painting B generation prompt: {right['prompt']}

        Inspect both attached paintings and return the anchor-correlated triplet.
        {feedback}
        """.strip()
        correction = ""
        last_error = None
        for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
            try:
                response = OPENAI_CLIENT.responses.parse(
                    model=OPENAI_MODEL,
                    reasoning={"effort": OPENAI_REASONING_EFFORT},
                    store=False,
                    max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                    input=[
                        {"role": "system", "content": CHIMERA_SAP_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": request_text + correction},
                                {"type": "input_text", "text": "Painting A:"},
                                {"type": "input_image", "image_url": image_data_url(left["path"]), "detail": OPENAI_IMAGE_DETAIL},
                                {"type": "input_text", "text": "Painting B:"},
                                {"type": "input_image", "image_url": image_data_url(right["path"]), "detail": OPENAI_IMAGE_DETAIL},
                            ],
                        },
                    ],
                    text_format=ChimeraPromptTriplet,
                )
                proposal = validate_chimera_triplet(extract_parsed_triplet(response))
                return proposal, response
            except (ValidationError, json.JSONDecodeError, ValueError) as error:
                last_error = error
                correction = (
                    f"\n\nThe prior response failed validation: {error}. Return shorter, "
                    "complete structured fields that satisfy every prompt rule."
                )
                if attempt < OPENAI_MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise RuntimeError(f"SAP triplet failed after {attempt} attempts") from error
        raise RuntimeError(f"SAP triplet generation failed: {last_error}")

    print("CHIMERA anchor-correlated SAP prompt contract ready.")
    '''
)

notebook["cells"][18]["source"] = lines(
    """
    ## 9. One-model CHIMERA session and optional one-gap quality gate

    The standalone fused anchor generator is released.  One unfused LoRA-aware
    FLUX.2 model is then retained for all reverse Euler inversions and CHIMERA
    denoising calls.  The gate renders a single cyclic gap before the full run.
    """
)
notebook["cells"][19]["source"] = lines(
    r'''
    import gc
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from flowmorph_klein.chimera import (
        ChimeraConfig,
        ChimeraFlux2Session,
        LTMCalibration,
        prompt_anchor_reliability,
        select_flux_feature_groups,
    )
    from flowmorph_klein.cli import select_hardware_profile
    from flowmorph_klein.config import ProjectTemplateConfig, load_config, resolve_config
    from flowmorph_klein.pipeline import FlowMorphRunner

    def validate_chimera_runner_contract(config):
        for name, value in (("width", config.input.width), ("height", config.input.height)):
            if not 256 <= value <= 2048 or value % 16 != 0:
                raise ValueError(f"input.{name} must be 256-2048 and divisible by 16")
        if config.flowmorph.scheduler_points != CHIMERA_DENOISING_STEPS:
            raise ValueError("Runner schedule must match CHIMERA_DENOISING_STEPS")

    ProjectTemplateConfig._validate_full_shape_contract = validate_chimera_runner_contract

    BASE_PROMPT_TOKEN_COUNTS = {
        record["uid"]: validate_flux_prompt_length(record["prompt"], f"{record['uid']} prompt")
        for record in BASE_RECORDS
    }
    print({"anchor_prompt_token_counts": BASE_PROMPT_TOKEN_COUNTS})
    release_flux_pipeline()

    def usage_payload(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return usage.model_dump(mode="json") if hasattr(usage, "model_dump") else str(usage)

    def stable_fingerprint(payload):
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def load_or_create_triplet(left, right, round_number, gap_index, path):
        request_fingerprint, request_contract = chimera_request_fingerprint(left, right)
        if REUSE_EXISTING_MIDPOINTS and path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("request_fingerprint") == request_fingerprint:
                proposal = validate_chimera_triplet(
                    ChimeraPromptTriplet.model_validate(saved["proposal"])
                )
                return {
                    "left": left,
                    "right": right,
                    "round": round_number,
                    "gap_index": gap_index,
                    "proposal_path": path,
                    "proposal": proposal,
                    "response_id": saved.get("openai_response_id"),
                    "usage": saved.get("usage"),
                    "request_fingerprint": request_fingerprint,
                    "request_contract": request_contract,
                    "reliability": saved.get("reliability"),
                    "requeries": int(saved.get("requeries", 0)),
                }
        proposal, response = propose_chimera_triplet(left, right)
        return {
            "left": left,
            "right": right,
            "round": round_number,
            "gap_index": gap_index,
            "proposal_path": path,
            "proposal": proposal,
            "response_id": response.id,
            "usage": usage_payload(response),
            "request_fingerprint": request_fingerprint,
            "request_contract": request_contract,
            "reliability": None,
            "requeries": 0,
        }

    def proposal_prompts(job):
        proposal = job["proposal"]
        return (proposal.anchor_prompt, proposal.prompt_a, proposal.prompt_b)

    def save_triplet_job(job):
        path = Path(job["proposal_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "round": job["round"],
            "gap_index": job["gap_index"],
            "left_uid": job["left"]["uid"],
            "right_uid": job["right"]["uid"],
            "request_fingerprint": job["request_fingerprint"],
            "request_contract": job["request_contract"],
            "proposal": job["proposal"].model_dump(mode="json"),
            "reliability": job["reliability"],
            "threshold": CHIMERA_ANCHOR_RELIABILITY_THRESHOLD,
            "requeries": job["requeries"],
            "openai_model": OPENAI_MODEL,
            "openai_response_id": job["response_id"],
            "usage": job["usage"],
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    CHIMERA_ROOT = RUN_DIRECTORY / "chimera_sequence"
    CHIMERA_SESSION_DIRECTORY = CHIMERA_ROOT / "session"
    CHIMERA_ASSET_ROOT = CHIMERA_ROOT / "encoded_inputs"
    for directory in (CHIMERA_ROOT, CHIMERA_SESSION_DIRECTORY, CHIMERA_ASSET_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    bootstrap_left = BASE_RECORDS[0]
    bootstrap_right = BASE_RECORDS[1]
    bootstrap_schedule = [bootstrap_left["prompt"], bootstrap_left["prompt"], bootstrap_right["prompt"]]
    session_overrides = {
        "run_mode": "experimental",
        "project.name": f"{PROJECT_NAME}_chimera_session",
        "model.id": MODEL_ID,
        "model.revision": MODEL_REVISION,
        "lora.source": str(LOCAL_LORA_PATH),
        "lora.revision": None,
        "lora.weight_name": LOCAL_LORA_PATH.name,
        "lora.adapter_name": LORA_ADAPTER_NAME,
        "lora.fit_scale": CHIMERA_LORA_SCALE,
        "lora.render_scale": CHIMERA_LORA_SCALE,
        "lora.require_base_9b_compatibility": False,
        "lora.allow_distilled_9b": True,
        "input.source_image": str(bootstrap_left["path"]),
        "input.target_image": str(bootstrap_right["path"]),
        "input.source_prompt": bootstrap_left["prompt"],
        "input.target_prompt": bootstrap_right["prompt"],
        "input.bridge_prompt": None,
        "input.bridge_prompts": bootstrap_schedule,
        "input.width": IMAGE_WIDTH,
        "input.height": IMAGE_HEIGHT,
        "flowmorph.scheduler_points": CHIMERA_DENOISING_STEPS,
        "flowmorph.start_timestep_index": 0,
        "flowmorph.optimization_steps_source": 1,
        "flowmorph.optimization_steps_target": 1,
        "flowmorph.frame_count": len(bootstrap_schedule),
        "flowmorph.render_indices": [0, CHIMERA_DENOISING_STEPS // 2, CHIMERA_DENOISING_STEPS - 1],
        "flowmorph.render_conditioning_mode": "prompt_schedule",
        "guidance.scale": CHIMERA_GUIDANCE_SCALE,
        "reproducibility.seed": BASE_SEED,
        "paths.input_root": str(RUN_DIRECTORY),
        "paths.work_root": str(Path(LOCAL_ASSET_ROOT) / PROJECT_NAME / "chimera_work"),
        "paths.result_root": str(CHIMERA_ROOT),
        "paths.hf_cache": HF_CACHE_DIR,
        "paths.drive_root": None,
        "output.save_contact_sheet": False,
        "output.save_webp": False,
        "output.save_gif": False,
        "output.save_mp4": False,
        "output.create_zip": True,
    }
    session_template = load_config(CONFIG_PATH, overrides=session_overrides)
    session_profile = select_hardware_profile(PROFILE if PROFILE != "auto" else session_template.model.profile)
    session_config = resolve_config(session_template, selected_profile=session_profile, check_input_files=True)
    session_resume = RESUME_CHIMERA_SEQUENCE and (CHIMERA_SESSION_DIRECTORY / "run_manifest.json").is_file()
    CHIMERA_RUNNER = FlowMorphRunner.from_config(session_config, run_directory=CHIMERA_SESSION_DIRECTORY)
    CHIMERA_RUNNER.prepare(resume=session_resume)
    CHIMERA_CONFIG = ChimeraConfig(
        inversion_steps=CHIMERA_INVERSION_STEPS,
        denoising_steps=CHIMERA_DENOISING_STEPS,
        aci_weight=CHIMERA_ACI_WEIGHT,
        sap_active_ratio=CHIMERA_SAP_ACTIVE_RATIO,
        anchor_max_tokens=CHIMERA_ANCHOR_MAX_TOKENS,
        anchor_reliability_threshold=CHIMERA_ANCHOR_RELIABILITY_THRESHOLD,
        ltm_mode=CHIMERA_LTM_MODE,
        ltm_bands=CHIMERA_LTM_BANDS,
        ltm_channel_chunk_size=CHIMERA_LTM_CHANNEL_CHUNK_SIZE,
        cache_stride=CHIMERA_CACHE_STRIDE,
        cache_storage=CHIMERA_CACHE_STORAGE,
        render_batch_size=CHIMERA_RENDER_BATCH_SIZE,
        decode_batch_size=CHIMERA_DECODE_BATCH_SIZE,
        guidance_scale=CHIMERA_GUIDANCE_SCALE,
        lora_scale=CHIMERA_LORA_SCALE,
        cfg_execution=CHIMERA_CFG_EXECUTION,
        oom_backoff=CHIMERA_BATCH_OOM_BACKOFF,
    )
    CHIMERA_SESSION = ChimeraFlux2Session(CHIMERA_RUNNER, config=CHIMERA_CONFIG)
    IMAGE_ASSET_CACHE, PROMPT_CONDITIONING_CACHE = CHIMERA_SESSION.seed_prepared_assets(
        bootstrap_left["uid"], bootstrap_right["uid"]
    )

    # The paper's LTM prototypes are dataset-level, not pair-specific.  We
    # approximate that contract with an evenly spaced subset of this run's
    # independently generated anchors, persist it on Drive, and reuse it for
    # every pair.  Calibration retains only 16-value spectra, never full caches.
    CHIMERA_LTM_CALIBRATION_PATH = (
        RUN_DIRECTORY / "metadata" / "chimera_ltm_calibration.json"
    )
    CHIMERA_LTM_CALIBRATION = None
    if CHIMERA_LTM_MODE == "fft":
        calibration_indices = [
            (index * len(BASE_RECORDS)) // CHIMERA_LTM_CALIBRATION_ANCHORS
            for index in range(CHIMERA_LTM_CALIBRATION_ANCHORS)
        ]
        calibration_records = [BASE_RECORDS[index] for index in calibration_indices]
        calibration_images = {
            record["uid"]: (
                record["path"],
                CHIMERA_ASSET_ROOT / f"{record['uid']}.png",
            )
            for record in calibration_records
            if record["uid"] not in IMAGE_ASSET_CACHE
        }
        calibration_prompts = [
            record["prompt"]
            for record in calibration_records
            if record["prompt"] not in PROMPT_CONDITIONING_CACHE
        ]
        if calibration_images or calibration_prompts:
            encoded_prompts, encoded_images = CHIMERA_SESSION.encode_missing_assets(
                prompts=list(dict.fromkeys(calibration_prompts)),
                images=calibration_images,
            )
            PROMPT_CONDITIONING_CACHE.update(encoded_prompts)
            IMAGE_ASSET_CACHE.update(encoded_images)

        calibration_contract = {
            "method": "CHIMERA FFT LTM radial magnitude prototypes v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "lora_sha256": file_sha256(LOCAL_LORA_PATH),
            "lora_scale": CHIMERA_LORA_SCALE,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "steps": CHIMERA_DENOISING_STEPS,
            "bands": CHIMERA_LTM_BANDS,
            "descriptor_normalized": False,
            "feature_groups": [
                group.label
                for group in select_flux_feature_groups(
                    CHIMERA_RUNNER.pipeline.transformer
                )
            ],
            "anchors": [
                {
                    "uid": record["uid"],
                    "image_sha256": file_sha256(record["path"]),
                    "prompt": record["prompt"],
                }
                for record in calibration_records
            ],
        }
        calibration_contract_fingerprint = stable_fingerprint(calibration_contract)
        calibration_reused = False
        if REUSE_CHIMERA_LTM_CALIBRATION and CHIMERA_LTM_CALIBRATION_PATH.is_file():
            saved_calibration = json.loads(
                CHIMERA_LTM_CALIBRATION_PATH.read_text(encoding="utf-8")
            )
            if saved_calibration.get("contract_fingerprint") == calibration_contract_fingerprint:
                candidate = LTMCalibration.from_dict(saved_calibration["calibration"])
                if saved_calibration.get("calibration_fingerprint") != candidate.fingerprint:
                    raise RuntimeError("Saved LTM calibration fingerprint is corrupt")
                CHIMERA_SESSION.set_ltm_calibration(candidate)
                CHIMERA_LTM_CALIBRATION = candidate
                calibration_reused = True

        if CHIMERA_LTM_CALIBRATION is None:
            print({
                "ltm_calibration": "running",
                "anchor_uids": [record["uid"] for record in calibration_records],
                "inversions": len(calibration_records),
                "bands": CHIMERA_LTM_BANDS,
            })
            CHIMERA_LTM_CALIBRATION = CHIMERA_SESSION.calibrate_ltm(tuple(
                (
                    IMAGE_ASSET_CACHE[record["uid"]],
                    PROMPT_CONDITIONING_CACHE[record["prompt"]],
                )
                for record in calibration_records
            ))
            CHIMERA_LTM_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHIMERA_LTM_CALIBRATION_PATH.write_text(json.dumps({
                "contract_fingerprint": calibration_contract_fingerprint,
                "calibration_fingerprint": CHIMERA_LTM_CALIBRATION.fingerprint,
                "contract": calibration_contract,
                "calibration": CHIMERA_LTM_CALIBRATION.to_dict(),
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        CHIMERA_LTM_FINGERPRINT = stable_fingerprint({
            "contract": calibration_contract_fingerprint,
            "calibration": CHIMERA_LTM_CALIBRATION.fingerprint,
        })
        CHIMERA_LTM_REPORT = {
            "mode": "fft",
            "reused": calibration_reused,
            "path": str(CHIMERA_LTM_CALIBRATION_PATH),
            "sample_count": CHIMERA_LTM_CALIBRATION.sample_count,
            "mapping": list(CHIMERA_LTM_CALIBRATION.mapping),
            "fingerprint": CHIMERA_LTM_FINGERPRINT,
        }
    else:
        CHIMERA_LTM_FINGERPRINT = stable_fingerprint({
            "mode": "linear",
            "steps": CHIMERA_DENOISING_STEPS,
        })
        CHIMERA_LTM_REPORT = {
            "mode": "linear",
            "mapping": "fixed early/middle/late thirds",
            "fingerprint": CHIMERA_LTM_FINGERPRINT,
        }

    print({
        "model_loads": 1,
        "backward_probes": 0,
        "training_or_endpoint_optimization": False,
        "feature_groups": [group.label for group in select_flux_feature_groups(CHIMERA_RUNNER.pipeline.transformer)],
        "cache_storage": CHIMERA_CACHE_STORAGE,
        "cache_stride": CHIMERA_CACHE_STRIDE,
        "ltm": CHIMERA_LTM_REPORT,
    })

    def encode_prompt_values(prompts):
        missing = [prompt for prompt in dict.fromkeys(prompts) if prompt not in PROMPT_CONDITIONING_CACHE]
        if missing:
            encoded, _ = CHIMERA_SESSION.encode_missing_assets(prompts=missing, images={})
            PROMPT_CONDITIONING_CACHE.update(encoded)

    def ensure_reliable_triplet(job):
        encode_prompt_values(proposal_prompts(job))
        while True:
            proposal = job["proposal"]
            similarity_a, similarity_b, reliability = prompt_anchor_reliability(
                PROMPT_CONDITIONING_CACHE[proposal.anchor_prompt],
                PROMPT_CONDITIONING_CACHE[proposal.prompt_a],
                PROMPT_CONDITIONING_CACHE[proposal.prompt_b],
            )
            job["reliability"] = {
                "anchor_to_a": similarity_a,
                "anchor_to_b": similarity_b,
                "minimum": reliability,
            }
            if reliability >= CHIMERA_ANCHOR_RELIABILITY_THRESHOLD:
                save_triplet_job(job)
                return job
            if job["requeries"] >= CHIMERA_SAP_MAX_REQUERIES:
                raise RuntimeError(
                    f"SAP reliability stayed at {reliability:.4f} below "
                    f"{CHIMERA_ANCHOR_RELIABILITY_THRESHOLD:.4f} for "
                    f"{job['left']['uid']} -> {job['right']['uid']}"
                )
            feedback = (
                f"anchor-to-A cosine={similarity_a:.4f}, anchor-to-B cosine={similarity_b:.4f}, "
                f"required minimum={CHIMERA_ANCHOR_RELIABILITY_THRESHOLD:.4f}."
            )
            proposal, response = propose_chimera_triplet(job["left"], job["right"], feedback)
            job["proposal"] = proposal
            job["response_id"] = response.id
            job["usage"] = usage_payload(response)
            job["requeries"] += 1
            encode_prompt_values(proposal_prompts(job))

    def render_chimera_job(job, alphas, output_paths):
        proposal = job["proposal"]
        source_cache, target_cache = CHIMERA_SESSION.invert_pair(
            pair_key=f"r{job['round']:02d}_g{job['gap_index']:04d}",
            source_asset=IMAGE_ASSET_CACHE[job["left"]["uid"]],
            target_asset=IMAGE_ASSET_CACHE[job["right"]["uid"]],
            source_conditioning=PROMPT_CONDITIONING_CACHE[proposal.prompt_a],
            target_conditioning=PROMPT_CONDITIONING_CACHE[proposal.prompt_b],
        )
        frames = CHIMERA_SESSION.render_pair(
            source_cache=source_cache,
            target_cache=target_cache,
            source_conditioning=PROMPT_CONDITIONING_CACHE[proposal.prompt_a],
            target_conditioning=PROMPT_CONDITIONING_CACHE[proposal.prompt_b],
            anchor_conditioning=PROMPT_CONDITIONING_CACHE[proposal.anchor_prompt],
            alphas=alphas,
        )
        CHIMERA_SESSION.decode_frames_to_paths(frames, output_paths)
        cache_report = {
            "source_cache_mib": source_cache.storage_bytes / (1024 ** 2),
            "target_cache_mib": target_cache.storage_bytes / (1024 ** 2),
            "feature_groups": dict(source_cache.group_modules),
            "ltm_mode": CHIMERA_LTM_MODE,
            "ltm_mapping": (
                list(CHIMERA_LTM_CALIBRATION.mapping)
                if CHIMERA_LTM_CALIBRATION is not None
                else "fixed early/middle/late thirds"
            ),
            "ltm_fingerprint": CHIMERA_LTM_FINGERPRINT,
        }
        del source_cache, target_cache, frames
        gc.collect()
        torch.cuda.empty_cache()
        return cache_report

    if RUN_CHIMERA_ONE_GAP_TEST:
        test_index = CHIMERA_ONE_GAP_TEST_INDEX
        test_left = BASE_RECORDS[test_index]
        test_right = BASE_RECORDS[(test_index + 1) % len(BASE_RECORDS)]
        test_directory = RUN_DIRECTORY / "quality_gates" / "chimera_one_gap"
        test_directory.mkdir(parents=True, exist_ok=True)
        test_job = load_or_create_triplet(test_left, test_right, 0, test_index, test_directory / "sap_triplet.json")
        ensure_reliable_triplet(test_job)
        missing_images = {
            record["uid"]: (record["path"], CHIMERA_ASSET_ROOT / f"{record['uid']}.png")
            for record in (test_left, test_right)
            if record["uid"] not in IMAGE_ASSET_CACHE
        }
        if missing_images:
            _, images = CHIMERA_SESSION.encode_missing_assets(prompts=[], images=missing_images)
            IMAGE_ASSET_CACHE.update(images)
        test_paths = [test_directory / f"alpha_{alpha:.3f}.png" for alpha in CHIMERA_ONE_GAP_TEST_ALPHAS]
        test_cache_report = render_chimera_job(test_job, CHIMERA_ONE_GAP_TEST_ALPHAS, test_paths)
        test_sheet = test_directory / "quality_sheet.png"
        test_images = [Image.open(path).convert("RGB") for path in [test_left["path"], *test_paths, test_right["path"]]]
        make_contact_sheet(test_images, test_sheet, columns=len(test_images), labels=["A", *[f"a={a:.2f}" for a in CHIMERA_ONE_GAP_TEST_ALPHAS], "B"])
        for image in test_images:
            image.close()
        preview = Image.open(test_sheet).convert("RGB")
        preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
        display(Markdown("### One-gap CHIMERA quality gate"))
        display(preview)
        preview.close()
        print({"quality_sheet": str(test_sheet), "cache": test_cache_report, "full_run": "Run the next cell after accepting this preview."})
    else:
        print("One-gap CHIMERA quality test skipped.")
    '''
)

notebook["cells"][20]["source"] = lines(
    """
    ### Full flat CHIMERA run

    Run this only after accepting the one-gap sheet.  Each pair is independently
    resumable.  Endpoint feature caches are held only for the active pair and
    immediately released after its ten PNGs are decoded, bounding host memory
    across the complete fifteen-gap pass.
    """
)
notebook["cells"][21]["source"] = lines(
    r'''
    CURRENT_RECORDS = list(BASE_RECORDS)
    ROUND_MANIFESTS = []
    CHIMERA_COMPLETION_PATHS = []
    OPENAI_SAP_PROMPT_COUNT = 0
    CHIMERA_PAIR_RENDER_COUNT = 0

    for round_number, round_spec in enumerate(CHIMERA_ROUND_SPECS, start=1):
        midpoint_count = int(round_spec["midpoint_count"])
        fractions = [index / (midpoint_count + 1) for index in range(1, midpoint_count + 1)]
        round_directory = RUN_DIRECTORY / "rounds" / f"round_{round_number:02d}"
        image_directory = round_directory / "images"
        proposal_directory = round_directory / "sap_triplets"
        progress_directory = round_directory / "streaming_progress"
        for directory in (round_directory, image_directory, proposal_directory, progress_directory):
            directory.mkdir(parents=True, exist_ok=True)

        incoming = list(CURRENT_RECORDS)
        gap_count = len(incoming)
        prompt_jobs = []
        for gap_index, left in enumerate(incoming):
            right = incoming[(gap_index + 1) % gap_count]
            prompt_jobs.append((
                left,
                right,
                round_number,
                gap_index,
                proposal_directory / f"r{round_number:02d}_g{gap_index:04d}.json",
            ))
        with ThreadPoolExecutor(max_workers=min(OPENAI_CONCURRENCY, len(prompt_jobs))) as executor:
            pair_jobs = list(executor.map(lambda args: load_or_create_triplet(*args), prompt_jobs))
        OPENAI_SAP_PROMPT_COUNT += len(pair_jobs)

        missing_images = {
            record["uid"]: (record["path"], CHIMERA_ASSET_ROOT / f"{record['uid']}.png")
            for record in incoming
            if record["uid"] not in IMAGE_ASSET_CACHE
        }
        initial_prompts = [prompt for job in pair_jobs for prompt in proposal_prompts(job)]
        missing_prompts = [prompt for prompt in dict.fromkeys(initial_prompts) if prompt not in PROMPT_CONDITIONING_CACHE]
        if missing_images or missing_prompts:
            new_prompts, new_images = CHIMERA_SESSION.encode_missing_assets(
                prompts=missing_prompts,
                images=missing_images,
            )
            PROMPT_CONDITIONING_CACHE.update(new_prompts)
            IMAGE_ASSET_CACHE.update(new_images)
        for job in pair_jobs:
            ensure_reliable_triplet(job)

        for job in pair_jobs:
            gap_index = job["gap_index"]
            pair_uid = f"r{round_number:02d}_g{gap_index:04d}"
            frame_records = [
                {
                    "uid": f"{pair_uid}_m{index:02d}",
                    "alpha": alpha,
                    "output_path": image_directory / f"{pair_uid}_m{index:02d}.png",
                }
                for index, alpha in enumerate(fractions, start=1)
            ]
            proposal = job["proposal"]
            pair_contract = {
                "method": "CHIMERA FLUX.2 port: Euler inversion + FFT LTM + IDM + ACI + SAP",
                "round": round_number,
                "gap_index": gap_index,
                "left_uid": job["left"]["uid"],
                "left_path": str(job["left"]["path"]),
                "left_image_sha256": file_sha256(job["left"]["path"]),
                "right_uid": job["right"]["uid"],
                "right_path": str(job["right"]["path"]),
                "right_image_sha256": file_sha256(job["right"]["path"]),
                "sap_triplet": proposal.model_dump(mode="json"),
                "sap_reliability": job["reliability"],
                "alphas": fractions,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "lora_sha256": file_sha256(LOCAL_LORA_PATH),
                "chimera_config": {
                    "steps": CHIMERA_DENOISING_STEPS,
                    "aci_weight": CHIMERA_ACI_WEIGHT,
                    "sap_active_ratio": CHIMERA_SAP_ACTIVE_RATIO,
                    "ltm_mode": CHIMERA_LTM_MODE,
                    "ltm_bands": CHIMERA_LTM_BANDS,
                    "ltm_fingerprint": CHIMERA_LTM_FINGERPRINT,
                    "cache_stride": CHIMERA_CACHE_STRIDE,
                    "cache_storage": CHIMERA_CACHE_STORAGE,
                    "guidance_scale": CHIMERA_GUIDANCE_SCALE,
                    "lora_scale": CHIMERA_LORA_SCALE,
                },
            }
            pair_fingerprint = stable_fingerprint(pair_contract)
            completion_path = image_directory / f"{pair_uid}.chimera.json"
            completed = False
            completion = None
            if completion_path.is_file() and all(record["output_path"].is_file() for record in frame_records):
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                completed = completion.get("pair_fingerprint") == pair_fingerprint
            job.update({
                "pair_uid": pair_uid,
                "frame_records": frame_records,
                "pair_contract": pair_contract,
                "pair_fingerprint": pair_fingerprint,
                "completion_path": completion_path,
                "completion": completion,
                "completed": completed,
            })

        pending = [job for job in pair_jobs if not job["completed"]]
        for chunk_start in range(0, len(pending), CHIMERA_STREAM_PAIRS_PER_CHUNK):
            chunk = pending[chunk_start : chunk_start + CHIMERA_STREAM_PAIRS_PER_CHUNK]
            for job in chunk:
                CHIMERA_PAIR_RENDER_COUNT += 1
                output_paths = [record["output_path"] for record in job["frame_records"]]
                cache_report = render_chimera_job(job, fractions, output_paths)
                completion = {
                    "status": "complete",
                    "pair_uid": job["pair_uid"],
                    "pair_fingerprint": job["pair_fingerprint"],
                    "pair_contract": job["pair_contract"],
                    "cache_report": cache_report,
                    "render_batch_size": CHIMERA_SESSION.last_render_batch_size,
                    "decode_batch_size": CHIMERA_SESSION.last_decode_batch_size,
                    "inserted": [
                        {"alpha": record["alpha"], "image": str(record["output_path"])}
                        for record in job["frame_records"]
                    ],
                }
                job["completion_path"].write_text(
                    json.dumps(completion, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                job["completion"] = completion
                job["completed"] = True
                print({
                    "pair": job["pair_uid"],
                    "interiors": len(output_paths),
                    "cache_mib": round(cache_report["source_cache_mib"] + cache_report["target_cache_mib"], 1),
                    "latest_png": str(output_paths[-1]),
                })
            if CHIMERA_STREAM_DISPLAY_PROGRESS and chunk:
                representative_paths = [
                    job["frame_records"][len(job["frame_records"]) // 2]["output_path"]
                    for job in chunk
                ]
                images = [Image.open(path).convert("RGB") for path in representative_paths]
                for image in images:
                    image.thumbnail((256, 256))
                chunk_number = chunk_start // CHIMERA_STREAM_PAIRS_PER_CHUNK + 1
                sheet_path = progress_directory / f"chunk_{chunk_number:03d}.png"
                make_contact_sheet(images, sheet_path, columns=len(images), labels=[job["pair_uid"] for job in chunk])
                for image in images:
                    image.close()
                preview = Image.open(sheet_path).convert("RGB")
                preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
                display(Markdown(f"### CHIMERA round {round_number}: {min(chunk_start + len(chunk), len(pending))}/{len(pending)} new gaps"))
                display(preview)
                preview.close()

        outgoing = []
        for job in pair_jobs:
            left = job["left"]
            outgoing.append(dict(left))
            for record in job["frame_records"]:
                outgoing.append({
                    "uid": record["uid"],
                    "kind": "chimera_interior",
                    "round": round_number,
                    "alpha": record["alpha"],
                    "left_uid": left["uid"],
                    "right_uid": job["right"]["uid"],
                    "science": job["proposal"].science_connection,
                    "visual_correspondence": job["proposal"].visual_correspondence,
                    "prompt": job["proposal"].anchor_prompt,
                    "sap_prompt_a": job["proposal"].prompt_a,
                    "sap_prompt_b": job["proposal"].prompt_b,
                    "sap_reliability": job["reliability"],
                    "path": str(record["output_path"]),
                    "proposal_path": str(job["proposal_path"]),
                    "chimera_completion_path": str(job["completion_path"]),
                    "chimera_fingerprint": job["pair_fingerprint"],
                })
            CHIMERA_COMPLETION_PATHS.append(str(job["completion_path"]))

        CURRENT_RECORDS = outgoing
        round_manifest_path = round_directory / "sequence_manifest.json"
        round_manifest_path.write_text(json.dumps({
            "round": round_number,
            "cyclic": True,
            "interpolation_method": "CHIMERA FLUX.2 port with Euler inversion, FFT LTM, ACI, IDM, and SAP",
            "input_count": len(incoming),
            "midpoints_per_gap": midpoint_count,
            "alphas": fractions,
            "output_count": len(outgoing),
            "records": outgoing,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ROUND_MANIFESTS.append(str(round_manifest_path))

        round_contact_sheet = round_directory / "contact_sheet.png"
        round_images = []
        for item in outgoing:
            image = Image.open(item["path"]).convert("RGB")
            image.thumbnail((160, 160))
            round_images.append(image)
        make_contact_sheet(
            round_images,
            round_contact_sheet,
            columns=min(CONTACT_SHEET_COLUMNS, len(round_images)),
            labels=[item["uid"] for item in outgoing],
        )
        for image in round_images:
            image.close()
        preview = Image.open(round_contact_sheet).convert("RGB")
        preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
        display(Markdown(f"### CHIMERA round {round_number}: {len(outgoing)} cyclic images"))
        display(preview)
        preview.close()

    FINAL_RECORDS = [dict(record) for record in CURRENT_RECORDS]
    FINAL_SEQUENCE_MANIFEST = RUN_DIRECTORY / "metadata" / "final_recursive_chimera_sequence.json"
    FINAL_SEQUENCE_MANIFEST.write_text(json.dumps({
        "project": PROJECT_NAME,
        "cyclic": True,
        "interpolation_method": "CHIMERA port for FLUX.2 Klein Base 9B + LoRA",
        "paper": "https://arxiv.org/abs/2512.07155",
        "anchor_count": len(BASE_RECORDS),
        "round_specs": CHIMERA_ROUND_SPECS,
        "model_loads": 1,
        "backward_probes": 0,
        "training_or_endpoint_optimization": False,
        "pair_renders": CHIMERA_PAIR_RENDER_COUNT,
        "openai_sap_triplets": OPENAI_SAP_PROMPT_COUNT,
        "completion_manifests": CHIMERA_COMPLETION_PATHS,
        "round_manifests": ROUND_MANIFESTS,
        "final_count": len(FINAL_RECORDS),
        "records": FINAL_RECORDS,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print({"final_images": len(FINAL_RECORDS), "manifest": str(FINAL_SEQUENCE_MANIFEST)})

    for component_name in ("transformer", "vae", "text_encoder"):
        component = getattr(CHIMERA_RUNNER.pipeline, component_name, None)
        if component is not None and callable(getattr(component, "to", None)):
            component.to("cpu")
    del IMAGE_ASSET_CACHE, PROMPT_CONDITIONING_CACHE, CHIMERA_SESSION, CHIMERA_RUNNER
    gc.collect()
    torch.cuda.empty_cache()
    print("Released the retained CHIMERA FLUX model; optional GLCS/RIFE can use the GPU.")
    '''
)

notebook["cells"][22]["source"] = lines(
    """
    ## 10. Optional DINO-backed GLCS, then assemble and audit the cyclic sequence

    GLCS is evaluated per completed CHIMERA pair.  It is disabled by default;
    enabling it downloads DINOv2 after the 9B model has been released.  The
    paper reports DINO as the strongest tested alternative similarity function.
    The remaining cell retains the source notebook's tone, seam, and preview
    workflow without altering raw CHIMERA PNGs.
    """
)
notebook["cells"][23]["source"] = lines(
    r'''
    FINAL_SEQUENCE_MANIFEST = RUN_DIRECTORY / "metadata" / "final_recursive_chimera_sequence.json"
    print(FINAL_SEQUENCE_MANIFEST, FINAL_SEQUENCE_MANIFEST.exists())

    if RUN_CHIMERA_DINO_GLCS:
        import numpy as np
        from transformers import AutoImageProcessor, AutoModel
        from flowmorph_klein.chimera import compute_glcs_from_similarities

        processor = AutoImageProcessor.from_pretrained(CHIMERA_DINO_MODEL_ID, cache_dir=HF_CACHE_DIR)
        dino = AutoModel.from_pretrained(CHIMERA_DINO_MODEL_ID, cache_dir=HF_CACHE_DIR).to("cuda").eval()

        def dino_embeddings(paths):
            images = []
            for path in paths:
                with Image.open(path) as opened:
                    images.append(opened.convert("RGB").copy())
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                vectors = dino(**inputs).last_hidden_state[:, 0].float()
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            for image in images:
                image.close()
            return vectors.cpu()

        glcs_rows = []
        for completion_name in CHIMERA_COMPLETION_PATHS:
            completion = json.loads(Path(completion_name).read_text(encoding="utf-8"))
            contract = completion["pair_contract"]
            interior_paths = [item["image"] for item in completion["inserted"]]
            vectors = dino_embeddings([contract["left_path"], *interior_paths, contract["right_path"]])
            left, interiors, right = vectors[0], vectors[1:-1], vectors[-1]
            similarity_a = (interiors @ left).tolist()
            similarity_b = (interiors @ right).tolist()
            matrix = [
                [float(left @ left), float(left @ right)],
                [float(right @ left), float(right @ right)],
            ]
            scores = compute_glcs_from_similarities(
                similarity_a,
                similarity_b,
                endpoint_similarity_matrix=matrix,
                gamma=CHIMERA_GLCS_GAMMA,
            )
            glcs_rows.append({"pair_uid": completion["pair_uid"], **scores})
        glcs_report = RUN_DIRECTORY / "diagnostics" / "chimera_dino_glcs.json"
        glcs_report.parent.mkdir(parents=True, exist_ok=True)
        glcs_report.write_text(json.dumps({
            "similarity_model": CHIMERA_DINO_MODEL_ID,
            "gamma": CHIMERA_GLCS_GAMMA,
            "pairs": glcs_rows,
            "mean_glcs": float(np.mean([row["glcs"] for row in glcs_rows])),
        }, indent=2) + "\n", encoding="utf-8")
        dino.to("cpu")
        del dino, processor
        torch.cuda.empty_cache()
        print({"dino_glcs_pairs": len(glcs_rows), "report": str(glcs_report)})
    else:
        print("DINO-backed GLCS disabled.")
    '''
)

# Keep the proven finishing stack, changing only workflow-specific names and
# manifest/diagnostic variables.  Package imports remain flowmorph_klein.
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") == "markdown" and index >= 22:
        cell["source"] = [line.replace("FlowMorph", "CHIMERA") for line in cell["source"]]
    if cell.get("cell_type") != "code" or index < 24:
        continue
    text = source(cell)
    text = text.replace("final_recursive_flowmorph_sequence", "final_recursive_chimera_sequence")
    text = text.replace("raw_flowmorph_path", "raw_chimera_path")
    text = text.replace("FLOWMORPH_ROUND_SPECS", "CHIMERA_ROUND_SPECS")
    text = text.replace("FLOWMORPH_RENDER_BATCH_SIZE", "CHIMERA_RENDER_BATCH_SIZE")
    text = text.replace("recursive_flowmorph_prompt_only", "recursive_chimera_prompt_only")
    text = text.replace("FlowMorph", "CHIMERA")
    cell["source"] = text.splitlines(keepends=True)

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"prompt-only-chimera-{index:02d}"
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
