"""Build the isolated periodic B-spline FlowMorph art notebook.

The notebook reuses the proven prompt-only anchor, endpoint fitting, diagnostics,
and RIFE cells, but replaces pairwise recursive interpolation with one global
periodic cubic B-spline through every fitted endpoint.  It never edits the
existing pairwise notebooks or their implementation path.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = (
    ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Prompt_Only.ipynb"
)
OUTPUT = Path(
    os.environ.get(
        "FLOWMORPH_PERIODIC_BSPLINE_NOTEBOOK_OUTPUT",
        ROOT / "notebooks" / "StillLife_Periodic_BSpline_FlowMorph.ipynb",
    )
)
if (
    "FLOWMORPH_PERIODIC_BSPLINE_NOTEBOOK_OUTPUT" not in os.environ
    and OUTPUT.exists()
    and os.environ.get("FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE") != "1"
):
    raise RuntimeError(
        "Refusing to overwrite the tracked periodic B-spline notebook. Set "
        "FLOWMORPH_PERIODIC_BSPLINE_NOTEBOOK_OUTPUT or explicitly set "
        "FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE=1."
    )


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def literal_assignment(notebook: dict, name: str):
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse(source(cell))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                try:
                    return ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    return None
    return None


if not SOURCE_NOTEBOOK.is_file():
    raise FileNotFoundError(
        "Build the prompt-only notebook first: "
        f"{SOURCE_NOTEBOOK}"
    )
prompt_notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

preserved_stage_source = source(prompt_notebook["cells"][4])
preserved_settings = {
    name: literal_assignment(prompt_notebook, name)
    for name in ("DRIVE_PROJECT_BASE", "RESUME_RUN_DIRECTORY")
}


settings_source = f'''
PROJECT_ROOT = "/content/FlowMorphKlein9B"
REPOSITORY_URL = "https://github.com/MNoichl/FluxFlowMorph.git"
UPDATE_REPOSITORY = True
PROJECT_NAME = "science_path_periodic_bspline_flowmorph"
CONFIG_PATH = f"{{PROJECT_ROOT}}/configs/full_9b_lora.yaml"
PROFILE = "auto"
LOCAL_ASSET_ROOT = "/content/flowmorph_periodic_art"
HF_CACHE_DIR = "/content/hf_cache"

# Drive persistence. Every image, manifest, diagnostic, and video is written
# immediately into the numbered run directory.
MOUNT_DRIVE = True
DRIVE_PROJECT_BASE = {preserved_settings["DRIVE_PROJECT_BASE"]!r}
RESUME_RUN_DIRECTORY = {preserved_settings["RESUME_RUN_DIRECTORY"]!r}

# Editable anchor selection. None uses the complete BASE_STAGES list.
BASE_PROMPT_COUNT = None
REGENERATE_BASE_FRAMES = True
RESUME_FLOWMORPH_SEQUENCE = True

# Global periodic spline timing. Every segment includes its left anchor and
# excludes its right anchor, so the opening image is never duplicated.
SPLINE_FRAMES_PER_ANCHOR = 10
SPLINE_MIN_FRAMES_PER_SEGMENT = 3
SPLINE_TIMING_DISTANCE_STRENGTH = 0.45
SPLINE_TIMING_DISTANCE_EXPONENT = 0.50
SPLINE_TIMING_MAX_SEGMENT_RATIO = 1.75
SPLINE_DISTANCE_ANALYSIS_SIZE = 128
SPLINE_DISTANCE_COLOR_WEIGHT = 0.75
RUN_SPLINE_COARSE_PREVIEW = True
SPLINE_PREVIEW_FRAMES_PER_ANCHOR = 2
SPLINE_STREAM_CHUNK_SIZE = 24
SPLINE_REUSE_RENDERED_FRAMES = True

# Sequence-native true FlowMorph fitting/rendering.
FLOWMORPH_FIT_LORA_SCALE = 1.2
FLOWMORPH_RENDER_LORA_SCALE = 1.2
FLOWMORPH_GUIDANCE_SCALE = 7.0
FLOWMORPH_SCHEDULER_POINTS = 100
FLOWMORPH_START_TIMESTEP_INDEX = 35
FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 50
FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 50
FLOWMORPH_PRED_LEARNING_RATE = 0.04
FLOWMORPH_U_LEARNING_RATE = 0.01
FLOWMORPH_RENDER_INDICES = [*range(35, 100, 5), 99]
FLOWMORPH_CHECKPOINT_EVERY = 25
FLOWMORPH_ENDPOINT_BATCH_SIZE = 2
FLOWMORPH_RENDER_BATCH_SIZE = 4
FLOWMORPH_DECODE_BATCH_SIZE = 8
FLOWMORPH_CFG_EXECUTION = "batched"
FLOWMORPH_BATCH_OOM_BACKOFF = True

# FLUX.2 Klein Base 9B + RIJKSOIL LoRA.
MODEL_ID = "Runware/BFL-FLUX.2-klein-base-9B"
MODEL_REVISION = "52d7274119d8a2b67f4fba1a43694d9169a44851"
LORA_SOURCE = "MaxNoichl/RIJKSOIL_FLUX2_KLEIN9B_lora_01_000001650"
LORA_REVISION = "042a31d6cd09bf55195f820461fac60b1a358409"
LORA_WEIGHT_NAME = "RIJKSOIL_FLUX2_KLEIN9B_lora_01_000001650.safetensors"
LORA_ADAPTER_NAME = "rijks_oil"
LORA_TRIGGER = "RIJKSOIL"

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
IMAGE_INFERENCE_STEPS = 50
IMAGE_GUIDANCE_SCALE = 7.0
IMAGE_LORA_SCALE = 1.2
BASE_SEED = 42

# Weak prompt-anchor continuity: blurred/grained previous image only. There is
# no beige or other flat canvas in this prompt-only workflow.
BASE_CONTINUITY_ENABLED = True
BASE_REFERENCE_BLUR = 16.0
BASE_REFERENCE_GRAIN_STRENGTH = 0.035
BASE_REFERENCE_DENOISE_STRENGTH = 0.75
SAVE_SOFT_REFERENCES = True
FLUX_PROMPT_MAX_SEQUENCE_LENGTH = 512

# Optional post-render tonal correction; raw spline PNGs are never overwritten.
TEMPORAL_TONE_STABILIZATION_ENABLED = False
TEMPORAL_TONE_WINDOW_RADIUS = 2
TEMPORAL_TONE_STRENGTH = 0.70
TEMPORAL_TONE_MEAN_THRESHOLD = 0.02
TEMPORAL_TONE_CONTRAST_THRESHOLD = 0.10
TEMPORAL_TONE_MAD_MULTIPLIER = 3.5
TEMPORAL_TONE_MAX_MEAN_SHIFT = 0.06
TEMPORAL_TONE_MAX_CONTRAST_SCALE_DELTA = 0.15
TEMPORAL_TONE_ANALYSIS_MAX_SIDE = 256
TEMPORAL_TONE_REUSE_EXISTING = True

# Read-only diagnosis of raw circular spline output.
RUN_FLICKER_DIAGNOSTIC = True
FLICKER_ANALYSIS_MAX_SIDE = 256
FLICKER_OUTLIER_MAD_MULTIPLIER = 3.5
FLICKER_MINIMUM_OUTLIER_SCORE = 3.0
FLICKER_MAX_LAG = 64

# Trial and notebook display.
RUN_TRIAL_KEYFRAME = True
TRIAL_KEYFRAME_INDEX = None
TRIAL_SEED = None
TRIAL_DISPLAY_MAX_WIDTH = 768
CONTACT_SHEET_COLUMNS = 8
CONTACT_SHEET_DISPLAY_MAX_WIDTH = 1100
LOOP_PREVIEW_DISPLAY_WIDTH = 768
LOOP_PREVIEW_RENDER_MAX_SIDE = 512

# Circular RIFE/SSIM finishing.
VIDEO_SLOWDOWN_FACTOR = 3.0
SOURCE_SEQUENCE_FPS = 12.0 / VIDEO_SLOWDOWN_FACTOR
LOOP_AUTO_ROTATE_TO_QUIETEST_CUT = True
LOOP_SEAM_ANALYSIS_SIZE = 192
RUN_RIFE_POSTPROCESS = True
RIFE_REPOSITORY_URL = "https://github.com/hzwer/Practical-RIFE.git"
RIFE_REPOSITORY_REVISION = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
RIFE_ROOT = "/content/Practical-RIFE"
RIFE_MODEL_REPOSITORY = "Bash2X/RIFE-Models"
RIFE_MODEL_REVISION = "feaf6d11238b4a1e9f015a5d18c18df152affd20"
RIFE_MODEL_FILENAME = "RIFE_v4.25.zip"
RIFE_MULTIPLIER = int(round(2 * VIDEO_SLOWDOWN_FACTOR))
RIFE_SCALE = 1.0
RIFE_BATCH_SIZE = 4
RIFE_USE_FP16 = True
RIFE_RETRY_WITH_FP32 = True
RIFE_FINAL_FPS = 24.0
RIFE_SSIM_ANALYSIS_SIZE = 192
RIFE_SSIM_WEIGHT_FLOOR = 1e-6
RIFE_VIDEO_CRF = 16
RIFE_KEEP_WORK_FRAMES = False
RIFE_DISPLAY_WIDTH = 768
DOWNLOAD_FINAL_VIDEO = False
'''


install_source = source(prompt_notebook["cells"][6])
openai_start = install_source.index("\ntry:\n    import openai")
openai_end = install_source.index("\nimport importlib", openai_start)
install_source = install_source[:openai_start] + install_source[openai_end:]
install_source = install_source.replace(
    '    "openai_sdk": openai.__version__,\n',
    "",
)


run_directory_source = '''
import json
import re
from datetime import datetime, timezone

if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", PROJECT_NAME):
    raise ValueError("PROJECT_NAME may contain only letters, numbers, underscores, and hyphens")

DRIVE_ENABLED = False
if MOUNT_DRIVE:
    try:
        from google.colab import drive
    except ImportError as error:
        raise RuntimeError("Drive mounting requires a Google Colab kernel.") from error
    drive.mount("/content/drive")
    drive_base = Path(DRIVE_PROJECT_BASE)
    drive_base.mkdir(parents=True, exist_ok=True)
    DRIVE_ENABLED = True
else:
    drive_base = None

def reserve_numbered_run(parent, project_name):
    project_root = Path(parent) / project_name
    project_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    prefix = f"{project_name}_"
    for candidate in project_root.iterdir():
        if candidate.is_dir() and candidate.name.startswith(prefix):
            token = candidate.name[len(prefix):].split("_", 1)[0]
            if token.isdigit():
                numbers.append(int(token))
    sequence = max(numbers, default=0) + 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        candidate = project_root / f"{project_name}_{sequence:04d}_{timestamp}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            sequence += 1
            continue
        return candidate

if RESUME_RUN_DIRECTORY is not None:
    RUN_DIRECTORY = Path(RESUME_RUN_DIRECTORY).expanduser()
    if not RUN_DIRECTORY.is_dir():
        raise FileNotFoundError(f"RESUME_RUN_DIRECTORY does not exist: {RUN_DIRECTORY}")
elif DRIVE_ENABLED:
    RUN_DIRECTORY = reserve_numbered_run(drive_base, PROJECT_NAME)
else:
    RUN_DIRECTORY = reserve_numbered_run(LOCAL_ASSET_ROOT, PROJECT_NAME)

for child in (
    "base_frames", "trials", "previews", "video", "metadata",
    "periodic_spline", "diagnostics",
):
    (RUN_DIRECTORY / child).mkdir(parents=True, exist_ok=True)
Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

run_identity = {
    "project": PROJECT_NAME,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "persistent": DRIVE_ENABLED,
    "run_directory": str(RUN_DIRECTORY),
    "interpolation": "periodic_cubic_bspline_through_fitted_flowmorph_endpoints",
    "terminal_duplicate": False,
}
(RUN_DIRECTORY / "metadata" / "run_identity.json").write_text(
    json.dumps(run_identity, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8",
)
print("Run directory:", RUN_DIRECTORY)
print("Every generated image and manifest is written here as soon as it exists.")
'''


validation_source = '''
if BASE_PROMPT_COUNT is None:
    BASE_PROMPT_COUNT = len(BASE_STAGES)
elif not 4 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):
    raise ValueError(f"BASE_PROMPT_COUNT must be between 4 and {len(BASE_STAGES)}")
if BASE_PROMPT_COUNT < 4:
    raise ValueError("A periodic cubic B-spline needs at least four anchors")
if not (256 <= IMAGE_WIDTH <= 2048 and IMAGE_WIDTH % 16 == 0):
    raise ValueError("IMAGE_WIDTH must be 256–2048 and divisible by 16")
if not (256 <= IMAGE_HEIGHT <= 2048 and IMAGE_HEIGHT % 16 == 0):
    raise ValueError("IMAGE_HEIGHT must be 256–2048 and divisible by 16")
if not 1 <= IMAGE_INFERENCE_STEPS <= 100:
    raise ValueError("IMAGE_INFERENCE_STEPS must be between 1 and 100")
if not 0 <= IMAGE_GUIDANCE_SCALE <= 20:
    raise ValueError("IMAGE_GUIDANCE_SCALE must be between 0 and 20")
if not 0 < IMAGE_LORA_SCALE <= 4:
    raise ValueError("IMAGE_LORA_SCALE must lie in (0, 4]")
if not 0 <= BASE_REFERENCE_GRAIN_STRENGTH <= 0.25:
    raise ValueError("BASE_REFERENCE_GRAIN_STRENGTH must lie in [0, 0.25]")
if not 0 < BASE_REFERENCE_DENOISE_STRENGTH <= 1:
    raise ValueError("BASE_REFERENCE_DENOISE_STRENGTH must lie in (0, 1]")
if not 32 <= FLUX_PROMPT_MAX_SEQUENCE_LENGTH <= 512:
    raise ValueError("FLUX_PROMPT_MAX_SEQUENCE_LENGTH must lie in [32, 512]")
if FLOWMORPH_START_TIMESTEP_INDEX != FLOWMORPH_RENDER_INDICES[0]:
    raise ValueError("The first render index must equal the FlowMorph start index")
if FLOWMORPH_RENDER_INDICES != sorted(set(FLOWMORPH_RENDER_INDICES)):
    raise ValueError("FLOWMORPH_RENDER_INDICES must be strictly increasing")
if FLOWMORPH_RENDER_INDICES[-1] >= FLOWMORPH_SCHEDULER_POINTS:
    raise ValueError("FLOWMORPH_RENDER_INDICES must be smaller than scheduler points")
if FLOWMORPH_SOURCE_OPTIMIZATION_STEPS != FLOWMORPH_TARGET_OPTIMIZATION_STEPS:
    raise ValueError("Cached endpoints require one shared optimization-step count")
if FLOWMORPH_SOURCE_OPTIMIZATION_STEPS < 1:
    raise ValueError("FlowMorph optimization steps must be positive")
if FLOWMORPH_FIT_LORA_SCALE != IMAGE_LORA_SCALE:
    raise ValueError("FlowMorph fit LoRA scale must match IMAGE_LORA_SCALE")
if FLOWMORPH_RENDER_LORA_SCALE != IMAGE_LORA_SCALE:
    raise ValueError("FlowMorph render LoRA scale must match IMAGE_LORA_SCALE")
if FLOWMORPH_GUIDANCE_SCALE != IMAGE_GUIDANCE_SCALE:
    raise ValueError("FlowMorph guidance must match IMAGE_GUIDANCE_SCALE")
for name, value in {
    "FLOWMORPH_ENDPOINT_BATCH_SIZE": FLOWMORPH_ENDPOINT_BATCH_SIZE,
    "FLOWMORPH_RENDER_BATCH_SIZE": FLOWMORPH_RENDER_BATCH_SIZE,
    "FLOWMORPH_DECODE_BATCH_SIZE": FLOWMORPH_DECODE_BATCH_SIZE,
    "SPLINE_STREAM_CHUNK_SIZE": SPLINE_STREAM_CHUNK_SIZE,
}.items():
    if value < 1:
        raise ValueError(f"{name} must be positive")
if FLOWMORPH_CFG_EXECUTION not in {"sequential", "batched"}:
    raise ValueError("FLOWMORPH_CFG_EXECUTION must be sequential or batched")
if SPLINE_MIN_FRAMES_PER_SEGMENT < 2:
    raise ValueError("SPLINE_MIN_FRAMES_PER_SEGMENT must be at least 2")
if SPLINE_FRAMES_PER_ANCHOR < SPLINE_MIN_FRAMES_PER_SEGMENT:
    raise ValueError("SPLINE_FRAMES_PER_ANCHOR is below the per-segment minimum")
if SPLINE_PREVIEW_FRAMES_PER_ANCHOR < 2:
    raise ValueError("SPLINE_PREVIEW_FRAMES_PER_ANCHOR must be at least 2")
if not 0 <= SPLINE_TIMING_DISTANCE_STRENGTH <= 1:
    raise ValueError("SPLINE_TIMING_DISTANCE_STRENGTH must lie in [0, 1]")
if not 0 < SPLINE_TIMING_DISTANCE_EXPONENT <= 1:
    raise ValueError("SPLINE_TIMING_DISTANCE_EXPONENT must lie in (0, 1]")
if not 1 <= SPLINE_TIMING_MAX_SEGMENT_RATIO <= 3:
    raise ValueError("SPLINE_TIMING_MAX_SEGMENT_RATIO must lie in [1, 3]")
if TEMPORAL_TONE_WINDOW_RADIUS < 1:
    raise ValueError("TEMPORAL_TONE_WINDOW_RADIUS must be positive")
if not 0 <= TEMPORAL_TONE_STRENGTH <= 1:
    raise ValueError("TEMPORAL_TONE_STRENGTH must lie in [0, 1]")
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

SPLINE_TOTAL_FRAMES = BASE_PROMPT_COUNT * SPLINE_FRAMES_PER_ANCHOR
print({
    "anchor_images": BASE_PROMPT_COUNT,
    "unique_endpoint_fits": BASE_PROMPT_COUNT,
    "periodic_spline_frames": SPLINE_TOTAL_FRAMES,
    "terminal_duplicate": False,
    "continuity_at_seam": "C2 (position, velocity, and curvature)",
    "timing_distance_strength": SPLINE_TIMING_DISTANCE_STRENGTH,
    "maximum_segment_ratio": SPLINE_TIMING_MAX_SEGMENT_RATIO,
    "rife_multiplier": RIFE_MULTIPLIER,
})
print("Circular anchor order:", " → ".join(ids), "→", ids[0])
'''


model_source = source(prompt_notebook["cells"][12])
anchor_source = source(prompt_notebook["cells"][14])
anchor_sheet_source = source(prompt_notebook["cells"][15])


timing_source = '''
import matplotlib.pyplot as plt
import numpy as np
from flowmorph_klein.spline_trajectory import (
    PeriodicCubicBSplineBasis,
    allocate_periodic_segment_frames,
    periodic_thumbnail_distances,
    regularized_periodic_timing,
    sample_periodic_timeline,
)

SPLINE_RAW_DISTANCES = periodic_thumbnail_distances(
    [record["path"] for record in BASE_RECORDS],
    analysis_size=SPLINE_DISTANCE_ANALYSIS_SIZE,
    color_weight=SPLINE_DISTANCE_COLOR_WEIGHT,
)
SPLINE_TIMING = regularized_periodic_timing(
    SPLINE_RAW_DISTANCES,
    distance_strength=SPLINE_TIMING_DISTANCE_STRENGTH,
    distance_exponent=SPLINE_TIMING_DISTANCE_EXPONENT,
    maximum_segment_ratio=SPLINE_TIMING_MAX_SEGMENT_RATIO,
)
SPLINE_SEGMENT_FRAME_COUNTS = allocate_periodic_segment_frames(
    SPLINE_TIMING.segment_durations,
    total_frames=SPLINE_TOTAL_FRAMES,
    minimum_frames_per_segment=SPLINE_MIN_FRAMES_PER_SEGMENT,
)
SPLINE_SAMPLES = sample_periodic_timeline(
    SPLINE_TIMING,
    SPLINE_SEGMENT_FRAME_COUNTS,
)
SPLINE_BASIS = PeriodicCubicBSplineBasis(SPLINE_TIMING.knot_times)
knot_weights = SPLINE_BASIS.weights(SPLINE_TIMING.knot_times[:-1])
if not np.allclose(
    knot_weights,
    np.eye(len(BASE_RECORDS)),
    atol=1e-9,
    rtol=0,
):
    raise RuntimeError("Periodic spline does not interpolate every anchor exactly")
for derivative in (0, 1, 2):
    seam_values = SPLINE_BASIS.weights(
        (0.0, 1.0),
        derivative=derivative,
    )
    if not np.allclose(
        seam_values[0],
        seam_values[1],
        atol=1e-9,
        rtol=0,
    ):
        raise RuntimeError(
            f"Periodic spline derivative {derivative} is discontinuous at the seam"
        )

timing_records = []
for index, record in enumerate(BASE_RECORDS):
    timing_records.append({
        "segment": index,
        "left_uid": record["uid"],
        "right_uid": BASE_RECORDS[(index + 1) % len(BASE_RECORDS)]["uid"],
        "raw_visual_distance": SPLINE_RAW_DISTANCES[index],
        "regularized_duration": SPLINE_TIMING.segment_durations[index],
        "frame_count": SPLINE_SEGMENT_FRAME_COUNTS[index],
        "knot_time": SPLINE_TIMING.knot_times[index],
    })
SPLINE_TIMING_MANIFEST = RUN_DIRECTORY / "metadata" / "periodic_spline_timing.json"
SPLINE_TIMING_MANIFEST.write_text(json.dumps({
    "method": "regularized visual-distance periodic timing",
    "distance_strength": SPLINE_TIMING_DISTANCE_STRENGTH,
    "distance_exponent": SPLINE_TIMING_DISTANCE_EXPONENT,
    "maximum_segment_ratio": SPLINE_TIMING_MAX_SEGMENT_RATIO,
    "total_unique_frames": len(SPLINE_SAMPLES),
    "terminal_duplicate": False,
    "segments": timing_records,
}, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")

figure, axis = plt.subplots(figsize=(max(9, len(BASE_RECORDS) * 0.7), 3.5))
axis.bar(
    range(len(BASE_RECORDS)),
    SPLINE_SEGMENT_FRAME_COUNTS,
    color="#536f87",
)
axis.set_xticks(range(len(BASE_RECORDS)))
axis.set_xticklabels(
    [record["uid"] for record in BASE_RECORDS],
    rotation=55,
    ha="right",
)
axis.set_ylabel("unique frames in outgoing segment")
axis.set_title("Restrained nonuniform periodic timing")
axis.grid(axis="y", alpha=0.2)
figure.tight_layout()
SPLINE_TIMING_PLOT = RUN_DIRECTORY / "previews" / "periodic_spline_timing.png"
figure.savefig(SPLINE_TIMING_PLOT, dpi=160, facecolor="white")
plt.close(figure)
timing_preview = Image.open(SPLINE_TIMING_PLOT).convert("RGB")
timing_preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
display(timing_preview)
timing_preview.close()
print({
    "timing_manifest": str(SPLINE_TIMING_MANIFEST),
    "segment_frame_counts": SPLINE_SEGMENT_FRAME_COUNTS,
    "duration_ratio": round(
        max(SPLINE_TIMING.segment_durations)
        / min(SPLINE_TIMING.segment_durations),
        4,
    ),
    "exact_anchor_knots": len(BASE_RECORDS),
    "runtime_seam_audit": "C2 passed",
    "terminal_duplicate": False,
})
'''


sequence_source = source(prompt_notebook["cells"][19])
helper_start = sequence_source.index("SEQUENCE_ROOT =")
helper_end = sequence_source.index("\nif RUN_FLOWMORPH_ONE_GAP_TEST:")
sequence_helpers = sequence_source[helper_start:helper_end]
sequence_helpers = sequence_helpers.replace(
    '"conditioning": "piecewise_source_midpoint_target_embeddings",',
    '"conditioning": "periodic_cubic_bspline_anchor_embeddings",',
)
sequence_helpers = sequence_helpers.replace(
    "ROUND_MANIFESTS = []\n"
    "FLOWMORPH_PAIR_RENDER_COUNT = 0\n"
    "OPENAI_SHARED_PROMPT_COUNT = 0\n"
    "UNIQUE_ENDPOINT_FIT_COUNT = 0\n"
    "CURRENT_RECORDS = list(BASE_RECORDS)\n",
    "UNIQUE_ENDPOINT_FIT_COUNT = 0\n",
)

session_source = '''
import gc
import hashlib
import torch
from flowmorph_klein.cli import select_hardware_profile
from flowmorph_klein.config import ProjectTemplateConfig, load_config, resolve_config
from flowmorph_klein.pipeline import FlowMorphRunner
from flowmorph_klein.sequence import FlowMorphSequenceSession, SequenceEndpointRequest

def validate_sequence_flowmorph_contract(config):
    for name, value in (("width", config.input.width), ("height", config.input.height)):
        if not 256 <= value <= 2048 or value % 16 != 0:
            raise ValueError(f"input.{name} must be 256–2048 and divisible by 16")
    if config.flowmorph.frame_count < 3:
        raise ValueError("The bootstrap FlowMorph config needs at least three prompt slots")
    if config.flowmorph.render_conditioning_mode.value != "prompt_schedule":
        raise ValueError("Sequence FlowMorph requires prompt_schedule conditioning")
    if len(config.input.bridge_prompts or ()) != config.flowmorph.frame_count:
        raise ValueError("Bootstrap prompt schedule length must equal frame_count")

ProjectTemplateConfig._validate_full_shape_contract = validate_sequence_flowmorph_contract
print("Isolated periodic-spline FlowMorph contract enabled.")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def stable_fingerprint(payload):
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()

BASE_PROMPT_TOKEN_COUNTS = {}
for record in BASE_RECORDS:
    BASE_PROMPT_TOKEN_COUNTS[record["uid"]] = {
        "prompt": validate_flux_prompt_length(
            record["prompt"],
            f"{record['uid']} FlowMorph endpoint prompt",
        ),
        "generation_prompt": validate_flux_prompt_length(
            record.get("generation_prompt", record["prompt"]),
            f"{record['uid']} anchor generation prompt",
        ),
    }
print({
    "flux_prompt_max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
    "anchor_prompt_token_counts": BASE_PROMPT_TOKEN_COUNTS,
})

# Release the fused anchor generator. One differentiable model is loaded below
# and retained for all endpoint fits and every spline render chunk.
release_flux_pipeline()
''' + sequence_helpers + '''

ensure_sequence_assets(BASE_RECORDS)
fit_sequence_endpoints(BASE_RECORDS, "Periodic endpoint fit")
if len(ENDPOINT_CACHE) != len(BASE_RECORDS):
    raise RuntimeError("Not every unique anchor endpoint is fitted")
if len(ENDPOINT_RECONSTRUCTION_PATHS) != len(BASE_RECORDS):
    raise RuntimeError("Not every fitted endpoint has a canonical reconstruction")

fit_summary = {
    "unique_endpoint_count": len(ENDPOINT_CACHE),
    "new_endpoint_fits_this_session": UNIQUE_ENDPOINT_FIT_COUNT,
    "canonical_reconstructions": {
        uid: str(path)
        for uid, path in ENDPOINT_RECONSTRUCTION_PATHS.items()
    },
    "session_contract": SEQUENCE_SESSION_CONTRACT,
    "session_fingerprint": SEQUENCE_SESSION_FINGERPRINT,
}
(RUN_DIRECTORY / "metadata" / "periodic_endpoint_fit_summary.json").write_text(
    json.dumps(fit_summary, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8",
)
print({
    "unique_endpoints_ready": len(ENDPOINT_CACHE),
    "fitted_only_once_per_unique_image": True,
    "canonical_endpoint_reconstructions": len(ENDPOINT_RECONSTRUCTION_PATHS),
    "model_loads": 1,
    "backward_probes": 1,
})
'''


preview_source = '''
from flowmorph_klein.spline_trajectory import (
    PeriodicConditioningSpline,
    PeriodicFlowMorphSpline,
    PeriodicSplineFlowMorphRenderer,
)

SPLINE_ENDPOINTS = [ENDPOINT_CACHE[record["uid"]] for record in BASE_RECORDS]
SPLINE_CONDITIONINGS = [
    PROMPT_CONDITIONING_CACHE[record["prompt"]]
    for record in BASE_RECORDS
]
SPLINE_STATE_TRAJECTORY = PeriodicFlowMorphSpline(
    SPLINE_ENDPOINTS,
    SPLINE_BASIS,
)
SPLINE_PROMPT_TRAJECTORY = PeriodicConditioningSpline(
    SPLINE_CONDITIONINGS,
    SPLINE_BASIS,
)
SPLINE_RENDERER = PeriodicSplineFlowMorphRenderer(
    SEQUENCE_SESSION,
    SPLINE_STATE_TRAJECTORY,
    SPLINE_PROMPT_TRAJECTORY,
)

if RUN_SPLINE_COARSE_PREVIEW:
    preview_counts = tuple(
        SPLINE_PREVIEW_FRAMES_PER_ANCHOR for _ in BASE_RECORDS
    )
    preview_samples = sample_periodic_timeline(SPLINE_TIMING, preview_counts)
    preview_fingerprint = stable_fingerprint({
        "session_fingerprint": SEQUENCE_SESSION_FINGERPRINT,
        "endpoint_fingerprints": [
            ENDPOINT_FINGERPRINTS[record["uid"]]
            for record in BASE_RECORDS
        ],
        "sample_times": [item.time for item in preview_samples],
    })
    preview_directory = (
        RUN_DIRECTORY
        / "trials"
        / f"periodic_spline_coarse_{preview_fingerprint[:12]}"
    )
    preview_directory.mkdir(parents=True, exist_ok=True)
    preview_records = []
    interior_samples = [
        item for item in preview_samples if item.anchor_index is None
    ]
    interior_paths = [
        preview_directory / f"frame_{item.frame_index:05d}.png"
        for item in interior_samples
    ]
    missing = [
        (sample, path)
        for sample, path in zip(interior_samples, interior_paths, strict=True)
        if not path.is_file()
    ]
    if missing:
        frames = SPLINE_RENDERER.render([item.time for item, _ in missing])
        SEQUENCE_SESSION.decode_frames_to_paths(
            frames,
            [path for _, path in missing],
        )
        del frames
    interior_lookup = {
        sample.frame_index: path
        for sample, path in zip(interior_samples, interior_paths, strict=True)
    }
    for sample in preview_samples:
        if sample.anchor_index is not None:
            anchor = BASE_RECORDS[sample.anchor_index]
            path = ENDPOINT_RECONSTRUCTION_PATHS[anchor["uid"]]
            label = f"{anchor['uid']} exact knot"
        else:
            path = interior_lookup[sample.frame_index]
            label = (
                f"segment {sample.segment_index} "
                f"{sample.segment_fraction:.2f}"
            )
        preview_records.append({"path": str(path), "label": label})
    thumbnails = []
    for item in preview_records:
        with Image.open(item["path"]) as opened:
            thumbnail = opened.convert("RGB")
            thumbnail.thumbnail((256, 256))
            thumbnails.append(thumbnail)
    preview_sheet = preview_directory / "periodic_spline_coarse_sheet.png"
    make_contact_sheet(
        thumbnails,
        preview_sheet,
        columns=4,
        labels=[item["label"] for item in preview_records],
    )
    for image in thumbnails:
        image.close()
    shown = Image.open(preview_sheet).convert("RGB")
    shown.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
    display(Markdown("### Coarse global periodic spline quality gate"))
    display(shown)
    shown.close()
    print({
        "preview_sheet": str(preview_sheet),
        "preview_unique_frames": len(preview_records),
        "includes_last_to_first_segment": True,
        "terminal_duplicate": False,
    })
else:
    print("Coarse periodic spline preview skipped.")
'''


render_source = '''
SPLINE_RENDER_CONTRACT = {
    "session_fingerprint": SEQUENCE_SESSION_FINGERPRINT,
    "endpoint_fingerprints": [
        ENDPOINT_FINGERPRINTS[record["uid"]]
        for record in BASE_RECORDS
    ],
    "knot_times": list(SPLINE_TIMING.knot_times),
    "segment_frame_counts": list(SPLINE_SEGMENT_FRAME_COUNTS),
    "sample_times": [item.time for item in SPLINE_SAMPLES],
    "state_curve": "periodic interpolating cubic B-spline over z, delta, and direction/log-magnitude u",
    "conditioning_curve": "same periodic cubic B-spline over anchor prompt embeddings",
    "seam_continuity": "C2",
    "terminal_duplicate": False,
}
SPLINE_RENDER_FINGERPRINT = stable_fingerprint(SPLINE_RENDER_CONTRACT)
SPLINE_FRAME_DIRECTORY = (
    RUN_DIRECTORY
    / "periodic_spline"
    / f"frames_{SPLINE_RENDER_FINGERPRINT[:12]}"
)
SPLINE_FRAME_DIRECTORY.mkdir(parents=True, exist_ok=True)
FINAL_SEQUENCE_MANIFEST = (
    RUN_DIRECTORY / "metadata" / "final_periodic_bspline_flowmorph_sequence.json"
)

interior_jobs = []
FINAL_RECORDS = []
for sample in SPLINE_SAMPLES:
    if sample.anchor_index is not None:
        anchor = BASE_RECORDS[sample.anchor_index]
        frame_path = ENDPOINT_RECONSTRUCTION_PATHS[anchor["uid"]]
        kind = "canonical_fitted_anchor"
        anchor_uid = anchor["uid"]
    else:
        frame_path = (
            SPLINE_FRAME_DIRECTORY
            / f"{sample.frame_index:07d}_t{sample.time:.10f}.png"
        )
        kind = "periodic_bspline_interior"
        anchor_uid = None
        if not (
            SPLINE_REUSE_RENDERED_FRAMES
            and frame_path.is_file()
        ):
            interior_jobs.append((sample, frame_path))
    FINAL_RECORDS.append({
        "uid": f"spline_{sample.frame_index:07d}",
        "kind": kind,
        "path": str(frame_path),
        "time": sample.time,
        "segment_index": sample.segment_index,
        "segment_fraction": sample.segment_fraction,
        "anchor_uid": anchor_uid,
        "render_fingerprint": SPLINE_RENDER_FINGERPRINT,
    })

for start in range(0, len(interior_jobs), SPLINE_STREAM_CHUNK_SIZE):
    chunk = interior_jobs[start:start + SPLINE_STREAM_CHUNK_SIZE]
    frames = SPLINE_RENDERER.render([item.time for item, _ in chunk])
    paths = [path for _, path in chunk]
    SEQUENCE_SESSION.decode_frames_to_paths(frames, paths)
    del frames
    completed = min(start + len(chunk), len(interior_jobs))
    print(
        f"Periodic spline interiors rendered: {completed}/"
        f"{len(interior_jobs)}; latest={paths[-1].name}",
        flush=True,
    )

missing_final_paths = [
    item["path"] for item in FINAL_RECORDS
    if not Path(item["path"]).is_file()
]
if missing_final_paths:
    raise FileNotFoundError(
        "Periodic render is incomplete; first missing path: "
        + missing_final_paths[0]
    )

final_payload = {
    "project": PROJECT_NAME,
    "method": "global periodic cubic B-spline through fitted FlowMorph endpoints",
    "render_contract": SPLINE_RENDER_CONTRACT,
    "render_fingerprint": SPLINE_RENDER_FINGERPRINT,
    "final_count": len(FINAL_RECORDS),
    "anchor_count": len(BASE_RECORDS),
    "interior_count": sum(
        item["kind"] == "periodic_bspline_interior"
        for item in FINAL_RECORDS
    ),
    "exact_canonical_anchor_knots": True,
    "seam_continuity": "C2",
    "terminal_duplicate": False,
    "records": FINAL_RECORDS,
}
FINAL_SEQUENCE_MANIFEST.write_text(
    json.dumps(final_payload, indent=2, ensure_ascii=False) + "\\n",
    encoding="utf-8",
)
print({
    "final_manifest": str(FINAL_SEQUENCE_MANIFEST),
    "unique_frames": len(FINAL_RECORDS),
    "new_interiors_rendered": len(interior_jobs),
    "canonical_anchor_reuses": len(BASE_RECORDS),
    "smooth_last_to_first_segment": True,
    "terminal_duplicate": False,
})

# RIFE needs the GPU next. The PNGs and manifest are already safely on Drive.
for name in (
    "SPLINE_RENDERER",
    "SPLINE_STATE_TRAJECTORY",
    "SPLINE_PROMPT_TRAJECTORY",
    "SPLINE_ENDPOINTS",
    "SPLINE_CONDITIONINGS",
    "ENDPOINT_CACHE",
    "IMAGE_ASSET_CACHE",
    "PROMPT_CONDITIONING_CACHE",
    "SEQUENCE_SESSION",
    "SEQUENCE_RUNNER",
):
    globals().pop(name, None)
gc.collect()
torch.cuda.empty_cache()
print("Released the periodic FlowMorph model; RIFE can now use the GPU.")
'''


assembly_heading = '''
## 11. Stabilize, preview, and audit the circular spline sequence

The raw output is already a closed curve: there is no special final pair and no
duplicated first frame. Every fitted anchor appears exactly once. Optional
temporal tone stabilization writes corrected copies only; raw spline PNGs remain
untouched. The quiet-cut rotation changes only where playback begins.
'''

assembly_source = source(prompt_notebook["cells"][24])
assembly_source = assembly_source.replace(
    "final_recursive_flowmorph_sequence_tone_stabilized.json",
    "final_periodic_bspline_flowmorph_sequence_tone_stabilized.json",
)
assembly_source = assembly_source.replace(
    "final_recursive_flowmorph_sequence.json",
    "final_periodic_bspline_flowmorph_sequence.json",
)


rife_prepare_source = source(prompt_notebook["cells"][26])
rife_run_source = source(prompt_notebook["cells"][28]).replace(
    "final_recursive_flowmorph_sequence.json",
    "final_periodic_bspline_flowmorph_sequence.json",
)
rife_finish_source = source(prompt_notebook["cells"][30]).replace(
    "recursive_flowmorph_prompt_only_rife_ssim_loop.mp4",
    "periodic_bspline_flowmorph_rife_ssim_loop.mp4",
)
flicker_source = source(prompt_notebook["cells"][32])
flicker_source = flicker_source.replace(
    "final_recursive_flowmorph_sequence.json",
    "final_periodic_bspline_flowmorph_sequence.json",
)
flicker_source = flicker_source.replace(
    "final_recursive_flowmorph_sequence_tone_stabilized.json",
    "final_periodic_bspline_flowmorph_sequence_tone_stabilized.json",
)
old_gap = (
    "    final_gap_size = (\n"
    "        int(FLOWMORPH_ROUND_SPECS[-1][\"midpoint_count\"]) + 1\n"
    "    )\n"
)
new_gap = (
    "    final_gap_size = max(\n"
    "        2,\n"
    "        int(round(len(diagnostic_records) / BASE_PROMPT_COUNT)),\n"
    "    )\n"
)
if old_gap not in flicker_source:
    raise RuntimeError("Could not locate pairwise flicker gap calculation")
flicker_source = flicker_source.replace(old_gap, new_gap, 1)


cells = [
    markdown(
        """
        # Periodic B-spline still-life loop — global true FlowMorph trajectory

        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MNoichl/FluxFlowMorph/blob/main/notebooks/StillLife_Periodic_BSpline_FlowMorph.ipynb)

        This is a fully separate experimental path. It does not change the
        working pairwise FlowMorph implementation or notebooks.

        Edit `BASE_STAGES` near the top. The notebook generates the anchors,
        fits each unique endpoint once, and constructs one interpolating
        periodic cubic B-spline through all fitted FlowMorph states and anchor
        prompt embeddings. The anchors are exact knots; position, velocity, and
        curvature match across the last→first seam. Restrained image-distance
        timing gives harder gaps a little more time without allowing one gap to
        dominate. The final first frame is intentionally not duplicated.

        A coarse global preview precedes the full render. Canonical endpoint
        reconstructions, raw spline frames, optional tone-corrected copies,
        manifests, diagnostics, and the RIFE/SSIM H.264 loop are written
        directly into the timestamped Drive run.
        """
    ),
    markdown(
        """
        ## 1. Editable prompt, anchor, spline, FlowMorph, and video settings

        The main spline controls are frames per anchor and the three timing
        regularizers. `SPLINE_TIMING_DISTANCE_STRENGTH=0` gives uniform timing.
        The default retains 55% uniform timing and caps the largest/smallest
        segment-duration ratio at 1.75.
        """
    ),
    code(settings_source),
    copy.deepcopy(prompt_notebook["cells"][3]),
    code(preserved_stage_source),
    markdown(
        """
        ## 3. GPU, repository, and compatible dependencies

        This uses the same proven Colab environment as the pairwise notebook.
        A dependency install requests one kernel restart; subsequent healthy
        imports do not reinstall.
        """
    ),
    code(install_source),
    markdown(
        """
        ## 4. Mount Drive and reserve or resume a numbered run

        No OpenAI key is needed: prompt conditioning is splined directly from
        the editable anchor prompts.
        """
    ),
    code(run_directory_source),
    markdown(
        """
        ## 5. Validate the circular spline contract and preview cost

        At least four anchors are required. The reported frame count is the
        unique pre-RIFE frame count; the opening anchor is not repeated at the
        end.
        """
    ),
    code(validation_source),
    markdown(
        """
        ## 6. Load RIJKSOIL and optionally test one random anchor

        Use this quick image to tune LoRA, guidance, dimensions, and inference
        steps before generating the complete anchor cycle.
        """
    ),
    code(model_source),
    markdown(
        """
        ## 7. Generate prompt-only circular anchor paintings

        The first is text-to-image. Later anchors may use a blurred, grained
        previous image as an ordinary latent img2img start. No flat beige
        canvas, masks, or post-compositing are involved.
        """
    ),
    code(anchor_source),
    code(anchor_sheet_source),
    markdown(
        """
        ## 8. Estimate restrained nonuniform timing around the complete loop

        Thumbnail color/gradient distance is only a timing proxy. Square-root
        tempering, a uniform blend, and a hard ratio cap prevent extreme
        dwell-time changes. The closing last→first distance is included.
        """
    ),
    code(timing_source),
    markdown(
        """
        ## 9. Fit every unique endpoint once and decode canonical knots

        One model remains loaded for the entire sequence and the backward
        preflight runs once. Each image is fitted once even though the global
        curve enters and leaves it. The decoded reconstruction is the exact
        knot used by the final sequence.
        """
    ),
    code(session_source),
    markdown(
        """
        ## 10. Coarse global spline quality gate

        This renders one interior sample in every circular segment, including
        the closing last→first segment. It tests the actual global state and
        prompt-embedding spline before the full render.
        """
    ),
    code(preview_source),
    markdown(
        """
        ### Full periodic B-spline FlowMorph render

        Frames stream to Drive in bounded chunks. Rerunning with an unchanged
        render fingerprint reuses completed PNGs. Exact anchor knots use their
        canonical fitted reconstructions and are never rerendered.
        """
    ),
    code(render_source),
    markdown(assembly_heading),
    code(assembly_source),
    markdown(
        """
        ## 12. Prepare pinned Practical-RIFE and its v4.25 model

        RIFE receives the lossless unique spline PNGs plus one temporary copy
        of frame zero, solely so it can interpolate the circular wrap edge.
        """
    ),
    code(rife_prepare_source),
    markdown(
        """
        ## 13. Interpolate every circular pair with RIFE

        The temporary terminal copy is verified pixel-identical and removed
        from the unique dense sequence after the wrap pair is interpolated.
        """
    ),
    code(rife_run_source),
    markdown(
        """
        ## 14. Circular SSIM motion equalization and final H.264 loop

        Circular `1 − SSIM` redistributes playback samples by visible motion.
        The exported H.264 still contains no duplicated terminal frame.
        """
    ),
    code(rife_finish_source),
    markdown(
        """
        ## 15. Read-only circular flicker diagnosis

        This analyzes the raw periodic-spline frames without modifying them and
        saves the full plot and JSON report to Drive.
        """
    ),
    code(flicker_source),
]

for index, cell in enumerate(cells):
    cell["id"] = f"periodic-bspline-flowmorph-{index:02d}"
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook = {
    "cells": cells,
    "metadata": copy.deepcopy(prompt_notebook.get("metadata", {})),
    "nbformat": 4,
    "nbformat_minor": 5,
}
notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(cells)} clean cells")
