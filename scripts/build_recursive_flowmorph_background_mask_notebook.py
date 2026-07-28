"""Build the separate background-mask trajectory FlowMorph notebook."""

from __future__ import annotations

import json
import os
import pprint
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Trajectory_Init.ipynb"
OUTPUT = Path(
    os.environ.get(
        "FLOWMORPH_BACKGROUND_MASK_NOTEBOOK_OUTPUT",
        ROOT
        / "notebooks"
        / "StillLife_Recursive_FlowMorph_Trajectory_Background_Mask.ipynb",
    )
)
if (
    "FLOWMORPH_BACKGROUND_MASK_NOTEBOOK_OUTPUT" not in os.environ
    and OUTPUT.exists()
    and os.environ.get("FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE") != "1"
):
    raise RuntimeError(
        "Refusing to overwrite the tracked mask notebook. Set a temporary "
        "FLOWMORPH_BACKGROUND_MASK_NOTEBOOK_OUTPUT or explicitly set "
        "FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE=1."
    )


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def code_blocks(*blocks: str) -> list[str]:
    joined = "\n\n".join(dedent(block).strip("\n") for block in blocks)
    return (joined + "\n").splitlines(keepends=True)


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def find_cell(notebook: dict, needle: str) -> dict:
    matches = [cell for cell in notebook["cells"] if needle in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one notebook cell containing {needle!r}")
    return matches[0]


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index:]


# These are the user-edited prompts recovered from the last executed notebook.
# Keep them here so rebuilding the notebook cannot silently restore the sparse
# prompts inherited from the trajectory-init template.
PRESERVED_BASE_STAGES = [
    {
        "id": "01_cosmos_physics",
        "science": "Astronomy & Physics",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of the physical cosmos: a flowing garland of night-blooming moonflowers, evening primrose, and star-shaped blossoms spreading across the whole panel, comet-vines and tendrils curling into every corner, sun, moon, and scattered stars set as small emblems among the leaves, a delicate armillary sphere and a prism woven into the scrollwork. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, chalky indigo-blue and pale gold, antique and weathered.",
    },
    {
        "id": "02_chemistry_materials",
        "science": "Chemistry & Materials",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of transformation: flowering vines and scrolling acanthus climbing the full height of the panel around slender alembics and a pear-shaped retort, crystalline buds and mineral clusters sprouting all along the stems, sulphur-yellow blossoms and looping distillation coils threading the whole field, a small glowing crucible set among them. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, warm amber and soft verdigris, antique and weathered.",
    },
    {
        "id": "03_geosciences",
        "science": "Geosciences & Earth",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of earth and sky: a dense braid of fern, moss, and ivy filling the panel, banded agates and small shells clustered along the swags, a terrestrial globe wreathed in leaves at the crook of the scroll, cloud and wave motifs rippling through the ornament from edge to edge. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, cool slate-blue and mineral green, antique and weathered.",
    },
    {
        "id": "04_ecology_evolution",
        "science": "Ecology & Evolution",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of life and deep time: a teeming all-over garland of wildflowers, grasses, and ferns alive with insects, snails, and small birds, spiral ammonites curling as volutes through the scrollwork, a branching tree-of-life motif spreading its limbs across the field, chrysalises and seed-pods tucked among the blooms. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, mossy green and fossil-ochre, antique and weathered.",
    },
    {
        "id": "05_botany",
        "science": "Botany & Plant Science",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of the plant kingdom: a rich, full festoon of tulips, roses, poppies, and fruiting branches wound with wheat and vine across the whole panel, herbarium sprigs with small labels threaded through the scroll, a comb of honey and a magnifier nestled among the abundant blooms. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, verdant green with rose-red and gold, antique and weathered.",
    },
    {
        "id": "06_genetics_cell",
        "science": "Genetics, Cell & Molecular Biology",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of heredity and the cell: pea-pod tendrils twisting into paired helical scrolls that spread across the panel, split pomegranates set as seeded medallions among the leaves, a honeycomb lattice running as a border, rosettes of clustered berries repeating like cells through the ornament. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, pale pearl-rose and warm ochre, antique and weathered.",
    },
    {
        "id": "07_medicine_body",
        "science": "Medicine & the Body",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of healing: a full garland of medicinal poppy, foxglove, and willow winding across the panel around a serpent-and-staff entwined with flowering vines, an anatomical heart set as a small medallion in the scroll, apothecary phials and a quiet memento-mori skull half-hidden among the abundant leaves. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, soft carmine and ivory with herb-green, antique and weathered.",
    },
    {
        "id": "08_neuroscience_mind",
        "science": "Neuroscience & Mind",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of the mind: branching dendrite-vines and coral spreading in a dense all-over tracery across the panel, a brain worked as an ornate medallion at the heart of the scroll, moths and songbirds perched throughout the tendrils, a small labyrinth and a pendulum woven into the ornament. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, chalky blue-violet and dim silver, antique and weathered.",
    },
    {
        "id": "09_philosophy_society",
        "science": "Philosophy & Social Sciences",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of thought and society: a flowing garland of laurel and olive filling the panel around a cartouche of an owl perched on books beside an hourglass, scales of justice and a globe woven into the scrollwork, Platonic solids set as ornamental bosses repeating among the leaves. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, warm sepia and old gold, antique and weathered.",
    },
    {
        "id": "10_computation_math",
        "science": "Computation & Mathematics",
        "prompt": "RIJKSOIL, a floral fresco wall ornament of pure form: a dense interlacing arabesque lattice blooming into stylized geometric flowers across the whole panel, an orrery and interlocking clock gears set within the scroll, compasses and nested Platonic solids repeating through the ornament, a spiral of stars winding through it to close the circle. Painted as a floral fresco wall ornament on aged lime plaster, light airy brushstrokes and delicate translucent washes, a richly ornamental composition of flowing garlands, festoons, and scrolling acanthus spreading to fill the whole panel, densely distributed yet weightless, in the manner of Renaissance grottesche and Baroque wall painting, chalky faded pigments and fine hairline cracks, cool silver-grey and brass turning toward indigo, antique and weathered.",
    },
]


notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
notebook["cells"] = [
    cell
    for cell in notebook["cells"]
    if cell.get("cell_type") != "code" or source(cell).strip()
]
for index, cell in enumerate(notebook["cells"]):
    cell.setdefault("id", f"background-mask-cell-{index:02d}")
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        cell["source"] = (
            source(cell)
            .replace(
                '"repository_commit": project_com mit,',
                '"repository_commit": project_commit,',
            )
            .splitlines(keepends=True)
        )

notebook["cells"][0]["source"] = lines(
    """
    # Recursive science still-life loop — background-mask trajectory + true FlowMorph

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MNoichl/FluxFlowMorph/blob/main/notebooks/StillLife_Recursive_FlowMorph_Trajectory_Background_Mask.ipynb)

    This experiment loads grayscale images from a ZIP as soft spatial initializers:

    - dark mask values become a settable beige field with faint residual noise;
    - bright values become deterministic RGB noise softly tinted toward beige;
    - a Gaussian-smoothed copy of the grayscale mask blends those two fields;
    - every anchor uses the resulting image through conventional latent img2img;
    - later anchors also mix in a weak blurred and grained previous image;
    - the effective mask and unchanged original prompt are sent to the OpenAI
      vision model so it can write a detailed prompt around the actual geometry;
    - the remotely adapted prompt must begin with the LoRA trigger exactly once;
    - there is no post-generation mask or exact compositing step.

    Run the trial first to inspect the smoothed activity mask and actual img2img init.
    """
)
notebook["cells"][1]["source"] = lines(
    """
    ## 1. Editable run, background-mask, model, API, FlowMorph, image, and video settings

    Point the mask ZIP settings at naturally ordered black/white or grayscale files.
    `MASK_INIT_BACKGROUND_RGB` controls the quiet field. Two endpoint-mix controls
    tint the active noise toward beige and retain faint noise in the quiet field;
    `MASK_INIT_GAUSSIAN_BLUR` softens the spatial transition between them.
    This is only an img2img initialization: FLUX may repaint and cross its boundaries.
    """
)

settings_cell = find_cell(
    notebook,
    "# Strong trajectory initialization for independently generated anchors.",
)
settings = source(settings_cell).replace(
    'PROJECT_NAME = "science_path_trajectory_flowmorph"',
    'PROJECT_NAME = "science_path_trajectory_background_mask"',
    1,
).replace(
    "FLOWMORPH_FIT_LORA_SCALE = 1.0",
    "FLOWMORPH_FIT_LORA_SCALE = 1.2",
    1,
).replace(
    "FLOWMORPH_RENDER_LORA_SCALE = 1.0",
    "FLOWMORPH_RENDER_LORA_SCALE = 1.2",
    1,
).replace(
    "FLOWMORPH_GUIDANCE_SCALE = 4.0",
    "FLOWMORPH_GUIDANCE_SCALE = 7.0",
    1,
).replace(
    "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 100",
    "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 50",
    1,
).replace(
    "FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 100",
    "FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 50",
    1,
).replace(
    "IMAGE_INFERENCE_STEPS = 28",
    "IMAGE_INFERENCE_STEPS = 50",
    1,
).replace(
    "IMAGE_GUIDANCE_SCALE = 4.0",
    "IMAGE_GUIDANCE_SCALE = 7.0",
    1,
).replace(
    "IMAGE_LORA_SCALE = 1.0",
    "IMAGE_LORA_SCALE = 1.2",
    1,
).replace(
    "BASE_SEED = 98123479812734987 #1729",
    "BASE_SEED = 42 #98123479812734987 #1729",
    1,
).replace(
    "SOURCE_SEQUENCE_FPS = 12.0",
    """VIDEO_SLOWDOWN_FACTOR = 3.0
SOURCE_SEQUENCE_FPS = 12.0 / VIDEO_SLOWDOWN_FACTOR""",
    1,
).replace(
    "RIFE_MULTIPLIER = 2",
    "RIFE_MULTIPLIER = int(round(2 * VIDEO_SLOWDOWN_FACTOR))",
    1,
)
mask_settings = dedent(
    """
    # Grayscale-mask ZIP. Files are regularly sampled to match the active prompts.
    MASK_ZIP_DIRECTORY = "/content/drive/MyDrive/FluxFlowMorphArt/trajectory_inputs"
    MASK_ZIP_FILENAME = "mask_2.zip"
    MASK_MEMBER_PREFIX = ""  # Optional folder inside the ZIP, without leading slash.
    MASK_FRAME_OFFSET = 0
    MASK_REVERSE_ORDER = False

    # Direct mask interpretation. Defaults preserve every input grayscale value.
    MASK_INVERT = True
    MASK_GAMMA = 1.0
    MASK_EXPANSION = 0
    MASK_FEATHER = 0.0
    MASK_MIN_EDITABLE_FRACTION = 0.001
    MASK_MAX_EDITABLE_FRACTION = 1.0

    # Remote vision planning. BASE_STAGES stays untouched and directly editable.
    MASK_PROMPT_REWRITE_ENABLED = True
    MASK_PROMPT_REWRITE_IMAGE_DETAIL = "original"
    MASK_PROMPT_REWRITE_MAX_OUTPUT_TOKENS = 6000
    MASK_PROMPT_REWRITE_MAX_ATTEMPTS = 3
    MASK_PROMPT_REWRITE_REUSE_CACHE = True
    MASK_PROMPT_REWRITE_FORCE_REFRESH = False
    MASK_PROMPT_REWRITE_DISPLAY = True
    MASK_PROMPT_REWRITE_DISPLAY_MASK = True
    MASK_PROMPT_DISPLAY_MAX_SIDE = 320
    MASK_PROMPT_CACHE_SUBDIRECTORY = "_mask_prompt_cache"
    MASK_PROMPT_GEOMETRY_THRESHOLD = 0.35
    MASK_PROMPT_MIN_COMPONENT_FRACTION = 0.0005
    MASK_PROMPT_MAX_COMPONENTS = 12
    # Official Flux2KleinPipeline prompt sequence limit, including chat-template tokens.
    FLUX_PROMPT_MAX_SEQUENCE_LENGTH = 512

    # Soft noisy mask initialization. No mask is applied after FLUX decoding.
    MASK_INIT_BACKGROUND_RGB = (238, 233, 218)
    MASK_INIT_GAUSSIAN_BLUR = 64.0
    MASK_INIT_NOISE_LOW = 0
    MASK_INIT_NOISE_HIGH = 255
    # 0 = pure RGB noise; 1 = completely beige at the bright endpoint.
    MASK_INIT_NOISE_BEIGE_MIX = 0.18
    # 0 = flat beige; larger values retain more noise at the dark endpoint.
    MASK_INIT_BACKGROUND_NOISE_MIX = 0.09
    MASK_INIT_DENOISE_STRENGTH = 0.75
    # Previous-reference influence at fully bright/active and fully dark/quiet
    # points of the Gaussian-smoothed mask. Intermediate grayscale values blend
    # continuously between these two strengths.
    MASK_INIT_PREVIOUS_ACTIVE_MIX = 0.18
    MASK_INIT_PREVIOUS_QUIET_MIX = 0.03
    SAVE_MASK_INITS = True

    # Weak sequential continuity mixed into the noisy mask init for anchors 2+.
    PREVIOUS_INIT_ENABLED = True
    PREVIOUS_INIT_BLEND = 0.12
    PREVIOUS_INIT_BLUR = 16.0
    PREVIOUS_INIT_GRAIN_STRENGTH = 0.035
    PREVIOUS_INIT_BACKGROUND_RGB = MASK_INIT_BACKGROUND_RGB
    SAVE_PREVIOUS_INITS = True

    # Optional post-FlowMorph tonal outlier correction before previews and RIFE.
    # Raw FlowMorph PNGs are never overwritten. Only detected outliers receive
    # corrected copies; unchanged records continue to reference their raw PNGs.
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

    # Read-only flicker diagnosis rendered at the very end of the notebook.
    RUN_FLICKER_DIAGNOSTIC = True
    FLICKER_ANALYSIS_MAX_SIDE = 256
    FLICKER_OUTLIER_MAD_MULTIPLIER = 3.5
    FLICKER_MINIMUM_OUTLIER_SCORE = 3.0
    FLICKER_MAX_LAG = 64

    TRAJECTORY_REMOVE_SYMMETRY_LANGUAGE = True

    """
).lstrip()
settings = replace_between(
    settings,
    "# Strong trajectory initialization for independently generated anchors.\n",
    "# Trial and notebook display.\n",
    mask_settings,
)
settings_cell["source"] = settings.splitlines(keepends=True)

prompts_cell = find_cell(notebook, "SCIENCE_FRESCOES = [")
prompts_cell["source"] = (
    "BASE_STAGES = "
    + pprint.pformat(
        PRESERVED_BASE_STAGES,
        sort_dicts=False,
        width=100,
    )
    + "\n"
).splitlines(keepends=True)

validation_cell = find_cell(notebook, "def trajectory_generation_prompt")
validation = source(validation_cell)
mask_validation = dedent(
    """
    if len(MASK_INIT_BACKGROUND_RGB) != 3 or any(
        not 0 <= channel <= 255 for channel in MASK_INIT_BACKGROUND_RGB
    ):
        raise ValueError("MASK_INIT_BACKGROUND_RGB must contain three values in [0, 255]")
    if MASK_GAMMA <= 0:
        raise ValueError("MASK_GAMMA must be positive")
    if not isinstance(MASK_EXPANSION, int) or not 0 <= MASK_EXPANSION <= 128:
        raise ValueError("MASK_EXPANSION must be an integer in [0, 128]")
    if MASK_FEATHER < 0:
        raise ValueError("MASK_FEATHER cannot be negative")
    if not 0 <= MASK_MIN_EDITABLE_FRACTION < MASK_MAX_EDITABLE_FRACTION <= 1:
        raise ValueError("Editable-fraction limits must satisfy 0 <= min < max <= 1")
    if MASK_INIT_GAUSSIAN_BLUR < 0:
        raise ValueError("MASK_INIT_GAUSSIAN_BLUR cannot be negative")
    if not (
        isinstance(MASK_INIT_NOISE_LOW, int)
        and isinstance(MASK_INIT_NOISE_HIGH, int)
        and 0 <= MASK_INIT_NOISE_LOW < MASK_INIT_NOISE_HIGH <= 255
    ):
        raise ValueError(
            "MASK_INIT_NOISE_LOW/HIGH must be integers satisfying 0 <= low < high <= 255"
        )
    if not 0 <= MASK_INIT_NOISE_BEIGE_MIX <= 1:
        raise ValueError("MASK_INIT_NOISE_BEIGE_MIX must lie in [0, 1]")
    if not 0 <= MASK_INIT_BACKGROUND_NOISE_MIX <= 1:
        raise ValueError("MASK_INIT_BACKGROUND_NOISE_MIX must lie in [0, 1]")
    if MASK_INIT_NOISE_BEIGE_MIX + MASK_INIT_BACKGROUND_NOISE_MIX >= 1:
        raise ValueError(
            "MASK_INIT_NOISE_BEIGE_MIX + MASK_INIT_BACKGROUND_NOISE_MIX "
            "must be less than 1 so bright areas remain noisier than dark areas"
        )
    if not 0 < MASK_INIT_DENOISE_STRENGTH <= 1:
        raise ValueError("MASK_INIT_DENOISE_STRENGTH must lie in (0, 1]")
    if not 0 <= MASK_INIT_PREVIOUS_QUIET_MIX <= MASK_INIT_PREVIOUS_ACTIVE_MIX <= 1:
        raise ValueError(
            "Previous-image mixes must satisfy 0 <= quiet <= active <= 1"
        )
    if not 0 < PREVIOUS_INIT_BLEND <= 1:
        raise ValueError("PREVIOUS_INIT_BLEND must lie in (0, 1]")
    if PREVIOUS_INIT_BLUR < 0:
        raise ValueError("PREVIOUS_INIT_BLUR cannot be negative")
    if not 0 <= PREVIOUS_INIT_GRAIN_STRENGTH <= 0.25:
        raise ValueError("PREVIOUS_INIT_GRAIN_STRENGTH must lie in [0, 0.25]")
    if len(PREVIOUS_INIT_BACKGROUND_RGB) != 3 or any(
        not 0 <= channel <= 255 for channel in PREVIOUS_INIT_BACKGROUND_RGB
    ):
        raise ValueError("PREVIOUS_INIT_BACKGROUND_RGB must contain three values in [0, 255]")
    if TEMPORAL_TONE_WINDOW_RADIUS < 1:
        raise ValueError("TEMPORAL_TONE_WINDOW_RADIUS must be positive")
    if not 0 <= TEMPORAL_TONE_STRENGTH <= 1:
        raise ValueError("TEMPORAL_TONE_STRENGTH must lie in [0, 1]")
    if TEMPORAL_TONE_MEAN_THRESHOLD < 0:
        raise ValueError("TEMPORAL_TONE_MEAN_THRESHOLD cannot be negative")
    if TEMPORAL_TONE_CONTRAST_THRESHOLD < 0:
        raise ValueError("TEMPORAL_TONE_CONTRAST_THRESHOLD cannot be negative")
    if TEMPORAL_TONE_MAD_MULTIPLIER < 0:
        raise ValueError("TEMPORAL_TONE_MAD_MULTIPLIER cannot be negative")
    if not 0 <= TEMPORAL_TONE_MAX_MEAN_SHIFT <= 1:
        raise ValueError("TEMPORAL_TONE_MAX_MEAN_SHIFT must lie in [0, 1]")
    if not 0 <= TEMPORAL_TONE_MAX_CONTRAST_SCALE_DELTA < 1:
        raise ValueError(
            "TEMPORAL_TONE_MAX_CONTRAST_SCALE_DELTA must lie in [0, 1)"
        )
    if TEMPORAL_TONE_ANALYSIS_MAX_SIDE < 32:
        raise ValueError("TEMPORAL_TONE_ANALYSIS_MAX_SIDE must be at least 32")
    if VIDEO_SLOWDOWN_FACTOR < 1:
        raise ValueError("VIDEO_SLOWDOWN_FACTOR must be at least 1")
    if FLICKER_ANALYSIS_MAX_SIDE < 32:
        raise ValueError("FLICKER_ANALYSIS_MAX_SIDE must be at least 32")
    if FLICKER_OUTLIER_MAD_MULTIPLIER < 0:
        raise ValueError("FLICKER_OUTLIER_MAD_MULTIPLIER cannot be negative")
    if FLICKER_MINIMUM_OUTLIER_SCORE < 0:
        raise ValueError("FLICKER_MINIMUM_OUTLIER_SCORE cannot be negative")
    if FLICKER_MAX_LAG < 1:
        raise ValueError("FLICKER_MAX_LAG must be positive")
    if MASK_PROMPT_REWRITE_IMAGE_DETAIL not in {"low", "high", "original", "auto"}:
        raise ValueError(
            "MASK_PROMPT_REWRITE_IMAGE_DETAIL must be low, high, original, or auto"
        )
    if MASK_PROMPT_REWRITE_MAX_OUTPUT_TOKENS < 1000:
        raise ValueError("MASK_PROMPT_REWRITE_MAX_OUTPUT_TOKENS must be at least 1000")
    if not 1 <= MASK_PROMPT_REWRITE_MAX_ATTEMPTS <= 10:
        raise ValueError("MASK_PROMPT_REWRITE_MAX_ATTEMPTS must lie in [1, 10]")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", MASK_PROMPT_CACHE_SUBDIRECTORY):
        raise ValueError(
            "MASK_PROMPT_CACHE_SUBDIRECTORY may contain letters, numbers, underscores, and hyphens"
        )
    if MASK_PROMPT_DISPLAY_MAX_SIDE < 128:
        raise ValueError("MASK_PROMPT_DISPLAY_MAX_SIDE must be at least 128")
    if not 0 < MASK_PROMPT_GEOMETRY_THRESHOLD < 1:
        raise ValueError("MASK_PROMPT_GEOMETRY_THRESHOLD must lie in (0, 1)")
    if not 0 <= MASK_PROMPT_MIN_COMPONENT_FRACTION < 0.1:
        raise ValueError(
            "MASK_PROMPT_MIN_COMPONENT_FRACTION must lie in [0, 0.1)"
        )
    if not 1 <= MASK_PROMPT_MAX_COMPONENTS <= 50:
        raise ValueError("MASK_PROMPT_MAX_COMPONENTS must lie in [1, 50]")
    if not (
        isinstance(FLUX_PROMPT_MAX_SEQUENCE_LENGTH, int)
        and 32 <= FLUX_PROMPT_MAX_SEQUENCE_LENGTH <= 512
    ):
        raise ValueError(
            "FLUX_PROMPT_MAX_SEQUENCE_LENGTH must be an integer in [32, 512]"
        )

    def trajectory_generation_prompt(prompt):
        # Compatibility helper only: no local stylistic rewrite is performed.
        return prompt
    """
).lstrip()
validation = replace_between(
    validation,
    "if not 0 <= TRAJECTORY_INIT_DETAIL_STRENGTH <= 1:\n",
    "if OPENAI_IMAGE_DETAIL not in",
    mask_validation,
)
validation = validation.replace(
    "TRAJECTORY_ZIP_FILENAME",
    "MASK_ZIP_FILENAME",
).replace(
    "TRAJECTORY_FRAME_OFFSET",
    "MASK_FRAME_OFFSET",
)
for active_check in (
    """if FLOWMORPH_SOURCE_OPTIMIZATION_STEPS != 100:
    raise ValueError("This quality-first notebook requires 100 endpoint fitting steps")""",
    """if FLOWMORPH_FIT_LORA_SCALE != IMAGE_LORA_SCALE:
    raise ValueError("FlowMorph fit LoRA scale must match IMAGE_LORA_SCALE")""",
    """if FLOWMORPH_RENDER_LORA_SCALE != IMAGE_LORA_SCALE:
    raise ValueError("FlowMorph render LoRA scale must match IMAGE_LORA_SCALE")""",
    """if FLOWMORPH_GUIDANCE_SCALE != IMAGE_GUIDANCE_SCALE:
    raise ValueError("FlowMorph guidance must match IMAGE_GUIDANCE_SCALE")""",
):
    validation = validation.replace(
        active_check,
        "\n".join(f"# {line}" for line in active_check.splitlines()),
        1,
    )
validation_cell["source"] = validation.splitlines(keepends=True)

staging_cell = find_cell(notebook, "TRAJECTORY_DRIVE_ARCHIVE")
staging = source(staging_cell)
for old_name, new_name in {
    "TRAJECTORY_ZIP_DIRECTORY": "MASK_ZIP_DIRECTORY",
    "TRAJECTORY_ZIP_FILENAME": "MASK_ZIP_FILENAME",
    "TRAJECTORY_MEMBER_PREFIX": "MASK_MEMBER_PREFIX",
    "TRAJECTORY_FRAME_OFFSET": "MASK_FRAME_OFFSET",
    "TRAJECTORY_REVERSE_ORDER": "MASK_REVERSE_ORDER",
}.items():
    staging = staging.replace(old_name, new_name)
staging = staging.replace("trajectory ZIP", "grayscale-mask ZIP").replace(
    "Trajectory ZIP",
    "Grayscale-mask ZIP",
)
staging_cell["source"] = staging.splitlines(keepends=True)

model_cell = find_cell(notebook, "if RUN_TRIAL_KEYFRAME:")
model = source(model_cell)
pipeline_setup = model[
    model.index("try:\n    import peft.tuners.lora.torchao")
    : model.index("if RUN_TRIAL_KEYFRAME:")
]
pipeline_setup = pipeline_setup.replace(
    'if "FLUX_PIPE" in globals() and globals().get("FLUX_PIPE_LORA_SCALE") != float(IMAGE_LORA_SCALE):\n',
    """if (
    "FLUX_PIPE" in globals()
    and type(globals()["FLUX_PIPE"]).__name__ != "Flux2KleinPipeline"
):
    print("Anchor backend changed to the standard generation pipeline; rebuilding.")
    release_flux_pipeline()
if "FLUX_PIPE" in globals() and globals().get("FLUX_PIPE_LORA_SCALE") != float(IMAGE_LORA_SCALE):
""",
)
pipeline_setup = pipeline_setup.replace(
    'else:\n    print("Reusing the fused pipeline at the current LoRA scale.")\n',
    """else:
    print("Reusing the fused standard generation pipeline at the current LoRA scale.")
if type(FLUX_PIPE).__name__ != "Flux2KleinPipeline":
    raise RuntimeError("Masked anchors require the standard Flux2KleinPipeline")
""",
)
pipeline_setup += """
# Retain the lightweight tokenizer after releasing the GPU-heavy generation pipeline.
# Prompt validation must remain available during later OpenAI/FlowMorph rounds.
FLUX_PROMPT_TOKENIZER = FLUX_PIPE.tokenizer
"""
model_cell["source"] = code_blocks(
    """
    import base64
    import gc
    import hashlib
    import io
    import os
    import random
    import shutil
    import time
    import numpy as np
    from scipy import ndimage
    from huggingface_hub import hf_hub_download
    from IPython.display import Markdown, display
    from PIL import Image, ImageFilter
    from pydantic import BaseModel, Field, ValidationError
    from flowmorph_klein.art_loop import make_soft_reference
    from flowmorph_klein.lora import load_flux2_lora
    from flowmorph_klein.trajectory import (
        prepare_flux2_klein_img2img_inputs,
        prepare_grayscale_edit_mask,
    )

    def build_background_mask(mask_source):
        result = prepare_grayscale_edit_mask(
            mask_source,
            invert=MASK_INVERT,
            gamma=MASK_GAMMA,
            expansion_radius=MASK_EXPANSION,
            feather_radius=MASK_FEATHER,
        )
        if not (
            MASK_MIN_EDITABLE_FRACTION
            <= result.editable_fraction
            <= MASK_MAX_EDITABLE_FRACTION
        ):
            raise RuntimeError(
                f"Editable mask coverage {result.editable_fraction:.1%} is outside "
                f"{MASK_MIN_EDITABLE_FRACTION:.1%}–{MASK_MAX_EDITABLE_FRACTION:.1%}. "
                "Adjust the mask file, MASK_INVERT, or the editable-fraction limits. "
                "White is editable; black is protected."
            )
        return result

    def smooth_mask_for_initialization(effective_mask):
        return effective_mask.convert("L").filter(
            ImageFilter.GaussianBlur(radius=MASK_INIT_GAUSSIAN_BLUR)
        )

    def build_mask_noise_initialization(mask_result, seed, previous=None):
        soft_mask = smooth_mask_for_initialization(mask_result.mask)
        width, height = soft_mask.size
        random_generator = np.random.default_rng(seed)
        noise_values = random_generator.integers(
            MASK_INIT_NOISE_LOW,
            MASK_INIT_NOISE_HIGH + 1,
            size=(height, width, 3),
            dtype=np.uint8,
        )
        noise_image = Image.fromarray(noise_values, mode="RGB")
        background = Image.new(
            "RGB",
            (width, height),
            tuple(MASK_INIT_BACKGROUND_RGB),
        )
        beige_tinted_noise = Image.blend(
            noise_image,
            background,
            MASK_INIT_NOISE_BEIGE_MIX,
        )
        faintly_noisy_background = Image.blend(
            background,
            noise_image,
            MASK_INIT_BACKGROUND_NOISE_MIX,
        )
        mask_noise_init = Image.composite(
            beige_tinted_noise,
            faintly_noisy_background,
            soft_mask,
        )
        previous_reference = None
        if previous is not None and PREVIOUS_INIT_ENABLED:
            previous_reference = make_soft_reference(
                previous,
                reference_blend=PREVIOUS_INIT_BLEND,
                blur_radius=PREVIOUS_INIT_BLUR,
                grain_strength=PREVIOUS_INIT_GRAIN_STRENGTH,
                grain_seed=seed,
                background_rgb=PREVIOUS_INIT_BACKGROUND_RGB,
            )
            active_previous_init = Image.blend(
                mask_noise_init,
                previous_reference,
                MASK_INIT_PREVIOUS_ACTIVE_MIX,
            )
            quiet_previous_init = Image.blend(
                mask_noise_init,
                previous_reference,
                MASK_INIT_PREVIOUS_QUIET_MIX,
            )
            generation_init = Image.composite(
                active_previous_init,
                quiet_previous_init,
                soft_mask,
            )
            active_previous_init.close()
            quiet_previous_init.close()
        else:
            generation_init = mask_noise_init.copy()
        noise_image.close()
        background.close()
        beige_tinted_noise.close()
        faintly_noisy_background.close()
        mask_noise_init.close()
        return generation_init, soft_mask, previous_reference

    def flux_prompt_token_count(prompt):
        tokenizer = globals().get("FLUX_PROMPT_TOKENIZER")
        if tokenizer is None:
            pipeline = globals().get("FLUX_PIPE")
            tokenizer = getattr(pipeline, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "The FLUX tokenizer is unavailable for prompt-length validation"
            )
        messages = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer(
            templated,
            padding=False,
            truncation=False,
        )["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)

    def validate_flux_prompt_length(prompt, label="Prompt"):
        token_count = flux_prompt_token_count(prompt)
        if token_count > FLUX_PROMPT_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"{label} tokenizes to {token_count} tokens after the FLUX chat "
                f"template; maximum is {FLUX_PROMPT_MAX_SEQUENCE_LENGTH}"
            )
        return token_count

    class MaskPromptPlan(BaseModel):
        spatial_analysis: str = Field(min_length=80, max_length=1800)
        object_layout: str = Field(min_length=120, max_length=2200)
        rewritten_prompt: str = Field(min_length=300, max_length=6000)

    MASK_PROMPT_SYSTEM_PROMPT = f'''
    Role: You are a meticulous art director rewriting one FLUX.2 Klein prompt for
    the {LORA_TRIGGER} oil-painting LoRA.

    Input: You receive (1) a science/theme, (2) the complete original art prompt,
    and (3) the Gaussian-smoothed activity geometry used to initialize FLUX. Bright
    regions receive beige-tinted RGB texture, dark regions receive a quiet beige
    field with faint residual texture, and gray regions softly mix the two. This
    is a compositional tendency, not a hard stencil: FLUX is free to repaint and
    cross every boundary.

    Task: Adapt the complete art prompt in rich visual detail so FLUX composes
    the subject organically around the active geometry. Inspect the geometry itself.
    Translate its actual lobes, islands,
    corridors, bends, diagonals, voids, edge contacts, relative areas, and visual
    center of gravity into a concrete composition.

    Required planning:
    - Assign every important object from the original prompt to a plausible broad
      lobe or island, using explicit canvas locations, relative sizes, orientation,
      overlap, and visual hierarchy.
    - Put recognizable focal objects in broad active spaces, but describe soft
      overlap and natural spill across boundaries rather than hard clipped edges.
    - Use flexible matter—vines, acanthus, stems, ribbons, smoke, clouds, roots,
      cloth, tendrils, chains, wave motifs, and shadows—to travel through narrow
      corridors, join neighboring masses, curl around internal voids, and taper
      naturally before boundaries.
    - Treat isolated active islands as deliberate satellite ornaments related to
      the main subject. Let quiet gaps remain calmer without making them blank.
    - Where the bright geometry meets a canvas edge, describe a natural entrance
      or exit such as a cropped garland, branch, drapery fold, or shadow—not a
      severed focal object.
    - Preserve the science, named objects, palette, aged-plaster fresco character,
      Baroque/Renaissance-grottesche language, oil materiality, and every useful
      stylistic fact from the original. You may change placement, scale, grouping,
      orientation, and which flexible element connects objects.
    - Make the result dense enough to occupy the described composition, with large
      coherent masses and material texture rather than tiny scattered icons.

    Rewritten-prompt contract:
    - It begins exactly with "{LORA_TRIGGER}," and contains that trigger exactly once.
    - It is a self-contained literal description of the final artwork, normally
      220–320 words. Repeat all desired content and style; do not refer back to
      the input.
    - After FLUX's Qwen chat template is applied, it must fit within
      {FLUX_PROMPT_MAX_SEQUENCE_LENGTH} tokens. Prefer concise concrete description
      over repetition.
    - It must not mention a mask, white/black areas, editable/protected pixels,
      alpha, compositing, cutouts, source material, prompt rewriting, or instructions
      to an editor. Convert all of that into ordinary spatial art direction.
    - Do not say "keep", "preserve", "place into the mask", or "fit the mask".

    Output fields:
    - spatial_analysis: a precise prose reading of the supplied geometry.
    - object_layout: a detailed mapping from the original objects to that geometry.
    - rewritten_prompt: only the final self-contained FLUX prompt.
    '''.strip()

    MASK_PROMPT_FORBIDDEN_TERMS = (
        "mask",
        "white area",
        "white region",
        "black area",
        "black region",
        "editable",
        "protected pixels",
        "alpha",
        "composite",
        "compositing",
        "cutout",
        "source prompt",
        "rewrite",
        "rewritten",
        "fit the mask",
        "place into the mask",
        "keep",
        "preserve",
        "retain",
    )
    MASK_PROMPT_MEMORY = {}

    def effective_mask_data_url(mask):
        image = mask.convert("L")
        image.thumbnail((VISION_IMAGE_MAX_SIDE, VISION_IMAGE_MAX_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def effective_mask_sha256(mask):
        buffer = io.BytesIO()
        mask.convert("L").save(buffer, format="PNG", optimize=True)
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    def measure_effective_mask_geometry(mask):
        values = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        height, width = values.shape
        binary = values >= MASK_PROMPT_GEOMETRY_THRESHOLD
        labels, component_count = ndimage.label(
            binary,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        minimum_pixels = max(
            1,
            int(round(
                MASK_PROMPT_MIN_COMPONENT_FRACTION * width * height
            )),
        )
        components = []
        for label_index, component_slice in enumerate(
            ndimage.find_objects(labels),
            start=1,
        ):
            if component_slice is None:
                continue
            component = labels[component_slice] == label_index
            area_pixels = int(component.sum())
            if area_pixels < minimum_pixels:
                continue
            y_slice, x_slice = component_slice
            y0, y1 = y_slice.start, y_slice.stop
            x0, x1 = x_slice.start, x_slice.stop
            local_y, local_x = np.nonzero(component)
            centroid_x = (x0 + float(local_x.mean())) / width
            centroid_y = (y0 + float(local_y.mean())) / height
            edge_contacts = []
            if y0 == 0:
                edge_contacts.append("top")
            if y1 == height:
                edge_contacts.append("bottom")
            if x0 == 0:
                edge_contacts.append("left")
            if x1 == width:
                edge_contacts.append("right")
            component_values = values[component_slice][component]
            components.append({
                "area_fraction": round(area_pixels / (width * height), 4),
                "bbox_normalized": {
                    "left": round(x0 / width, 4),
                    "top": round(y0 / height, 4),
                    "right": round(x1 / width, 4),
                    "bottom": round(y1 / height, 4),
                },
                "centroid_normalized": {
                    "x": round(centroid_x, 4),
                    "y": round(centroid_y, 4),
                },
                "mean_editability": round(
                    float(component_values.mean()),
                    4,
                ),
                "edge_contacts": edge_contacts,
            })
        components.sort(
            key=lambda item: item["area_fraction"],
            reverse=True,
        )
        kept_components = components[:MASK_PROMPT_MAX_COMPONENTS]
        weight_total = float(values.sum())
        if weight_total > 0:
            y_grid, x_grid = np.indices(values.shape, dtype=np.float32)
            weighted_centroid = {
                "x": round(float((x_grid * values).sum() / weight_total) / width, 4),
                "y": round(float((y_grid * values).sum() / weight_total) / height, 4),
            }
        else:
            weighted_centroid = None
        return {
            "coordinate_system": (
                "normalized canvas coordinates: x=0 left, x=1 right, "
                "y=0 top, y=1 bottom"
            ),
            "width": width,
            "height": height,
            "threshold": MASK_PROMPT_GEOMETRY_THRESHOLD,
            "bright_coverage_fraction": round(float(binary.mean()), 4),
            "mean_editability": round(float(values.mean()), 4),
            "gray_transition_fraction": round(
                float(((values > 0) & (values < 1)).mean()),
                4,
            ),
            "weighted_centroid_normalized": weighted_centroid,
            "component_count_before_size_filter": int(component_count),
            "component_count_after_size_filter": len(components),
            "component_count_reported": len(kept_components),
            "components_largest_first": kept_components,
        }

    def mask_prompt_usage(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return (
            usage.model_dump(mode="json")
            if hasattr(usage, "model_dump")
            else str(usage)
        )

    def extract_mask_prompt_plan(response):
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
            raise RuntimeError(
                "OpenAI refused the mask-aware prompt request: "
                + " | ".join(refusals)
            )
        raise RuntimeError("OpenAI returned no parsed mask-aware prompt")

    def validate_mask_rewritten_prompt(prompt):
        clean = " ".join(prompt.split())
        if not clean.startswith(f"{LORA_TRIGGER},"):
            raise ValueError(
                f"Rewritten prompt must begin exactly with {LORA_TRIGGER},"
            )
        if clean.casefold().count(LORA_TRIGGER.casefold()) != 1:
            raise ValueError("Rewritten prompt must contain the LoRA trigger once")
        if len(clean) < 900:
            raise ValueError(
                "Rewritten prompt is too brief for geometry-specific art direction"
            )
        found = [
            term
            for term in MASK_PROMPT_FORBIDDEN_TERMS
            if re.search(rf"\\b{re.escape(term)}\\b", clean, re.IGNORECASE)
        ]
        if found:
            raise ValueError(
                "Rewritten prompt exposes production language: "
                + ", ".join(found)
            )
        validate_flux_prompt_length(clean, "Rewritten prompt")
        return clean

    def mask_prompt_request_fingerprint(stage, effective_mask, base_prompt):
        geometry = measure_effective_mask_geometry(effective_mask)
        contract = {
            "model": OPENAI_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "image_detail": MASK_PROMPT_REWRITE_IMAGE_DETAIL,
            "system_prompt": MASK_PROMPT_SYSTEM_PROMPT,
            "science": stage["science"],
            "original_prompt": stage["prompt"],
            "prepared_base_prompt": base_prompt,
            "effective_mask_sha256": effective_mask_sha256(effective_mask),
            "deterministic_geometry": geometry,
        }
        serialized = json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest(), contract

    def request_mask_prompt_plan(stage, effective_mask, base_prompt):
        geometry = measure_effective_mask_geometry(effective_mask)
        request_text = f'''
        Science/theme:
        {stage["science"]}

        Complete original prompt:
        {base_prompt}

        Deterministic measurements of the smoothed initialization activity:
        {json.dumps(geometry, indent=2, ensure_ascii=False)}

        Inspect the attached smoothed grayscale activity map. Write a detailed
        spatial analysis, map the original objects around its active shapes and
        quiet intervals without hard clipping, then return a complete
        self-contained adapted generation prompt.
        '''.strip()
        correction = ""
        last_error = None
        for attempt in range(1, MASK_PROMPT_REWRITE_MAX_ATTEMPTS + 1):
            try:
                response = OPENAI_CLIENT.responses.parse(
                    model=OPENAI_MODEL,
                    reasoning={"effort": OPENAI_REASONING_EFFORT},
                    store=False,
                    max_output_tokens=MASK_PROMPT_REWRITE_MAX_OUTPUT_TOKENS,
                    input=[
                        {
                            "role": "system",
                            "content": MASK_PROMPT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": request_text + correction,
                                },
                                {
                                    "type": "input_image",
                                    "image_url": effective_mask_data_url(
                                        effective_mask
                                    ),
                                    "detail": MASK_PROMPT_REWRITE_IMAGE_DETAIL,
                                },
                            ],
                        },
                    ],
                    text_format=MaskPromptPlan,
                )
                plan = extract_mask_prompt_plan(response)
                plan.rewritten_prompt = validate_mask_rewritten_prompt(
                    plan.rewritten_prompt
                )
                return plan, response
            except (ValidationError, json.JSONDecodeError, ValueError) as error:
                last_error = error
                correction = (
                    "\\n\\nThe previous response failed validation: "
                    f"{error}. Return complete valid structured output, make the "
                    "rewritten prompt detailed but more concise, keep its complete "
                    f"chat-templated length below {FLUX_PROMPT_MAX_SEQUENCE_LENGTH} "
                    f"tokens, begin it with {LORA_TRIGGER}, and remove all "
                    "production-language terms."
                )
                if attempt < MASK_PROMPT_REWRITE_MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise RuntimeError(
                    "Mask-aware prompt rewriting failed after "
                    f"{attempt} attempts: {last_error}"
                ) from error
        raise RuntimeError(f"Mask-aware prompt rewriting failed: {last_error}")

    def resolve_mask_aware_prompt(stage, effective_mask, stage_index):
        # Deliberately pass the user's prompt byte-for-byte unchanged. All
        # substantive adaptation belongs to the remote model.
        base_prompt = stage["prompt"]
        fingerprint, request_contract = mask_prompt_request_fingerprint(
            stage,
            effective_mask,
            base_prompt,
        )
        if fingerprint in MASK_PROMPT_MEMORY:
            return MASK_PROMPT_MEMORY[fingerprint]

        run_plan_directory = RUN_DIRECTORY / "metadata" / "mask_prompt_plans"
        run_plan_directory.mkdir(parents=True, exist_ok=True)
        persistent_cache_directory = (
            Path(drive_base)
            / PROJECT_NAME
            / MASK_PROMPT_CACHE_SUBDIRECTORY
        )
        persistent_cache_directory.mkdir(parents=True, exist_ok=True)
        cache_path = (
            persistent_cache_directory
            / f"{stage['id']}_{fingerprint[:20]}.json"
        )
        run_plan_path = (
            run_plan_directory
            / f"{stage_index:03d}_{stage['id']}.json"
        )

        cache_hit = False
        response_id = None
        usage = None
        if not MASK_PROMPT_REWRITE_ENABLED:
            plan = MaskPromptPlan(
                spatial_analysis=(
                    "Mask-aware rewriting is disabled by settings, so no remote "
                    "spatial analysis of the effective reveal geometry is used."
                ),
                object_layout=(
                    "The original prepared prompt is passed through without a "
                    "remote spatial rewrite. Object locations therefore follow "
                    "the original prompt rather than a geometry-specific plan."
                ),
                rewritten_prompt=base_prompt,
            )
        elif (
            MASK_PROMPT_REWRITE_REUSE_CACHE
            and not MASK_PROMPT_REWRITE_FORCE_REFRESH
            and cache_path.is_file()
        ):
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") != fingerprint:
                    raise ValueError("fingerprint mismatch")
                plan = MaskPromptPlan.model_validate(cached["plan"])
                plan.rewritten_prompt = validate_mask_rewritten_prompt(
                    plan.rewritten_prompt
                )
                response_id = cached.get("openai_response_id")
                usage = cached.get("usage")
                cache_hit = True
            except Exception as error:
                print(
                    f"Ignoring invalid mask-prompt cache for {stage['id']}: "
                    f"{error}"
                )
                plan, response = request_mask_prompt_plan(
                    stage,
                    effective_mask,
                    base_prompt,
                )
                response_id = response.id
                usage = mask_prompt_usage(response)
        else:
            plan, response = request_mask_prompt_plan(
                stage,
                effective_mask,
                base_prompt,
            )
            response_id = response.id
            usage = mask_prompt_usage(response)

        prompt_token_count = validate_flux_prompt_length(
            plan.rewritten_prompt,
            "Mask-adapted prompt",
        )
        payload = {
            "fingerprint": fingerprint,
            "stage_index": stage_index,
            "stage_id": stage["id"],
            "science": stage["science"],
            "enabled": MASK_PROMPT_REWRITE_ENABLED,
            "cache_hit": cache_hit,
            "request_contract": request_contract,
            "plan": plan.model_dump(mode="json"),
            "flux_prompt_token_count": prompt_token_count,
            "flux_prompt_max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
            "openai_model": OPENAI_MODEL,
            "openai_response_id": response_id,
            "usage": usage,
            "effective_mask_image_stored_in_manifest": False,
        }
        if MASK_PROMPT_REWRITE_ENABLED and not cache_hit:
            cache_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\\n",
                encoding="utf-8",
            )
        run_plan_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\\n",
            encoding="utf-8",
        )
        result = (plan.rewritten_prompt, payload)
        MASK_PROMPT_MEMORY[fingerprint] = result
        print(
            f"Mask-aware prompt {stage_index + 1}/{len(ACTIVE_BASE_STAGES)} "
            f"ready for {stage['id']} "
            f"({'cache' if cache_hit else 'new rewrite'})."
        )
        if MASK_PROMPT_REWRITE_DISPLAY:
            display(Markdown(f"#### Mask-aware plan — {stage['id']}"))
            if MASK_PROMPT_REWRITE_DISPLAY_MASK:
                mask_preview = effective_mask.convert("L")
                mask_preview.thumbnail(
                    (MASK_PROMPT_DISPLAY_MAX_SIDE, MASK_PROMPT_DISPLAY_MAX_SIDE)
                )
                display(mask_preview)
                mask_preview.close()
            display(Markdown(
                "**Remote spatial analysis**\\n\\n"
                f"{plan.spatial_analysis}\\n\\n"
                "**Remote object layout**\\n\\n"
                f"{plan.object_layout}\\n\\n"
                "**Adapted FLUX prompt**\\n\\n"
                f"{plan.rewritten_prompt}"
            ))
        return result

    def generate_mask_initialized_anchor(prompt, seed, init_image):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        generation_inputs = prepare_flux2_klein_img2img_inputs(
            FLUX_PIPE,
            init_image,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=IMAGE_INFERENCE_STEPS,
            strength=MASK_INIT_DENOISE_STRENGTH,
            generator=generator,
        )
        result = FLUX_PIPE(
            prompt=prompt,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=IMAGE_INFERENCE_STEPS,
            sigmas=list(generation_inputs.sigmas),
            latents=generation_inputs.latents,
            guidance_scale=IMAGE_GUIDANCE_SCALE,
            generator=generator,
            output_type="pil",
            max_sequence_length=FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
        )
        if not result.images:
            raise RuntimeError("FLUX returned no anchor image")
        image = result.images[0].convert("RGB")
        generation_report = {
            "backend": type(FLUX_PIPE).__name__,
            "mode": "latent_img2img_from_soft_mask_noise_init",
            "mask_used_as_direct_pipeline_argument": False,
            "mask_derived_init_used_by_model": True,
            "post_decode_mask_application": "none",
            "requested_img2img_strength": generation_inputs.requested_strength,
            "effective_start_sigma": generation_inputs.effective_start_sigma,
            "effective_denoising_steps": generation_inputs.denoising_steps,
        }
        return image, generation_report

    """,
    pipeline_setup,
    """
    if RUN_TRIAL_KEYFRAME:
        system_random = random.SystemRandom()
        trial_index = (
            TRIAL_KEYFRAME_INDEX
            if TRIAL_KEYFRAME_INDEX is not None
            else system_random.randrange(len(ACTIVE_BASE_STAGES))
        )
        if not 0 <= trial_index < len(ACTIVE_BASE_STAGES):
            raise IndexError("TRIAL_KEYFRAME_INDEX is outside the active anchor range")
        trial_seed = TRIAL_SEED if TRIAL_SEED is not None else system_random.randrange(2**31)
        trial_stage = ACTIVE_BASE_STAGES[trial_index]
        trial_trajectory = TRAJECTORY_RECORDS[trial_index]
        with Image.open(trial_trajectory["path"]) as opened:
            trial_mask_source = opened.convert("L")
        trial_mask = build_background_mask(trial_mask_source)
        (
            trial_init,
            trial_soft_mask,
            trial_previous_reference,
        ) = build_mask_noise_initialization(
            trial_mask,
            trial_seed,
        )
        if trial_previous_reference is not None:
            raise RuntimeError("A standalone trial must not use a previous reference")
        (
            trial_generation_prompt,
            trial_prompt_plan_report,
        ) = resolve_mask_aware_prompt(
            trial_stage,
            trial_soft_mask,
            trial_index,
        )
        trial_image, trial_generation_report = generate_mask_initialized_anchor(
            trial_generation_prompt,
            trial_seed,
            trial_init,
        )
        trial_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        trial_directory = (
            RUN_DIRECTORY / "trials" / f"{trial_stamp}_{trial_stage['id']}_{trial_seed}"
        )
        trial_directory.mkdir(parents=True, exist_ok=False)
        trial_path = trial_directory / "trial_soft_mask_initialized.png"
        trial_init_path = trial_directory / "mask_noise_init.png"
        trial_source_path = trial_directory / "grayscale_mask_source.png"
        trial_mask_path = trial_directory / "effective_activity_mask.png"
        trial_soft_mask_path = trial_directory / "gaussian_smoothed_activity_mask.png"
        trial_image.save(trial_path)
        trial_init.save(trial_init_path)
        trial_mask_source.save(trial_source_path)
        trial_mask.mask.save(trial_mask_path)
        trial_soft_mask.save(trial_soft_mask_path)
        (trial_directory / "settings.json").write_text(json.dumps({
            "stage": trial_stage,
            "trajectory": trial_trajectory,
            "seed": trial_seed,
            "mask_init_background_rgb": list(MASK_INIT_BACKGROUND_RGB),
            "mask_init_gaussian_blur": MASK_INIT_GAUSSIAN_BLUR,
            "mask_init_noise_low": MASK_INIT_NOISE_LOW,
            "mask_init_noise_high": MASK_INIT_NOISE_HIGH,
            "mask_init_noise_beige_mix": MASK_INIT_NOISE_BEIGE_MIX,
            "mask_init_background_noise_mix": MASK_INIT_BACKGROUND_NOISE_MIX,
            "mask_init_denoise_strength": MASK_INIT_DENOISE_STRENGTH,
            "mask_invert": MASK_INVERT,
            "mask_gamma": MASK_GAMMA,
            "mask_expansion": MASK_EXPANSION,
            "mask_feather": MASK_FEATHER,
            "editable_fraction": trial_mask.editable_fraction,
            "mask_polarity": (
                "bright_beige_tinted_noise_activity_"
                "dark_faintly_noisy_beige_quiet"
            ),
            "mask_values_preserved_without_binarization": True,
            "generation_backend": trial_generation_report["backend"],
            "generation_mode": trial_generation_report["mode"],
            "mask_derived_init_used_by_model": (
                trial_generation_report["mask_derived_init_used_by_model"]
            ),
            "post_decode_mask_application": (
                trial_generation_report["post_decode_mask_application"]
            ),
            "previous_init_used": False,
            "source_used_as_direct_latent_init": False,
            "mask_noise_image_used_as_img2img_init": True,
            "effective_denoising_steps": (
                trial_generation_report["effective_denoising_steps"]
            ),
            "mask_prompt_rewrite": trial_prompt_plan_report,
            "generation_prompt": trial_generation_prompt,
            "generation_prompt_token_count": trial_prompt_plan_report[
                "flux_prompt_token_count"
            ],
            "flux_prompt_max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
        }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        previews = [
            ("Loaded grayscale mask source", trial_mask_source.convert("RGB")),
            ("Effective activity mask before Gaussian smoothing", trial_mask.mask.convert("RGB")),
            ("Gaussian-smoothed activity mask", trial_soft_mask.convert("RGB")),
            (
                "Actual blended beige + tinted-noise img2img initialization",
                trial_init.copy(),
            ),
            ("Direct FLUX result — no post-generation mask", trial_image.copy()),
        ]
        for heading, preview in previews:
            preview.thumbnail((TRIAL_DISPLAY_MAX_WIDTH, TRIAL_DISPLAY_MAX_WIDTH))
            display(Markdown(f"### {heading}"))
            display(preview)
            preview.close()
        print({
            "path": str(trial_path),
            "init_path": str(trial_init_path),
            "source_path": str(trial_source_path),
            "mask_path": str(trial_mask_path),
            "soft_mask_path": str(trial_soft_mask_path),
            "editable_fraction": trial_mask.editable_fraction,
            "seed": trial_seed,
            "prompt_index": trial_index,
        })
        del (
            trial_image,
            trial_init,
            trial_mask_source,
            trial_mask,
            trial_soft_mask,
            trial_previous_reference,
            trial_prompt_plan_report,
            trial_generation_report,
        )
    else:
        print("Trial skipped.")
    """
)

anchor_markdown = find_cell(notebook, "## 7. Generate trajectory-conditioned")
anchor_markdown["source"] = lines(
    """
    ## 7. Generate soft-mask-initialized cyclic anchor paintings

    Each selected grayscale frame is Gaussian-smoothed and used to blend a faintly noisy
    beige field with deterministic RGB noise tinted toward the same beige. The remote
    vision model adapts the prompt around the same soft activity geometry while retaining
    the LoRA trigger exactly once.
    Every anchor is generated by img2img from that initialization; anchors 2+ also mix
    in the weak blurred/grained previous image, continuously weighted by the same
    Gaussian-smoothed activity mask. The decoded result is saved directly:
    there is no exact output mask, alpha composite, or post-generation clipping.
    """
)

anchor_cell = find_cell(notebook, "BASE_MANIFEST_PATH")
anchor_cell["source"] = lines(
    """
    BASE_DIRECTORY = RUN_DIRECTORY / "base_frames"
    MASK_DIRECTORY = BASE_DIRECTORY / "effective_activity_masks"
    SOFT_MASK_DIRECTORY = BASE_DIRECTORY / "gaussian_smoothed_activity_masks"
    INIT_DIRECTORY = BASE_DIRECTORY / "mask_noise_inits"
    PREVIOUS_INIT_DIRECTORY = BASE_DIRECTORY / "previous_anchor_references"
    BASE_MANIFEST_PATH = RUN_DIRECTORY / "metadata" / "base_manifest.json"
    BASE_RECORDS = []
    MASK_CONTRACT = {
        "source": "gaussian_smoothed_grayscale_mask_noise_initialization",
        "mask_init_background_rgb": list(MASK_INIT_BACKGROUND_RGB),
        "mask_init_gaussian_blur": MASK_INIT_GAUSSIAN_BLUR,
        "mask_init_noise_low": MASK_INIT_NOISE_LOW,
        "mask_init_noise_high": MASK_INIT_NOISE_HIGH,
        "mask_init_noise_beige_mix": MASK_INIT_NOISE_BEIGE_MIX,
        "mask_init_background_noise_mix": MASK_INIT_BACKGROUND_NOISE_MIX,
        "mask_init_denoise_strength": MASK_INIT_DENOISE_STRENGTH,
        "mask_init_previous_active_mix": MASK_INIT_PREVIOUS_ACTIVE_MIX,
        "mask_init_previous_quiet_mix": MASK_INIT_PREVIOUS_QUIET_MIX,
        "invert": MASK_INVERT,
        "gamma": MASK_GAMMA,
        "expansion": MASK_EXPANSION,
        "feather": MASK_FEATHER,
        "generation_backend": "Flux2KleinPipeline",
        "all_anchor_mode": "latent_img2img_from_soft_mask_noise_init",
        "post_decode_mask_application": "none",
        "mask_used_as_direct_pipeline_argument": False,
        "mask_derived_init_used_by_model": True,
        "mask_used_by_prompt_planner": MASK_PROMPT_REWRITE_ENABLED,
        "mask_prompt_planner_model": (
            OPENAI_MODEL if MASK_PROMPT_REWRITE_ENABLED else None
        ),
        "mask_prompt_rewrite_image_detail": MASK_PROMPT_REWRITE_IMAGE_DETAIL,
        "flux_prompt_max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
        "polarity": (
            "bright_beige_tinted_noise_activity_"
            "dark_faintly_noisy_beige_quiet"
        ),
        "continuous_values_preserved": True,
        "previous_init_enabled": PREVIOUS_INIT_ENABLED,
        "previous_init_blend": PREVIOUS_INIT_BLEND,
        "previous_init_blur": PREVIOUS_INIT_BLUR,
        "previous_init_grain_strength": PREVIOUS_INIT_GRAIN_STRENGTH,
        "previous_init_background_rgb": list(PREVIOUS_INIT_BACKGROUND_RGB),
        "source_mask_used_as_direct_latent_init": False,
        "mask_noise_image_used_as_img2img_init": True,
        "previous_reference_mixed_into_mask_init": True,
        "previous_reference_spatial_weighting": (
            "gaussian_smoothed_activity_mask"
        ),
    }

    MASK_PROMPT_RECORDS = []
    for index, (stage, trajectory) in enumerate(
        zip(ACTIVE_BASE_STAGES, TRAJECTORY_RECORDS, strict=True)
    ):
        with Image.open(trajectory["path"]) as opened:
            planning_mask_source = opened.convert("L")
        planning_mask_result = build_background_mask(planning_mask_source)
        planning_soft_mask = smooth_mask_for_initialization(
            planning_mask_result.mask
        )
        generation_prompt, prompt_plan_report = resolve_mask_aware_prompt(
            stage,
            planning_soft_mask,
            index,
        )
        MASK_PROMPT_RECORDS.append({
            "generation_prompt": generation_prompt,
            "fingerprint": prompt_plan_report["fingerprint"],
            "cache_hit": prompt_plan_report["cache_hit"],
            "plan_path": str(
                RUN_DIRECTORY
                / "metadata"
                / "mask_prompt_plans"
                / f"{index:03d}_{stage['id']}.json"
            ),
        })
        planning_mask_source.close()
        planning_mask_result.mask.close()
        planning_soft_mask.close()

    expected_anchor_contract = [
        {
            "uid": f"base_{index:03d}",
            "science": stage["science"],
            "prompt": stage["prompt"],
            "generation_prompt": prompt_record["generation_prompt"],
            "mask_prompt_fingerprint": prompt_record["fingerprint"],
            "trajectory_member": trajectory["member"],
            "trajectory_member_sha256": trajectory["member_sha256"],
            "mask_contract": MASK_CONTRACT,
        }
        for index, (stage, trajectory, prompt_record) in enumerate(
            zip(
                ACTIVE_BASE_STAGES,
                TRAJECTORY_RECORDS,
                MASK_PROMPT_RECORDS,
                strict=True,
            )
        )
    ]

    if not REGENERATE_BASE_FRAMES and BASE_MANIFEST_PATH.is_file():
        saved_base_manifest = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
        BASE_RECORDS = saved_base_manifest["records"]
        missing = [item["path"] for item in BASE_RECORDS if not Path(item["path"]).is_file()]
        if missing:
            raise FileNotFoundError("Missing resumed anchor images: " + ", ".join(missing))
        resumed_contract = [
            {
                "uid": record["uid"],
                "science": record["science"],
                "prompt": record["prompt"],
                "generation_prompt": record.get("generation_prompt"),
                "mask_prompt_fingerprint": record.get(
                    "mask_prompt_fingerprint"
                ),
                "trajectory_member": record["trajectory_member"],
                "trajectory_member_sha256": record["trajectory_member_sha256"],
                "mask_contract": record.get("mask_contract"),
            }
            for record in BASE_RECORDS
        ]
        if (
            resumed_contract != expected_anchor_contract
            or saved_base_manifest.get("trajectory_archive_sha256")
            != TRAJECTORY_ARCHIVE_SHA256
        ):
            raise RuntimeError(
                "Prompts, mask settings, or archive differ from the saved anchors. "
                "Set REGENERATE_BASE_FRAMES=True or resume the matching run."
            )
        print(f"Loaded {len(BASE_RECORDS)} existing soft-mask-initialized anchors.")
    else:
        MASK_DIRECTORY.mkdir(parents=True, exist_ok=True)
        SOFT_MASK_DIRECTORY.mkdir(parents=True, exist_ok=True)
        if SAVE_MASK_INITS:
            INIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        if SAVE_PREVIOUS_INITS:
            PREVIOUS_INIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        previous = None
        for index, (stage, trajectory, prompt_record) in enumerate(
            zip(
                ACTIVE_BASE_STAGES,
                TRAJECTORY_RECORDS,
                MASK_PROMPT_RECORDS,
                strict=True,
            )
        ):
            seed = BASE_SEED + index
            with Image.open(trajectory["path"]) as opened:
                mask_source = opened.convert("L")
            generation_prompt = prompt_record["generation_prompt"]
            prepared_mask_result = build_background_mask(mask_source)
            generation_init, soft_mask, previous_reference = (
                build_mask_noise_initialization(
                    prepared_mask_result,
                    seed,
                    previous=previous,
                )
            )
            image, generation_report = generate_mask_initialized_anchor(
                generation_prompt,
                seed,
                generation_init,
            )
            mask_path = MASK_DIRECTORY / f"mask_{index:03d}.png"
            soft_mask_path = (
                SOFT_MASK_DIRECTORY / f"soft_mask_{index:03d}.png"
            )
            init_path = INIT_DIRECTORY / f"init_{index:03d}.png"
            previous_reference_path = (
                PREVIOUS_INIT_DIRECTORY / f"previous_{index:03d}.png"
                if previous_reference is not None
                else None
            )
            output_path = BASE_DIRECTORY / f"{index:03d}_{stage['id']}.png"
            prepared_mask_result.mask.save(
                mask_path,
                format="PNG",
                compress_level=4,
            )
            soft_mask.save(
                soft_mask_path,
                format="PNG",
                compress_level=4,
            )
            if SAVE_MASK_INITS:
                generation_init.save(
                    init_path,
                    format="PNG",
                    compress_level=4,
                )
            if previous_reference_path is not None and SAVE_PREVIOUS_INITS:
                previous_reference.save(
                    previous_reference_path,
                    format="PNG",
                    compress_level=4,
                )
            image.save(output_path, format="PNG", compress_level=4)
            record = {
                "uid": f"base_{index:03d}",
                "kind": "base",
                "round": 0,
                "science": stage["science"],
                "prompt": stage["prompt"],
                "generation_prompt": generation_prompt,
                "mask_prompt_fingerprint": prompt_record["fingerprint"],
                "mask_prompt_cache_hit": prompt_record["cache_hit"],
                "mask_prompt_plan_path": prompt_record["plan_path"],
                "generation_prompt_token_count": validate_flux_prompt_length(
                    generation_prompt,
                    "Anchor generation prompt",
                ),
                "seed": seed,
                "path": str(output_path),
                "trajectory_index": trajectory["trajectory_index"],
                "trajectory_member": trajectory["member"],
                "trajectory_member_sha256": trajectory["member_sha256"],
                "trajectory_source_path": trajectory["path"],
                "trajectory_edit_mask_path": str(mask_path),
                "trajectory_soft_activity_mask_path": str(soft_mask_path),
                "mask_noise_init_path": (
                    str(init_path) if SAVE_MASK_INITS else None
                ),
                "previous_reference_path": (
                    str(previous_reference_path)
                    if previous_reference_path is not None
                    and SAVE_PREVIOUS_INITS
                    else None
                ),
                "previous_init_used": previous_reference is not None,
                "editable_fraction": prepared_mask_result.editable_fraction,
                "mask_contract": MASK_CONTRACT,
                "generation_backend": generation_report["backend"],
                "generation_mode": generation_report["mode"],
                "mask_used_as_direct_pipeline_argument": generation_report[
                    "mask_used_as_direct_pipeline_argument"
                ],
                "mask_derived_init_used_by_model": generation_report[
                    "mask_derived_init_used_by_model"
                ],
                "post_decode_mask_application": generation_report[
                    "post_decode_mask_application"
                ],
                "img2img_strength": generation_report["requested_img2img_strength"],
                "effective_start_sigma": generation_report["effective_start_sigma"],
                "effective_denoising_steps": (
                    generation_report["effective_denoising_steps"]
                ),
            }
            BASE_RECORDS.append(record)
            BASE_MANIFEST_PATH.write_text(json.dumps({
                "trajectory_archive_sha256": TRAJECTORY_ARCHIVE_SHA256,
                "trajectory_selection_manifest": str(TRAJECTORY_SELECTION_MANIFEST),
                "records": BASE_RECORDS,
                "complete": len(BASE_RECORDS) == len(ACTIVE_BASE_STAGES),
            }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
            print(
                f"Anchor {index + 1}/{len(ACTIVE_BASE_STAGES)} saved: "
                f"{output_path.name} ← {trajectory['member']} "
                f"({prepared_mask_result.editable_fraction:.1%} activity, "
                f"previous reference={'yes' if previous_reference is not None else 'no'})"
            )
            if previous is not None:
                previous.close()
            previous = image.copy()
            mask_source.close()
            if previous_reference is not None:
                previous_reference.close()
            generation_init.close()
            soft_mask.close()
            image.close()
            prepared_mask_result.mask.close()
            del prepared_mask_result
            del generation_report
        if previous is not None:
            previous.close()
            del previous

    if len(BASE_RECORDS) != len(ACTIVE_BASE_STAGES):
        raise RuntimeError("The anchor manifest is incomplete; regenerate or resume the correct run.")
    print(f"Prepared {len(BASE_RECORDS)} soft-mask-initialized anchors in {BASE_DIRECTORY}")
    """
)

contact_cell = find_cell(notebook, "trajectory_source_paths =")
contact_cell["source"] = lines(
    """
    from flowmorph_klein.visualization import make_contact_sheet

    def load_contact_thumbnails(paths, size=192):
        images = []
        for path in paths:
            with Image.open(path) as opened:
                thumbnail = opened.convert("RGB")
                thumbnail.thumbnail((size, size))
                images.append(thumbnail)
        return images

    paired_contact_sheet_path = (
        RUN_DIRECTORY / "previews" / "mask_init_anchor_audit.png"
    )
    paired_images = []
    paired_labels = []
    for index, record in enumerate(BASE_RECORDS):
        row_images = load_contact_thumbnails([
            Path(record["trajectory_source_path"]),
            Path(record["trajectory_soft_activity_mask_path"]),
        ])
        if record.get("mask_noise_init_path"):
            row_images.extend(
                load_contact_thumbnails([Path(record["mask_noise_init_path"])])
            )
        else:
            row_images.append(
                Image.new("RGB", (192, 192), MASK_INIT_BACKGROUND_RGB)
            )
        if record.get("previous_reference_path"):
            row_images.extend(
                load_contact_thumbnails([Path(record["previous_reference_path"])])
            )
        else:
            row_images.append(
                Image.new("RGB", (192, 192), MASK_INIT_BACKGROUND_RGB)
            )
        row_images.extend(load_contact_thumbnails([Path(record["path"])]))
        paired_images.extend(row_images)
        paired_labels.extend([
            f"{index:02d} grayscale source",
            f"{index:02d} smoothed activity",
            f"{index:02d} mask-noise init",
            f"{index:02d} previous reference",
            f"{index:02d} generated",
        ])
    make_contact_sheet(
        paired_images,
        paired_contact_sheet_path,
        columns=5,
        labels=paired_labels,
    )
    for image in paired_images:
        image.close()

    preview = Image.open(paired_contact_sheet_path).convert("RGB")
    preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
    display(Markdown(
        "### Source → smoothed activity → actual init → previous reference → direct result"
    ))
    display(preview)
    preview.close()
    print({
        "full_resolution_sources": str(
            Path(BASE_RECORDS[0]["trajectory_source_path"]).parent
        ),
        "full_resolution_soft_activity_masks": str(
            Path(BASE_RECORDS[0]["trajectory_soft_activity_mask_path"]).parent
        ),
        "full_resolution_mask_noise_inits": (
            str(INIT_DIRECTORY) if SAVE_MASK_INITS else None
        ),
        "full_resolution_previous_references": (
            str(PREVIOUS_INIT_DIRECTORY) if SAVE_PREVIOUS_INITS else None
        ),
        "full_resolution_anchors": str(BASE_DIRECTORY),
        "paired_audit": str(paired_contact_sheet_path),
    })
    """
)

midpoint_cell = find_cell(notebook, "class MidpointProposal")
midpoint_source = source(midpoint_cell)
midpoint_source = midpoint_source.replace(
    '- The prompt begins exactly with "{LORA_TRIGGER}," and contains that trigger exactly once.\n',
    """- The prompt begins exactly with "{LORA_TRIGGER}," and contains that trigger exactly once.
- After FLUX's Qwen chat template is applied, the prompt must fit within
  {FLUX_PROMPT_MAX_SEQUENCE_LENGTH} tokens. Prefer concise concrete description
  over repetition.
""",
    1,
)
midpoint_source = midpoint_source.replace(
    '        "system_prompt": MIDPOINT_SYSTEM_PROMPT,\n',
    """        "system_prompt": MIDPOINT_SYSTEM_PROMPT,
        "flux_prompt_max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
""",
    1,
)
midpoint_source = midpoint_source.replace(
    """    if found:
        raise ValueError("Prompt contains production-language terms: " + ", ".join(found))
    return clean
""",
    """    if found:
        raise ValueError("Prompt contains production-language terms: " + ", ".join(found))
    validate_flux_prompt_length(clean, "Midpoint prompt")
    return clean
""",
    1,
)
midpoint_source = midpoint_source.replace(
    """                f"\\n\\nThe previous result failed semantic validation: {error}. "
                "Return a newly written literal prompt satisfying every prohibition."
""",
    """                f"\\n\\nThe previous result failed semantic validation: {error}. "
                "Return a newly written literal prompt satisfying every prohibition "
                f"and fitting within {FLUX_PROMPT_MAX_SEQUENCE_LENGTH} chat-templated tokens."
""",
    1,
)
midpoint_cell["source"] = midpoint_source.splitlines(keepends=True)

sequence_cell = find_cell(notebook, "SEQUENCE_SESSION_CONTRACT")
sequence_source = source(sequence_cell)
sequence_source = sequence_source.replace(
    """# The standalone anchor pipeline is fused and CPU-offloaded. Release it;
# the sequence session loads one unfused differentiable model and retains it.
release_flux_pipeline()
""",
    """# Audit every anchor prompt while the canonical tokenizer is available.
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

# The standalone anchor pipeline is fused and CPU-offloaded. Release it;
# the sequence session loads one unfused differentiable model and retains it.
# FLUX_PROMPT_TOKENIZER remains resident for later midpoint validation.
release_flux_pipeline()
""",
    1,
)
sequence_source = sequence_source.replace(
    """        if saved.get("combined_fingerprint") == combined_fingerprint:
            return (
                MidpointProposal.model_validate(saved["proposal"]),
                saved.get("openai_response_id"),
                saved.get("usage"),
                combined_fingerprint,
            )
""",
    """        if saved.get("combined_fingerprint") == combined_fingerprint:
            proposal = MidpointProposal.model_validate(saved["proposal"])
            proposal.prompt = validate_midpoint_prompt(proposal.prompt)
            return (
                proposal,
                saved.get("openai_response_id"),
                saved.get("usage"),
                combined_fingerprint,
            )
""",
    1,
)
sequence_source = sequence_source.replace(
    '        "proposal": proposal.model_dump(mode="json"),\n',
    """        "proposal": proposal.model_dump(mode="json"),
        "flux_prompt_token_count": validate_flux_prompt_length(
            proposal.prompt,
            "Saved midpoint prompt",
        ),
        "flux_prompt_max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
""",
    1,
)
sequence_source = sequence_source.replace(
    '    "conditioning": "piecewise_source_midpoint_target_embeddings",\n',
    """    "conditioning": "piecewise_source_midpoint_target_embeddings",
    "flux_prompt_max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
""",
    1,
)
sequence_source = sequence_source.replace(
    """    if record_prompts or missing_prompts or missing_images:
        new_prompts, new_images = SEQUENCE_SESSION.encode_missing_assets(
""",
    """    for prompt in [*record_prompts, *missing_prompts]:
        validate_flux_prompt_length(prompt, "FlowMorph conditioning prompt")
    if record_prompts or missing_prompts or missing_images:
        new_prompts, new_images = SEQUENCE_SESSION.encode_missing_assets(
""",
    1,
)
sequence_cell["source"] = sequence_source.splitlines(keepends=True)

assembly_markdown = find_cell(notebook, "## 10. Assemble, preview")
assembly_markdown["source"] = lines(
    """
    ## 10. Assemble, preview, and audit the generated cyclic sequence

    The directly decoded, soft-mask-initialized anchors now enter the unchanged recursive
    FlowMorph pipeline. FlowMorph receives ordinary completed anchor paintings; no masks
    are reapplied to its fitted endpoints or interpolated frames.
    Round 1 contributes one explicit midpoint, Round 2 contributes ten shared-prompt
    renders, and the final-to-first gap closes the loop before RIFE finishing.

    If enabled, the reusable temporal-tone helper compares every final frame with its
    cyclic neighbors and gently corrects only robust luminance/contrast outliers. Raw
    FlowMorph PNGs remain untouched, while previews and RIFE use the corrected paths.
    """
)

assembly_cell = find_cell(
    notebook,
    "# A Colab reconnect clears Python variables while completed manifests",
)
assembly_source = source(assembly_cell)
assembly_source = assembly_source.replace(
    "from flowmorph_klein.visualization import make_contact_sheet\n",
    """from flowmorph_klein.temporal_tone import (
    TemporalToneConfig,
    stabilize_cyclic_tone,
)
from flowmorph_klein.visualization import make_contact_sheet
""",
    1,
)
assembly_source = assembly_source.replace(
    """    restored_manifest_candidates.extend([
        RUN_DIRECTORY / "metadata" / "final_recursive_flowmorph_sequence.json",
""",
    """    restored_manifest_candidates.extend([
        RUN_DIRECTORY
        / "metadata"
        / "final_recursive_flowmorph_sequence_tone_stabilized.json",
        RUN_DIRECTORY / "metadata" / "final_recursive_flowmorph_sequence.json",
""",
    1,
)
assembly_source = assembly_source.replace(
    """                *RUN_DIRECTORY.parent.glob(
                    "*/metadata/final_recursive_flowmorph_sequence.json"
                ),
""",
    """                *RUN_DIRECTORY.parent.glob(
                    "*/metadata/final_recursive_flowmorph_sequence_tone_stabilized.json"
                ),
                *RUN_DIRECTORY.parent.glob(
                    "*/metadata/final_recursive_flowmorph_sequence.json"
                ),
""",
    1,
)
tone_stabilization = dedent(
    """
    raw_final_records = []
    for item in FINAL_RECORDS:
        raw_item = dict(item)
        raw_item["path"] = item.get("raw_flowmorph_path", item["path"])
        raw_final_records.append(raw_item)

    if TEMPORAL_TONE_STABILIZATION_ENABLED:
        tone_directory = RUN_DIRECTORY / "temporal_tone_stabilization"
        raw_sequence_manifest = (
            RUN_DIRECTORY / "metadata" / "final_recursive_flowmorph_sequence.json"
        )
        if not raw_sequence_manifest.is_file():
            raw_sequence_manifest = Path(FINAL_SEQUENCE_MANIFEST)
        tone_result = stabilize_cyclic_tone(
            [item["path"] for item in raw_final_records],
            tone_directory / "corrected_frames",
            config=TemporalToneConfig(
                window_radius=TEMPORAL_TONE_WINDOW_RADIUS,
                strength=TEMPORAL_TONE_STRENGTH,
                mean_threshold=TEMPORAL_TONE_MEAN_THRESHOLD,
                contrast_threshold=TEMPORAL_TONE_CONTRAST_THRESHOLD,
                mad_multiplier=TEMPORAL_TONE_MAD_MULTIPLIER,
                max_mean_shift=TEMPORAL_TONE_MAX_MEAN_SHIFT,
                max_contrast_scale_delta=(
                    TEMPORAL_TONE_MAX_CONTRAST_SCALE_DELTA
                ),
                analysis_max_side=TEMPORAL_TONE_ANALYSIS_MAX_SIDE,
            ),
            report_path=tone_directory / "temporal_tone_report.json",
            reuse_existing=TEMPORAL_TONE_REUSE_EXISTING,
        )
        stabilized_records = []
        for item, raw_item, stabilized_path, frame_audit in zip(
            FINAL_RECORDS,
            raw_final_records,
            tone_result.output_paths,
            tone_result.report["frames"],
            strict=True,
        ):
            stabilized = dict(item)
            stabilized["raw_flowmorph_path"] = raw_item["path"]
            stabilized["path"] = str(stabilized_path)
            stabilized["temporal_tone_corrected"] = frame_audit["corrected"]
            stabilized["temporal_tone_report_path"] = str(tone_result.report_path)
            stabilized_records.append(stabilized)
        FINAL_RECORDS = stabilized_records
        tone_sequence_manifest = (
            RUN_DIRECTORY
            / "metadata"
            / "final_recursive_flowmorph_sequence_tone_stabilized.json"
        )
        tone_sequence_manifest.write_text(json.dumps({
            "project": PROJECT_NAME,
            "cyclic": True,
            "source_manifest": str(raw_sequence_manifest),
            "temporal_tone_stabilization_enabled": True,
            "temporal_tone_fingerprint": tone_result.report["fingerprint"],
            "temporal_tone_report": str(tone_result.report_path),
            "corrected_count": tone_result.report["corrected_count"],
            "corrected_indices": tone_result.report["corrected_indices"],
            "cache_hit": tone_result.cache_hit,
            "final_count": len(FINAL_RECORDS),
            "records": FINAL_RECORDS,
        }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        FINAL_SEQUENCE_MANIFEST = tone_sequence_manifest
        print({
            "temporal_tone_stabilization": True,
            "corrected_frames": tone_result.report["corrected_count"],
            "total_frames": len(FINAL_RECORDS),
            "cache_hit": tone_result.cache_hit,
            "raw_frames_preserved": True,
            "report": str(tone_result.report_path),
        })
    else:
        FINAL_RECORDS = raw_final_records
        print("Temporal tone stabilization disabled; using raw FlowMorph frames.")

    """
)
assembly_source = assembly_source.replace(
    'if len(FINAL_RECORDS) < 3:\n',
    tone_stabilization + 'if len(FINAL_RECORDS) < 3:\n',
    1,
)
assembly_cell["source"] = assembly_source.splitlines(keepends=True)

final_video_cell = find_cell(notebook, "recursive_flowmorph_trajectory_rife_ssim_loop.mp4")
final_video_source = source(final_video_cell).replace(
    "recursive_flowmorph_trajectory_rife_ssim_loop.mp4",
    "recursive_flowmorph_background_mask_rife_ssim_loop.mp4",
)
final_video_cell["source"] = final_video_source.splitlines(keepends=True)

notebook["cells"].extend(
    [
        {
            "cell_type": "markdown",
            "id": "background-mask-flicker-diagnosis-heading",
            "metadata": {},
            "source": lines(
                """
                ## 14. Read-only cyclic flicker diagnosis

                This final cell measures the **raw FlowMorph frames** without changing
                them. It compares each frame with its cyclic neighbors, identifies pulse
                centers, and tests whether high scores repeat at a particular midpoint
                position, render-batch slot, whole-gap period, or other spectral period.

                The PNG plot and complete JSON audit are written into the persistent run
                directory. Tone stabilization is not used by this diagnosis.
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "background-mask-flicker-diagnosis",
            "metadata": {},
            "outputs": [],
            "source": lines(
                """
                import json
                from pathlib import Path
                from PIL import Image
                from IPython.display import Markdown, display
                from flowmorph_klein.flicker_diagnostics import (
                    FlickerDiagnosticConfig,
                    diagnose_cyclic_flicker,
                )

                if RUN_FLICKER_DIAGNOSTIC:
                    diagnostic_records = globals().get("FINAL_RECORDS")
                    if diagnostic_records is None:
                        manifest_candidates = [
                            RUN_DIRECTORY
                            / "metadata"
                            / "final_recursive_flowmorph_sequence.json",
                            RUN_DIRECTORY
                            / "metadata"
                            / "final_recursive_flowmorph_sequence_tone_stabilized.json",
                            RUN_DIRECTORY
                            / "metadata"
                            / "final_recursive_sequence.json",
                        ]
                        diagnostic_manifest = next(
                            (path for path in manifest_candidates if path.is_file()),
                            None,
                        )
                        if diagnostic_manifest is None:
                            raise RuntimeError(
                                "No final sequence is available for flicker diagnosis. "
                                "Run the FlowMorph sequence or set RESUME_RUN_DIRECTORY "
                                "to a completed run first."
                            )
                        diagnostic_records = json.loads(
                            diagnostic_manifest.read_text(encoding="utf-8")
                        )["records"]
                        print("Restored diagnostic records from", diagnostic_manifest)

                    final_gap_size = (
                        int(FLOWMORPH_ROUND_SPECS[-1]["midpoint_count"]) + 1
                    )
                    FLICKER_DIAGNOSTIC_RESULT = diagnose_cyclic_flicker(
                        diagnostic_records,
                        RUN_DIRECTORY / "diagnostics" / "flicker",
                        config=FlickerDiagnosticConfig(
                            analysis_max_side=FLICKER_ANALYSIS_MAX_SIDE,
                            outlier_mad_multiplier=(
                                FLICKER_OUTLIER_MAD_MULTIPLIER
                            ),
                            minimum_outlier_score=(
                                FLICKER_MINIMUM_OUTLIER_SCORE
                            ),
                            max_lag=FLICKER_MAX_LAG,
                            gap_size=final_gap_size,
                            render_batch_size=FLOWMORPH_RENDER_BATCH_SIZE,
                        ),
                    )
                    flicker_preview = Image.open(
                        FLICKER_DIAGNOSTIC_RESULT.plot_path
                    ).convert("RGB")
                    flicker_preview.thumbnail(
                        (CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000)
                    )
                    display(Markdown("### Raw FlowMorph flicker pattern diagnosis"))
                    display(flicker_preview)
                    flicker_preview.close()
                    strong_hypotheses = [
                        item
                        for item in FLICKER_DIAGNOSTIC_RESULT.report["hypotheses"]
                        if item.get("support") == "strong"
                    ]
                    print({
                        "raw_frames_analyzed": (
                            FLICKER_DIAGNOSTIC_RESULT.report["frame_count"]
                        ),
                        "pulse_centers": (
                            FLICKER_DIAGNOSTIC_RESULT.report["outlier_indices"]
                        ),
                        "strong_hypotheses": strong_hypotheses,
                        "dominant_periods": (
                            FLICKER_DIAGNOSTIC_RESULT.report["dominant_periods"][:5]
                        ),
                        "report": str(FLICKER_DIAGNOSTIC_RESULT.report_path),
                        "plot": str(FLICKER_DIAGNOSTIC_RESULT.plot_path),
                        "images_modified": False,
                    })
                else:
                    print("Flicker diagnosis disabled.")
                """
            ),
        },
    ]
)

notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} clean cells")
