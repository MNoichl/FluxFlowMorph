"""Build the prompt-only recursive FlowMorph production notebook.

The builder composes the proven prompt-only anchor workflow with the newest
sequence, canonical-endpoint, diagnostics, and RIFE cells from the background
mask workflow. Mask and trajectory input machinery is deliberately excluded.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import runpy
import tempfile
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
LEGACY_BUILDER = ROOT / "scripts" / "build_recursive_flowmorph_vision_notebook.py"
MASK_BUILDER = ROOT / "scripts" / "build_recursive_flowmorph_background_mask_notebook.py"
OUTPUT = Path(
    os.environ.get(
        "FLOWMORPH_PROMPT_ONLY_NOTEBOOK_OUTPUT",
        ROOT
        / "notebooks"
        / "StillLife_Recursive_FlowMorph_Prompt_Only.ipynb",
    )
)
if (
    "FLOWMORPH_PROMPT_ONLY_NOTEBOOK_OUTPUT" not in os.environ
    and OUTPUT.exists()
    and os.environ.get("FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE") != "1"
):
    raise RuntimeError(
        "Refusing to overwrite the tracked prompt-only notebook. Set "
        "FLOWMORPH_PROMPT_ONLY_NOTEBOOK_OUTPUT or explicitly set "
        "FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE=1."
    )


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one replacement target: {old!r}")
    return text.replace(old, new, 1)


def find_cell(notebook: dict, needle: str) -> dict:
    matches = [cell for cell in notebook["cells"] if needle in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one notebook cell containing {needle!r}")
    return matches[0]


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


def replace_literal_assignment(text: str, name: str, value) -> str:
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one literal assignment for {name}")
    node = matches[0]
    lines_ = text.splitlines()
    replacement = f"{name} = {value!r}"
    lines_[node.lineno - 1 : node.end_lineno] = [replacement]
    return "\n".join(lines_) + ("\n" if text.endswith("\n") else "")


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


preserved_stage_cell_source = None
preserved_settings: dict[str, object] = {}
if OUTPUT.is_file():
    try:
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        matching_stage_cells = [
            source(cell)
            for cell in existing["cells"]
            if cell.get("cell_type") == "code"
            and "BASE_STAGES = [" in source(cell)
        ]
        if len(matching_stage_cells) == 1:
            # Preserve the complete editable cell, including alternative
            # commented-out prompt sets and the user's formatting.
            preserved_stage_cell_source = matching_stage_cells[0]
        for name in (
            "DRIVE_PROJECT_BASE",
            "OPENAI_KEY_FILENAME",
            "RESUME_RUN_DIRECTORY",
        ):
            value = literal_assignment(existing, name)
            if value is not None:
                preserved_settings[name] = value
    except json.JSONDecodeError:
        preserved_stage_cell_source = None
        preserved_settings = {}


with tempfile.TemporaryDirectory(prefix="flowmorph_prompt_only_builder_") as temp:
    temp_root = Path(temp)
    prompt_base = temp_root / "prompt_base.ipynb"
    prompt_legacy = temp_root / "prompt_legacy.ipynb"
    mask_latest = temp_root / "mask_latest.ipynb"
    with temporary_environment(
        {
            "FLOWMORPH_BASE_NOTEBOOK_OUTPUT": str(prompt_base),
            "FLOWMORPH_NOTEBOOK_OUTPUT": str(prompt_legacy),
        }
    ):
        runpy.run_path(str(LEGACY_BUILDER), run_name="__main__")
    with temporary_environment(
        {"FLOWMORPH_BACKGROUND_MASK_NOTEBOOK_OUTPUT": str(mask_latest)}
    ):
        runpy.run_path(str(MASK_BUILDER), run_name="__main__")
    prompt_notebook = json.loads(prompt_legacy.read_text(encoding="utf-8"))
    mask_notebook = json.loads(mask_latest.read_text(encoding="utf-8"))


# Prompt-only title/settings/anchors through the section-8 heading, the newest
# midpoint contract, then the latest canonical sequence and finishing stack.
notebook = copy.deepcopy(prompt_notebook)
notebook["cells"] = (
    copy.deepcopy(prompt_notebook["cells"][:17])
    + [copy.deepcopy(mask_notebook["cells"][18])]
    + copy.deepcopy(mask_notebook["cells"][19:])
)
for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"prompt-only-flowmorph-{index:02d}"
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


notebook["cells"][0]["source"] = lines(
    """
    # Prompt-only recursive still-life loop — canonical true FlowMorph

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MNoichl/FluxFlowMorph/blob/main/notebooks/StillLife_Recursive_FlowMorph_Prompt_Only.ipynb)

    This version needs no uploaded prompt JSON, trajectory ZIP, or masks. Edit the
    ordered `BASE_STAGES` list directly, then run:

    1. Text-to-image anchor generation, optionally initialized by a weak,
       blurred, grained version of the previous anchor through conventional img2img.
    2. One retained FLUX.2 model, one backward probe, and one cached fit per unique
       endpoint—even when that endpoint belongs to two neighboring gaps.
    3. One explicit image-aware midpoint per gap in round 1.
    4. Ten true FlowMorph interiors per resulting gap in round 2, conditioned
       piecewise source → shared midpoint prompt → target.
    5. One canonical decoded reconstruction per fitted endpoint, reused identically
       for its incoming α=1 and outgoing α=0 appearances.
    6. Optional temporal tone stabilization, a read-only cyclic flicker audit,
       Practical-RIFE, circular SSIM timing equalization, and H.264 export.

    Raw anchors, raw FlowMorph renders, fitted endpoint reconstructions, corrected
    copies, manifests, diagnostics, and videos are written directly to the
    timestamped Google Drive run directory.
    """
)
notebook["cells"][1]["source"] = lines(
    """
    ## 1. Editable prompt-only run, generation, FlowMorph, and video settings

    `BASE_STAGES` is the only creative input. The defaults below reproduce the
    current quality/speed contract learned in the mask workflow while keeping every
    consequential value exposed. Increase both endpoint optimization-step settings
    from 50 to 100 for the slower quality-first fit.
    """
)


settings_cell = notebook["cells"][2]
settings = source(settings_cell)
settings = replace_once(
    settings,
    'PROJECT_NAME = "science_path_recursive_flowmorph"',
    'PROJECT_NAME = "science_path_prompt_only_flowmorph"',
)
settings = replace_once(
    settings,
    "BASE_PROMPT_COUNT = 15",
    "BASE_PROMPT_COUNT = None  # None uses every entry in BASE_STAGES.",
)
for old, new in (
    ("FLOWMORPH_FIT_LORA_SCALE = 1.0", "FLOWMORPH_FIT_LORA_SCALE = 1.2"),
    ("FLOWMORPH_RENDER_LORA_SCALE = 1.0", "FLOWMORPH_RENDER_LORA_SCALE = 1.2"),
    ("FLOWMORPH_GUIDANCE_SCALE = 4.0", "FLOWMORPH_GUIDANCE_SCALE = 7.0"),
    (
        "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 100",
        "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 50",
    ),
    (
        "FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 100",
        "FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 50",
    ),
    ("IMAGE_INFERENCE_STEPS = 28", "IMAGE_INFERENCE_STEPS = 50"),
    ("IMAGE_GUIDANCE_SCALE = 4.0", "IMAGE_GUIDANCE_SCALE = 7.0"),
    ("IMAGE_LORA_SCALE = 1.0", "IMAGE_LORA_SCALE = 1.2"),
    ("BASE_SEED = 1729", "BASE_SEED = 42  # Change for a new deterministic run."),
):
    settings = replace_once(settings, old, new)
settings = replace_once(
    settings,
    "BASE_REFERENCE_STRENGTH = 0.12\n",
    "",
)
settings = replace_once(
    settings,
    "REFERENCE_BACKGROUND = (116, 105, 91)\n",
    "BASE_REFERENCE_DENOISE_STRENGTH = 0.75\n",
)
settings = replace_once(
    settings,
    "SAVE_SOFT_REFERENCES = True  # Inspect in base_frames/soft_references and its preview sheet.\n",
    "SAVE_SOFT_REFERENCES = True  # Inspect in base_frames/soft_references.\n"
    "FLUX_PROMPT_MAX_SEQUENCE_LENGTH = 512\n\n"
    "# Optional post-FlowMorph tonal correction; raw PNGs are never overwritten.\n"
    "TEMPORAL_TONE_STABILIZATION_ENABLED = False\n"
    "TEMPORAL_TONE_WINDOW_RADIUS = 2\n"
    "TEMPORAL_TONE_STRENGTH = 0.70\n"
    "TEMPORAL_TONE_MEAN_THRESHOLD = 0.02\n"
    "TEMPORAL_TONE_CONTRAST_THRESHOLD = 0.10\n"
    "TEMPORAL_TONE_MAD_MULTIPLIER = 3.5\n"
    "TEMPORAL_TONE_MAX_MEAN_SHIFT = 0.06\n"
    "TEMPORAL_TONE_MAX_CONTRAST_SCALE_DELTA = 0.15\n"
    "TEMPORAL_TONE_ANALYSIS_MAX_SIDE = 256\n"
    "TEMPORAL_TONE_REUSE_EXISTING = True\n\n"
    "# Read-only diagnosis of raw cyclic FlowMorph output.\n"
    "RUN_FLICKER_DIAGNOSTIC = True\n"
    "FLICKER_ANALYSIS_MAX_SIDE = 256\n"
    "FLICKER_OUTLIER_MAD_MULTIPLIER = 3.5\n"
    "FLICKER_MINIMUM_OUTLIER_SCORE = 3.0\n"
    "FLICKER_MAX_LAG = 64\n",
)
settings = replace_once(
    settings,
    "SOURCE_SEQUENCE_FPS = 12.0\n",
    "VIDEO_SLOWDOWN_FACTOR = 3.0\n"
    "SOURCE_SEQUENCE_FPS = 12.0 / VIDEO_SLOWDOWN_FACTOR\n",
)
settings = replace_once(
    settings,
    "RIFE_MULTIPLIER = 2\n",
    "RIFE_MULTIPLIER = int(round(2 * VIDEO_SLOWDOWN_FACTOR))\n",
)
for name, value in preserved_settings.items():
    settings = replace_literal_assignment(settings, name, value)
settings_cell["source"] = settings.splitlines(keepends=True)


if preserved_stage_cell_source is not None:
    notebook["cells"][4]["source"] = preserved_stage_cell_source.splitlines(
        keepends=True
    )


validation_cell = notebook["cells"][10]
validation = source(validation_cell)
validation = replace_once(
    validation,
    "if not 3 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):\n"
    '    raise ValueError(f"BASE_PROMPT_COUNT must be between 3 and {len(BASE_STAGES)}")\n',
    "if BASE_PROMPT_COUNT is None:\n"
    "    BASE_PROMPT_COUNT = len(BASE_STAGES)\n"
    "elif not 3 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):\n"
    '    raise ValueError(f"BASE_PROMPT_COUNT must be between 3 and {len(BASE_STAGES)}")\n',
)
validation = validation.replace(
    'if FLOWMORPH_SOURCE_OPTIMIZATION_STEPS != 100:\n'
    '    raise ValueError("This quality-first notebook requires 100 endpoint fitting steps")\n',
    "",
    1,
)
validation = replace_once(
    validation,
    'if not 0 < BASE_REFERENCE_STRENGTH <= 1.0:\n'
    '    raise ValueError("BASE_REFERENCE_STRENGTH must lie in (0, 1]")\n',
    "",
)
validation = replace_once(
    validation,
    'if not 0 <= BASE_REFERENCE_GRAIN_STRENGTH <= 0.25:\n'
    '    raise ValueError("BASE_REFERENCE_GRAIN_STRENGTH must lie in [0, 0.25]")\n',
    'if not 0 <= BASE_REFERENCE_GRAIN_STRENGTH <= 0.25:\n'
    '    raise ValueError("BASE_REFERENCE_GRAIN_STRENGTH must lie in [0, 0.25]")\n'
    'if not 0 < BASE_REFERENCE_DENOISE_STRENGTH <= 1:\n'
    '    raise ValueError("BASE_REFERENCE_DENOISE_STRENGTH must lie in (0, 1]")\n'
    'if not 32 <= FLUX_PROMPT_MAX_SEQUENCE_LENGTH <= 512:\n'
    '    raise ValueError("FLUX_PROMPT_MAX_SEQUENCE_LENGTH must lie in [32, 512]")\n'
    'if TEMPORAL_TONE_WINDOW_RADIUS < 1:\n'
    '    raise ValueError("TEMPORAL_TONE_WINDOW_RADIUS must be positive")\n'
    'if not 0 <= TEMPORAL_TONE_STRENGTH <= 1:\n'
    '    raise ValueError("TEMPORAL_TONE_STRENGTH must lie in [0, 1]")\n'
    'if not 0 <= TEMPORAL_TONE_MAX_MEAN_SHIFT <= 1:\n'
    '    raise ValueError("TEMPORAL_TONE_MAX_MEAN_SHIFT must lie in [0, 1]")\n'
    'if not 0 <= TEMPORAL_TONE_MAX_CONTRAST_SCALE_DELTA < 1:\n'
    '    raise ValueError("TEMPORAL_TONE_MAX_CONTRAST_SCALE_DELTA must lie in [0, 1)")\n'
    'if VIDEO_SLOWDOWN_FACTOR < 1:\n'
    '    raise ValueError("VIDEO_SLOWDOWN_FACTOR must be at least 1")\n'
    'if FLICKER_ANALYSIS_MAX_SIDE < 32 or FLICKER_MAX_LAG < 1:\n'
    '    raise ValueError("Flicker analysis settings are invalid")\n',
)
validation_cell["source"] = validation.splitlines(keepends=True)


notebook["cells"][11]["source"] = lines(
    """
    ## 6. Load RIJKSOIL; optional anchor trial and one-gap FlowMorph gate

    The trial tests prompt, LoRA, guidance, inference-step, and image-size settings.
    After anchors exist, section 9 fits one gap and displays original and canonical
    fitted endpoints around α=.25/.5/.75 before the expensive full recursion.
    """
)


model_cell = notebook["cells"][12]
model_source = source(model_cell)
model_source = replace_once(
    model_source,
    "from flowmorph_klein.lora import load_flux2_lora\n",
    "from flowmorph_klein.lora import load_flux2_lora\n"
    "from flowmorph_klein.trajectory import prepare_flux2_klein_img2img_inputs\n",
)
token_helpers = dedent(
    """
    FLUX_PROMPT_TOKENIZER = FLUX_PIPE.tokenizer

    def flux_prompt_token_count(prompt):
        messages = [{"role": "user", "content": prompt}]
        templated = FLUX_PROMPT_TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = FLUX_PROMPT_TOKENIZER(
            templated,
            add_special_tokens=False,
            truncation=False,
        )
        return len(encoded["input_ids"])

    def validate_flux_prompt_length(prompt, label="Prompt"):
        token_count = flux_prompt_token_count(prompt)
        if token_count > FLUX_PROMPT_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"{label} tokenizes to {token_count} tokens after the FLUX chat "
                f"template; maximum is {FLUX_PROMPT_MAX_SEQUENCE_LENGTH}"
            )
        return token_count

    """
)
model_source = replace_once(
    model_source,
    'else:\n    print("Reusing the fused pipeline at the current LoRA scale.")\n\n'
    "if RUN_TRIAL_KEYFRAME:\n",
    'else:\n    print("Reusing the fused pipeline at the current LoRA scale.")\n\n'
    + token_helpers
    + "if RUN_TRIAL_KEYFRAME:\n",
)
model_source = replace_once(
    model_source,
    '    trial_result = FLUX_PIPE(\n        prompt=trial_stage["prompt"],\n',
    '    validate_flux_prompt_length(trial_stage["prompt"], "Trial anchor prompt")\n'
    '    trial_result = FLUX_PIPE(\n'
    '        prompt=trial_stage["prompt"],\n',
)
model_source = replace_once(
    model_source,
    '        output_type="pil",\n    )\n',
    '        output_type="pil",\n'
    '        max_sequence_length=FLUX_PROMPT_MAX_SEQUENCE_LENGTH,\n'
    '    )\n',
)
model_cell["source"] = model_source.splitlines(keepends=True)


notebook["cells"][13]["source"] = lines(
    """
    ## 7. Generate prompt-only cyclic anchor paintings

    The first anchor is ordinary text-to-image. Later anchors optionally receive a
    weak blurred/grained previous painting as a conventional latent img2img start.
    Gaussian smoothing removes fine structure while optional grain prevents a
    featureless wash. `BASE_REFERENCE_DENOISE_STRENGTH` controls how strongly FLUX
    repaints the result toward the new prompt. No solid-color canvas, mask, spatial
    constraint, or post-composite is used.
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
        generator = torch.Generator(device="cuda").manual_seed(seed)
        kwargs = {
            "prompt": prompt,
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "num_inference_steps": IMAGE_INFERENCE_STEPS,
            "guidance_scale": IMAGE_GUIDANCE_SCALE,
            "generator": generator,
            "output_type": "pil",
            "max_sequence_length": FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
        }
        generation_report = {
            "mode": "text_to_image",
            "requested_img2img_strength": None,
            "effective_start_sigma": None,
        }
        if reference is not None:
            generation_inputs = prepare_flux2_klein_img2img_inputs(
                FLUX_PIPE,
                reference,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                num_inference_steps=IMAGE_INFERENCE_STEPS,
                strength=BASE_REFERENCE_DENOISE_STRENGTH,
                generator=generator,
            )
            kwargs["sigmas"] = list(generation_inputs.sigmas)
            kwargs["latents"] = generation_inputs.latents
            generation_report = {
                "mode": "latent_img2img_from_weak_previous_reference",
                "requested_img2img_strength": (
                    generation_inputs.requested_strength
                ),
                "effective_start_sigma": generation_inputs.effective_start_sigma,
                "denoising_steps": generation_inputs.denoising_steps,
            }
        result = FLUX_PIPE(**kwargs)
        if not result.images:
            raise RuntimeError("FLUX returned no anchor image")
        return result.images[0].convert("RGB"), generation_report

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
                    # A blend of 1.0 means 100% blurred previous image. No
                    # fixed beige/gray background canvas contributes.
                    reference_blend=1.0,
                    blur_radius=BASE_REFERENCE_BLUR,
                    grain_strength=BASE_REFERENCE_GRAIN_STRENGTH,
                    grain_seed=seed,
                )
                if SAVE_SOFT_REFERENCES:
                    REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                    reference_path = (
                        REFERENCE_DIRECTORY / f"reference_{index:03d}.png"
                    )
                    reference.save(reference_path, format="PNG", compress_level=4)
            image, generation_report = generate_prompt_anchor(
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
                "base_reference_source": (
                    "blurred_grained_previous_without_flat_canvas"
                ),
                "base_reference_blur": BASE_REFERENCE_BLUR,
                "base_reference_grain_strength": (
                    BASE_REFERENCE_GRAIN_STRENGTH
                ),
                "generation_mode": generation_report["mode"],
                "img2img_strength": generation_report[
                    "requested_img2img_strength"
                ],
                "effective_start_sigma": generation_report[
                    "effective_start_sigma"
                ],
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


# Use the latest midpoint contract from the mask workflow; it has no mask
# dependency and includes exact Qwen chat-template token validation.
midpoint_cell = notebook["cells"][17]
if "validate_flux_prompt_length(clean, \"Midpoint prompt\")" not in source(midpoint_cell):
    raise RuntimeError("Latest midpoint prompt-length contract was not composed")


sequence_cell = find_cell(notebook, "SEQUENCE_SESSION_CONTRACT")
sequence_source = source(sequence_cell)
for import_line in (
    "from flowmorph_klein.conditioning import stack_conditioning_packages\n",
    "from flowmorph_klein.diagnostics import release_cuda_memory\n",
    "from flowmorph_klein.flow_schedule import get_render_chain\n",
    "from flowmorph_klein.renderer import RenderedLatentFrame, render_latent_trajectory\n",
    "from flowmorph_klein.types import RenderConditioningMode\n",
):
    sequence_source = sequence_source.replace(import_line, "", 1)
fallback_start = sequence_source.index(
    "\ndef render_canonical_endpoint_reconstructions("
)
fallback_end = sequence_source.index(
    "\ndef load_or_create_shared_prompt(",
    fallback_start,
)
sequence_source = (
    sequence_source[:fallback_start]
    + "\n"
    + sequence_source[fallback_end:]
)
sequence_source = replace_once(
    sequence_source,
    "frames = render_canonical_endpoint_reconstructions(\n"
    "        SEQUENCE_SESSION,\n",
    "frames = SEQUENCE_SESSION.render_endpoint_reconstructions(\n",
)
sequence_cell["source"] = sequence_source.splitlines(keepends=True)


assembly_markdown = find_cell(notebook, "## 10. Assemble, preview")
assembly_markdown["source"] = lines(
    """
    ## 10. Assemble, stabilize, preview, and audit the cyclic sequence

    Every cached fitted endpoint is decoded once and reused as the identical
    incoming/outgoing boundary image. Optional temporal tone stabilization changes
    only detected local luminance/contrast outliers and leaves raw FlowMorph PNGs
    untouched. The contact sheet and reduced preview use the selected final paths.
    """
)


for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    text = source(cell).replace(
        "recursive_flowmorph_background_mask_rife_ssim_loop.mp4",
        "recursive_flowmorph_prompt_only_rife_ssim_loop.mp4",
    )
    cell["source"] = text.splitlines(keepends=True)


flicker_cell = notebook["cells"][-1]
flicker_source = source(flicker_cell)
runtime_marker = "from IPython.display import Markdown, display\n\nif RUN_FLICKER_DIAGNOSTIC:"
runtime_index = flicker_source.rfind(runtime_marker)
if runtime_index < 0:
    raise RuntimeError("Could not locate flicker diagnostic runtime")
flicker_runtime = flicker_source[runtime_index:]
flicker_cell["source"] = lines(
    """
    from flowmorph_klein.flicker_diagnostics import (
        FlickerDiagnosticConfig,
        diagnose_cyclic_flicker,
        format_flicker_diagnostic_markdown,
    )
    """
) + flicker_runtime.splitlines(keepends=True)
flicker_heading = notebook["cells"][-2]
flicker_heading["source"] = lines(
    """
    ## 14. Read-only cyclic flicker diagnosis

    This final cell analyzes raw FlowMorph frames without modifying them. The
    implementation is imported from the installed project package, keeping this
    notebook compact while saving the complete plot and JSON audit to Drive.
    """
)


notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} clean cells")
