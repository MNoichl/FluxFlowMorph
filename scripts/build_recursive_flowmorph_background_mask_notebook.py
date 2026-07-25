"""Build the separate background-mask trajectory FlowMorph notebook."""

from __future__ import annotations

import json
import os
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
        cell["source"] = source(cell).replace(
            '"repository_commit": project_com mit,',
            '"repository_commit": project_commit,',
        ).splitlines(keepends=True)

notebook["cells"][0]["source"] = lines(
    """
    # Recursive science still-life loop — background-mask trajectory + true FlowMorph

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MNoichl/FluxFlowMorph/blob/main/notebooks/StillLife_Recursive_FlowMorph_Trajectory_Background_Mask.ipynb)

    This is a separate masking experiment. It keeps the full recursive FlowMorph
    and RIFE workflow, but the ZIP frames are used only to locate editable regions:

    - pixels close to `SOURCE_BACKGROUND_RGB` are black/protected in the mask;
    - every other pixel is white/editable and regenerated from noise;
    - the source image is never used as a latent init or FLUX image reference;
    - the protected region is locked during denoising and composited to the exact
      `OUTPUT_BACKGROUND_RGB` at the end.

    Run the trial first. Its mask preview makes polarity and tolerance errors visible
    before any full anchor batch is generated.
    """
)
notebook["cells"][1]["source"] = lines(
    """
    ## 1. Editable run, background-mask, model, API, FlowMorph, image, and video settings

    `SOURCE_BACKGROUND_RGB` is the color to recognize in the ZIP frames.
    `OUTPUT_BACKGROUND_RGB` is the flat color written outside the editable mask.
    White mask pixels are regenerated; black mask pixels are protected.
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
)
mask_settings = dedent(
    """
    # Background-mask trajectory experiment. The source supplies geometry only.
    TRAJECTORY_ZIP_DIRECTORY = "/content/drive/MyDrive/FluxFlowMorphArt/trajectory_inputs"
    TRAJECTORY_ZIP_FILENAME = "background.zip"
    TRAJECTORY_MEMBER_PREFIX = ""  # Optional folder inside the ZIP, without leading slash.
    TRAJECTORY_FRAME_OFFSET = 0
    TRAJECTORY_REVERSE_ORDER = False

    # Background detection: matching pixels are protected (black in the mask).
    SOURCE_BACKGROUND_RGB = (238, 233, 218)
    BACKGROUND_COLOR_TOLERANCE = 0.08  # Normalized RGB distance treated as background.
    BACKGROUND_MASK_SOFTNESS = 0.05  # Transition width after the tolerance boundary.
    BACKGROUND_MASK_EXPANSION = 2  # Grow white/editable regions by this many pixels.
    BACKGROUND_MASK_FEATHER = 3.0
    MASK_MIN_EDITABLE_FRACTION = 0.01
    MASK_MAX_EDITABLE_FRACTION = 0.95

    # Generation: source pixels never enter FLUX; editable regions begin from noise.
    OUTPUT_BACKGROUND_RGB = (238, 233, 218)
    MASK_DENOISE_STRENGTH = 1.0  # 1.0 gives editable regions maximum freedom.
    MASK_PROMPT_INSTRUCTION = (
        "Arrange the described painted forms densely within the available shaped field, "
        "with natural cropped edges and a calm unpainted surround."
    )
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

validation_cell = find_cell(notebook, "def trajectory_generation_prompt")
validation = source(validation_cell)
mask_validation = dedent(
    """
    if len(SOURCE_BACKGROUND_RGB) != 3 or any(
        not 0 <= channel <= 255 for channel in SOURCE_BACKGROUND_RGB
    ):
        raise ValueError("SOURCE_BACKGROUND_RGB must contain three values in [0, 255]")
    if len(OUTPUT_BACKGROUND_RGB) != 3 or any(
        not 0 <= channel <= 255 for channel in OUTPUT_BACKGROUND_RGB
    ):
        raise ValueError("OUTPUT_BACKGROUND_RGB must contain three values in [0, 255]")
    if not 0 <= BACKGROUND_COLOR_TOLERANCE < 1:
        raise ValueError("BACKGROUND_COLOR_TOLERANCE must lie in [0, 1)")
    if not 0 < BACKGROUND_MASK_SOFTNESS <= 1:
        raise ValueError("BACKGROUND_MASK_SOFTNESS must lie in (0, 1]")
    if not isinstance(BACKGROUND_MASK_EXPANSION, int) or not 0 <= BACKGROUND_MASK_EXPANSION <= 128:
        raise ValueError("BACKGROUND_MASK_EXPANSION must be an integer in [0, 128]")
    if BACKGROUND_MASK_FEATHER < 0:
        raise ValueError("BACKGROUND_MASK_FEATHER cannot be negative")
    if not 0 <= MASK_MIN_EDITABLE_FRACTION < MASK_MAX_EDITABLE_FRACTION <= 1:
        raise ValueError("Editable-fraction limits must satisfy 0 <= min < max <= 1")
    if not 0 < MASK_DENOISE_STRENGTH <= 1:
        raise ValueError("MASK_DENOISE_STRENGTH must lie in (0, 1]")

    def trajectory_generation_prompt(prompt):
        base = prompt
        if TRAJECTORY_REMOVE_SYMMETRY_LANGUAGE:
            base = re.sub(r"\\bsymmetr(?:ical|ically)\\b", "", base, flags=re.IGNORECASE)
        return " ".join(base.split()) + " " + MASK_PROMPT_INSTRUCTION
    """
).lstrip()
validation = replace_between(
    validation,
    "if not 0 <= TRAJECTORY_INIT_DETAIL_STRENGTH <= 1:\n",
    "if OPENAI_IMAGE_DETAIL not in",
    mask_validation,
)
validation_cell["source"] = validation.splitlines(keepends=True)

model_cell = find_cell(notebook, "if RUN_TRIAL_KEYFRAME:")
model = source(model_cell)
pipeline_setup = model[
    model.index("try:\n    import peft.tuners.lora.torchao")
    : model.index("if RUN_TRIAL_KEYFRAME:")
]
model_cell["source"] = code_blocks(
    """
    import gc
    import os
    import random
    import shutil
    from huggingface_hub import hf_hub_download
    from IPython.display import Markdown, display
    from PIL import Image
    from flowmorph_klein.lora import load_flux2_lora
    from flowmorph_klein.trajectory import (
        composite_generated_on_background,
        make_background_edit_mask,
        prepare_flux2_klein_masked_inpaint_inputs,
    )

    def build_background_mask(source):
        result = make_background_edit_mask(
            source,
            background_rgb=SOURCE_BACKGROUND_RGB,
            tolerance=BACKGROUND_COLOR_TOLERANCE,
            softness=BACKGROUND_MASK_SOFTNESS,
            expansion_radius=BACKGROUND_MASK_EXPANSION,
            feather_radius=BACKGROUND_MASK_FEATHER,
        )
        if not (
            MASK_MIN_EDITABLE_FRACTION
            <= result.editable_fraction
            <= MASK_MAX_EDITABLE_FRACTION
        ):
            raise RuntimeError(
                f"Editable mask coverage {result.editable_fraction:.1%} is outside "
                f"{MASK_MIN_EDITABLE_FRACTION:.1%}–{MASK_MAX_EDITABLE_FRACTION:.1%}. "
                "Adjust SOURCE_BACKGROUND_RGB or BACKGROUND_COLOR_TOLERANCE. "
                "White must mark content; black must mark background."
            )
        return result

    def generate_masked_anchor(source, prompt, seed):
        mask_result = build_background_mask(source)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        masked_inputs = prepare_flux2_klein_masked_inpaint_inputs(
            FLUX_PIPE,
            mask_result.mask,
            background_rgb=OUTPUT_BACKGROUND_RGB,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=IMAGE_INFERENCE_STEPS,
            strength=MASK_DENOISE_STRENGTH,
            generator=generator,
        )
        result = FLUX_PIPE(
            prompt=prompt,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            num_inference_steps=IMAGE_INFERENCE_STEPS,
            sigmas=list(masked_inputs.sigmas),
            latents=masked_inputs.latents,
            guidance_scale=IMAGE_GUIDANCE_SCALE,
            generator=generator,
            output_type="pil",
            callback_on_step_end=masked_inputs.callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        if not result.images:
            raise RuntimeError("FLUX returned no masked image")
        raw_image = result.images[0].convert("RGB")
        final_image = composite_generated_on_background(
            raw_image,
            mask_result.mask,
            background_rgb=OUTPUT_BACKGROUND_RGB,
        )
        return final_image, raw_image, mask_result, masked_inputs

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
            trial_source = opened.convert("RGB")
        trial_generation_prompt = trajectory_generation_prompt(trial_stage["prompt"])
        trial_image, trial_raw, trial_mask, trial_inputs = generate_masked_anchor(
            trial_source,
            trial_generation_prompt,
            trial_seed,
        )
        trial_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        trial_directory = (
            RUN_DIRECTORY / "trials" / f"{trial_stamp}_{trial_stage['id']}_{trial_seed}"
        )
        trial_directory.mkdir(parents=True, exist_ok=False)
        trial_path = trial_directory / "trial_background_locked.png"
        trial_raw_path = trial_directory / "trial_raw.png"
        trial_source_path = trial_directory / "trajectory_source.png"
        trial_mask_path = trial_directory / "edit_mask_white_is_editable.png"
        trial_image.save(trial_path)
        trial_raw.save(trial_raw_path)
        trial_source.save(trial_source_path)
        trial_mask.mask.save(trial_mask_path)
        (trial_directory / "settings.json").write_text(json.dumps({
            "stage": trial_stage,
            "trajectory": trial_trajectory,
            "seed": trial_seed,
            "source_background_rgb": list(SOURCE_BACKGROUND_RGB),
            "output_background_rgb": list(OUTPUT_BACKGROUND_RGB),
            "background_color_tolerance": BACKGROUND_COLOR_TOLERANCE,
            "background_mask_softness": BACKGROUND_MASK_SOFTNESS,
            "background_mask_expansion": BACKGROUND_MASK_EXPANSION,
            "background_mask_feather": BACKGROUND_MASK_FEATHER,
            "editable_fraction": trial_mask.editable_fraction,
            "mask_polarity": "white_editable_black_protected",
            "source_used_only_to_derive_mask": True,
            "source_used_as_latent_init": False,
            "source_used_as_image_reference": False,
            "effective_start_sigma": trial_inputs.effective_start_sigma,
            "effective_denoising_steps": trial_inputs.denoising_steps,
            "generation_prompt": trial_generation_prompt,
        }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        previews = [
            ("Source used only for mask geometry", trial_source.copy()),
            ("Edit mask — WHITE changes, BLACK stays background", trial_mask.mask.convert("RGB")),
            ("Raw masked-denoising result", trial_raw.copy()),
            ("Final exact-background result", trial_image.copy()),
        ]
        for heading, preview in previews:
            preview.thumbnail((TRIAL_DISPLAY_MAX_WIDTH, TRIAL_DISPLAY_MAX_WIDTH))
            display(Markdown(f"### {heading}"))
            display(preview)
            preview.close()
        print({
            "path": str(trial_path),
            "raw_path": str(trial_raw_path),
            "source_path": str(trial_source_path),
            "mask_path": str(trial_mask_path),
            "editable_fraction": trial_mask.editable_fraction,
            "seed": trial_seed,
            "prompt_index": trial_index,
        })
        del trial_image, trial_raw, trial_source, trial_mask, trial_inputs
    else:
        print("Trial skipped.")
    """
)

anchor_markdown = find_cell(notebook, "## 7. Generate trajectory-conditioned")
anchor_markdown["source"] = lines(
    """
    ## 7. Generate background-masked cyclic anchor paintings

    Each selected ZIP frame supplies only a white-edit/black-protect mask. Editable
    regions start from noise and receive the science prompt. Background regions are
    restored after every denoising step and written in the exact selected color.
    """
)

anchor_cell = find_cell(notebook, "BASE_MANIFEST_PATH")
anchor_cell["source"] = lines(
    """
    BASE_DIRECTORY = RUN_DIRECTORY / "base_frames"
    MASK_DIRECTORY = BASE_DIRECTORY / "background_edit_masks"
    RAW_DIRECTORY = BASE_DIRECTORY / "raw_masked_generations"
    BASE_MANIFEST_PATH = RUN_DIRECTORY / "metadata" / "base_manifest.json"
    BASE_RECORDS = []
    MASK_CONTRACT = {
        "source_background_rgb": list(SOURCE_BACKGROUND_RGB),
        "output_background_rgb": list(OUTPUT_BACKGROUND_RGB),
        "tolerance": BACKGROUND_COLOR_TOLERANCE,
        "softness": BACKGROUND_MASK_SOFTNESS,
        "expansion": BACKGROUND_MASK_EXPANSION,
        "feather": BACKGROUND_MASK_FEATHER,
        "denoise_strength": MASK_DENOISE_STRENGTH,
        "polarity": "white_editable_black_protected",
        "source_used_only_to_derive_mask": True,
        "source_used_as_latent_init": False,
        "source_used_as_image_reference": False,
    }
    expected_anchor_contract = [
        {
            "uid": f"base_{index:03d}",
            "science": stage["science"],
            "prompt": stage["prompt"],
            "generation_prompt": trajectory_generation_prompt(stage["prompt"]),
            "trajectory_member": trajectory["member"],
            "trajectory_member_sha256": trajectory["member_sha256"],
            "mask_contract": MASK_CONTRACT,
        }
        for index, (stage, trajectory) in enumerate(
            zip(ACTIVE_BASE_STAGES, TRAJECTORY_RECORDS, strict=True)
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
        print(f"Loaded {len(BASE_RECORDS)} existing background-masked anchors.")
    else:
        MASK_DIRECTORY.mkdir(parents=True, exist_ok=True)
        RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for index, (stage, trajectory) in enumerate(
            zip(ACTIVE_BASE_STAGES, TRAJECTORY_RECORDS, strict=True)
        ):
            seed = BASE_SEED + index
            with Image.open(trajectory["path"]) as opened:
                source_image = opened.convert("RGB")
            generation_prompt = trajectory_generation_prompt(stage["prompt"])
            image, raw_image, mask_result, masked_inputs = generate_masked_anchor(
                source_image,
                generation_prompt,
                seed,
            )
            mask_path = MASK_DIRECTORY / f"mask_{index:03d}.png"
            raw_path = RAW_DIRECTORY / f"{index:03d}_{stage['id']}_raw.png"
            output_path = BASE_DIRECTORY / f"{index:03d}_{stage['id']}.png"
            mask_result.mask.save(mask_path, format="PNG", compress_level=4)
            raw_image.save(raw_path, format="PNG", compress_level=4)
            image.save(output_path, format="PNG", compress_level=4)
            record = {
                "uid": f"base_{index:03d}",
                "kind": "base",
                "round": 0,
                "science": stage["science"],
                "prompt": stage["prompt"],
                "generation_prompt": generation_prompt,
                "seed": seed,
                "path": str(output_path),
                "raw_generation_path": str(raw_path),
                "trajectory_index": trajectory["trajectory_index"],
                "trajectory_member": trajectory["member"],
                "trajectory_member_sha256": trajectory["member_sha256"],
                "trajectory_source_path": trajectory["path"],
                "trajectory_edit_mask_path": str(mask_path),
                "editable_fraction": mask_result.editable_fraction,
                "mask_contract": MASK_CONTRACT,
                "effective_start_sigma": masked_inputs.effective_start_sigma,
                "effective_denoising_steps": masked_inputs.denoising_steps,
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
                f"({mask_result.editable_fraction:.1%} editable)"
            )
            source_image.close()
            raw_image.close()
            image.close()
            mask_result.mask.close()
            del masked_inputs

    if len(BASE_RECORDS) != len(ACTIVE_BASE_STAGES):
        raise RuntimeError("The anchor manifest is incomplete; regenerate or resume the correct run.")
    print(f"Prepared {len(BASE_RECORDS)} background-masked anchors in {BASE_DIRECTORY}")
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
        RUN_DIRECTORY / "previews" / "trajectory_source_mask_anchor_triplets.png"
    )
    paired_images = []
    paired_labels = []
    for index, record in enumerate(BASE_RECORDS):
        paired_images.extend(load_contact_thumbnails([
            Path(record["trajectory_source_path"]),
            Path(record["trajectory_edit_mask_path"]),
            Path(record["path"]),
        ]))
        paired_labels.extend([
            f"{index:02d} source",
            f"{index:02d} WHITE=edit",
            f"{index:02d} generated",
        ])
    make_contact_sheet(
        paired_images,
        paired_contact_sheet_path,
        columns=3,
        labels=paired_labels,
    )
    for image in paired_images:
        image.close()

    preview = Image.open(paired_contact_sheet_path).convert("RGB")
    preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
    display(Markdown("### Source → white-edit mask → generated anchor"))
    display(preview)
    preview.close()
    print({
        "full_resolution_sources": str(
            Path(BASE_RECORDS[0]["trajectory_source_path"]).parent
        ),
        "full_resolution_masks": str(
            Path(BASE_RECORDS[0]["trajectory_edit_mask_path"]).parent
        ),
        "full_resolution_anchors": str(BASE_DIRECTORY),
        "paired_audit": str(paired_contact_sheet_path),
    })
    """
)

assembly_markdown = find_cell(notebook, "## 10. Assemble, preview")
assembly_markdown["source"] = lines(
    """
    ## 10. Assemble, preview, and audit the generated cyclic masked sequence

    The background-masked anchors now enter the unchanged recursive FlowMorph pipeline.
    Round 1 contributes one explicit midpoint, Round 2 contributes ten shared-prompt
    renders, and the final-to-first gap closes the loop before RIFE finishing.
    """
)

final_video_cell = find_cell(notebook, "recursive_flowmorph_trajectory_rife_ssim_loop.mp4")
final_video_source = source(final_video_cell).replace(
    "recursive_flowmorph_trajectory_rife_ssim_loop.mp4",
    "recursive_flowmorph_background_mask_rife_ssim_loop.mp4",
)
final_video_cell["source"] = final_video_source.splitlines(keepends=True)

notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} clean cells")
