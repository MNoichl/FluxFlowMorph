"""Build the trajectory-ZIP variant of the recursive FlowMorph art notebook."""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Vision.ipynb"
OUTPUT = Path(
    os.environ.get(
        "FLOWMORPH_TRAJECTORY_NOTEBOOK_OUTPUT",
        ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Trajectory_Init.ipynb",
    )
)
if (
    "FLOWMORPH_TRAJECTORY_NOTEBOOK_OUTPUT" not in os.environ
    and OUTPUT.exists()
    and os.environ.get("FLOWMORPH_ALLOW_NOTEBOOK_OVERWRITE") != "1"
):
    raise RuntimeError(
        "Refusing to overwrite the tracked trajectory notebook. Set a temporary "
        "FLOWMORPH_TRAJECTORY_NOTEBOOK_OUTPUT or explicitly set "
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


notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
notebook["cells"] = [
    cell
    for cell in notebook["cells"]
    if cell.get("cell_type") != "code" or source(cell).strip()
]
for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
for index, cell in enumerate(notebook["cells"]):
    cell.setdefault("id", f"trajectory-cell-{index:02d}")

notebook["cells"][0]["source"] = lines(
    """
    # Recursive science still-life loop — trajectory ZIP + true FlowMorph

    This notebook follows the same recursive, cyclic FlowMorph workflow as
    `StillLife_Recursive_FlowMorph_Vision`, but the anchor paintings follow an
    external image trajectory stored as a ZIP on Google Drive.

    1. Edit the sciences/prompts and choose how many anchors to use.
    2. Point `TRAJECTORY_ZIP_DIRECTORY` and `TRAJECTORY_ZIP_FILENAME` at the Drive archive.
    3. The archive is naturally sorted and sampled at equal intervals. With 180
       images and 18 prompts, members `0, 10, …, 170` become the strong init frames.
    4. FLUX.2 Klein + RIJKSOIL regenerates each sampled frame with its corresponding
       science prompt while retaining the source trajectory.
    5. Vision midpoint prompting, cached/batched FlowMorph, streaming Drive outputs,
       cyclic assembly, batched RIFE, SSIM equalization, and MP4 export continue exactly
       as in the original notebook.

    The ZIP is copied once to local Colab storage for efficient access. Only the
    regularly sampled members are decoded and persisted into the run directory.
    """
)
notebook["cells"][1]["source"] = lines(
    """
    ## 1. Editable run, trajectory ZIP, model, API, FlowMorph, image, and video settings

    By default every entry in `BASE_STAGES` is used; there is no ten-prompt ceiling.
    Set `BASE_PROMPT_COUNT` only when you intentionally want to use a shorter prefix.
    True latent img2img plus the reference image preserve the sampled spatial layout.
    """
)

settings = source(notebook["cells"][2])
settings = replace_once(
    settings,
    'PROJECT_NAME = "science_path_recursive_flowmorph"',
    'PROJECT_NAME = "science_path_trajectory_flowmorph"',
)
settings = replace_once(
    settings,
    "BASE_PROMPT_COUNT = 15",
    "BASE_PROMPT_COUNT = None  # None uses every BASE_STAGES entry; an integer caps the list.",
)
settings = replace_once(
    settings,
    "TRIAL_KEYFRAME_INDEX = None  # None chooses randomly; otherwise 0..BASE_PROMPT_COUNT-1.",
    "TRIAL_KEYFRAME_INDEX = None  # None chooses randomly from the active prompt list.",
)
settings = replace_once(
    settings,
    '# Weak continuity applies only to standalone anchor generation.\n'
    'BASE_CONTINUITY_ENABLED = True\n'
    '# BASE_REFERENCE_STRENGTH = 0.12\n'
    'BASE_REFERENCE_STRENGTH = 0.3\n'
    'BASE_REFERENCE_BLUR = 16.0\n'
    'BASE_REFERENCE_GRAIN_STRENGTH = 0.035  # Normalized monochrome noise sigma; 0 disables.\n'
    'REFERENCE_BACKGROUND = (116, 105, 91)\n'
    'SAVE_SOFT_REFERENCES = True  # Inspect in base_frames/soft_references and its preview sheet.\n',
    '# Strong trajectory initialization for independently generated anchors.\n'
    'TRAJECTORY_ZIP_DIRECTORY = "/content/drive/MyDrive/FluxFlowMorphArt/trajectory_inputs"\n'
    'TRAJECTORY_ZIP_FILENAME = "base_frames.zip"\n'
    'TRAJECTORY_MEMBER_PREFIX = ""  # Optional folder inside the ZIP, without leading slash.\n'
    'TRAJECTORY_FRAME_OFFSET = 0  # Rotate the naturally sorted trajectory before sampling.\n'
    'TRAJECTORY_REVERSE_ORDER = False\n'
    'TRAJECTORY_INIT_DETAIL_STRENGTH = 1.0  # 1 keeps the selected frame intact.\n'
    'TRAJECTORY_INIT_BLUR = 0.0\n'
    'TRAJECTORY_INIT_GRAIN_STRENGTH = 0.0\n'
    'TRAJECTORY_DENOISE_STRENGTH = 0.12  # Lower preserves more source composition.\n'
    'TRAJECTORY_COMPOSITION_INSTRUCTION = (\n'
    '    "Use the reference image as a strict spatial map: preserve the placement, "\n'
    '    "scale, silhouette, occupied areas, and empty areas; do not recenter or symmetrize."\n'
    ')\n'
    'TRAJECTORY_REMOVE_SYMMETRY_LANGUAGE = True  # Prevent prompt text from fighting the init layout.\n'
    'SAVE_TRAJECTORY_REFERENCES = True\n',
)
notebook["cells"][2]["source"] = settings.splitlines(keepends=True)

prompts = source(notebook["cells"][4])
prompts = replace_once(
    prompts,
    "\nBASE_PROMPT_COUNT = len(BASE_STAGES)\n",
    "\n# BASE_PROMPT_COUNT is set in the editable settings cell above.\n",
)
notebook["cells"][4]["source"] = prompts.splitlines(keepends=True)

repository_setup = source(notebook["cells"][6])
repository_setup = replace_once(
    repository_setup,
    "importlib.invalidate_caches()\n"
    "import flowmorph_klein\n"
    "from diffusers import Flux2KleinPipeline\n",
    "importlib.invalidate_caches()\n"
    "import flowmorph_klein\n"
    'trajectory_module = sys.modules.get("flowmorph_klein.trajectory")\n'
    "if trajectory_module is not None:\n"
    "    trajectory_module = importlib.reload(trajectory_module)\n"
    "from flowmorph_klein.trajectory import prepare_flux2_klein_img2img_inputs\n"
    "from diffusers import Flux2KleinPipeline\n",
)
notebook["cells"][6]["source"] = repository_setup.splitlines(keepends=True)

notebook["cells"][9]["source"] = lines(
    """
    ## 5. Validate settings, stage the trajectory ZIP, and preview the recursive cost

    The archive is copied from Drive to local Colab storage once, validated, naturally
    sorted, and sampled according to the number of active prompts. The selected,
    normalized source frames and their manifest are written directly into this run.
    """
)

validation = source(notebook["cells"][10])
validation = replace_once(
    validation,
    'if not 3 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):\n'
    '    raise ValueError(f"BASE_PROMPT_COUNT must be between 3 and {len(BASE_STAGES)}")\n',
    'ACTIVE_BASE_STAGES = (\n'
    '    list(BASE_STAGES)\n'
    '    if BASE_PROMPT_COUNT is None\n'
    '    else list(BASE_STAGES[:max(0, int(BASE_PROMPT_COUNT))])\n'
    ')\n'
    'if not ACTIVE_BASE_STAGES:\n'
    '    raise ValueError("BASE_STAGES must contain at least one active prompt")\n'
    'BASE_PROMPT_COUNT = len(ACTIVE_BASE_STAGES)\n',
)
validation = replace_once(
    validation,
    'if not 0 < BASE_REFERENCE_STRENGTH <= 1.0:\n'
    '    raise ValueError("BASE_REFERENCE_STRENGTH must lie in (0, 1]")\n'
    'if not 0 <= BASE_REFERENCE_GRAIN_STRENGTH <= 0.25:\n'
    '    raise ValueError("BASE_REFERENCE_GRAIN_STRENGTH must lie in [0, 0.25]")\n',
    'if not TRAJECTORY_ZIP_FILENAME.casefold().endswith(".zip"):\n'
    '    raise ValueError("TRAJECTORY_ZIP_FILENAME must name a .zip archive")\n'
    'if not isinstance(TRAJECTORY_FRAME_OFFSET, int):\n'
    '    raise ValueError("TRAJECTORY_FRAME_OFFSET must be an integer")\n'
    'if not 0 <= TRAJECTORY_INIT_DETAIL_STRENGTH <= 1:\n'
    '    raise ValueError("TRAJECTORY_INIT_DETAIL_STRENGTH must lie in [0, 1]")\n'
    'if TRAJECTORY_INIT_BLUR < 0:\n'
    '    raise ValueError("TRAJECTORY_INIT_BLUR cannot be negative")\n'
    'if not 0 <= TRAJECTORY_INIT_GRAIN_STRENGTH <= 0.25:\n'
    '    raise ValueError("TRAJECTORY_INIT_GRAIN_STRENGTH must lie in [0, 0.25]")\n'
    'if not 0 < TRAJECTORY_DENOISE_STRENGTH <= 1:\n'
    '    raise ValueError("TRAJECTORY_DENOISE_STRENGTH must lie in (0, 1]")\n'
    '\n'
    'def trajectory_generation_prompt(prompt):\n'
    '    base = prompt\n'
    '    if TRAJECTORY_REMOVE_SYMMETRY_LANGUAGE:\n'
    '        base = re.sub(r"\\bsymmetr(?:ical|ically)\\b", "", base, flags=re.IGNORECASE)\n'
    '    return " ".join(base.split()) + " " + TRAJECTORY_COMPOSITION_INSTRUCTION\n',
)
validation = replace_once(
    validation,
    "ACTIVE_BASE_STAGES = BASE_STAGES[:BASE_PROMPT_COUNT]\n",
    "# ACTIVE_BASE_STAGES was resolved above without an upper prompt-count check.\n",
)
staging = r'''

import hashlib
import shutil
from flowmorph_klein.trajectory import stage_regular_keyframes

TRAJECTORY_DRIVE_ARCHIVE = (
    Path(TRAJECTORY_ZIP_DIRECTORY).expanduser() / TRAJECTORY_ZIP_FILENAME
)
if not TRAJECTORY_DRIVE_ARCHIVE.is_file():
    raise FileNotFoundError(
        f"Trajectory ZIP was not found: {TRAJECTORY_DRIVE_ARCHIVE}. "
        "Upload it to Drive or edit the two trajectory path settings."
    )
trajectory_cache = Path(LOCAL_ASSET_ROOT) / PROJECT_NAME / "trajectory_archive"
trajectory_cache.mkdir(parents=True, exist_ok=True)
TRAJECTORY_LOCAL_ARCHIVE = trajectory_cache / TRAJECTORY_ZIP_FILENAME
copy_state_path = trajectory_cache / "copy_state.json"
drive_stat = TRAJECTORY_DRIVE_ARCHIVE.stat()
drive_signature = {
    "source": str(TRAJECTORY_DRIVE_ARCHIVE),
    "size": drive_stat.st_size,
    "mtime_ns": drive_stat.st_mtime_ns,
}
saved_copy_state = (
    json.loads(copy_state_path.read_text(encoding="utf-8"))
    if copy_state_path.is_file()
    else None
)
if (
    not TRAJECTORY_LOCAL_ARCHIVE.is_file()
    or saved_copy_state != drive_signature
    or TRAJECTORY_LOCAL_ARCHIVE.stat().st_size != drive_stat.st_size
):
    print("Copying trajectory ZIP from Drive to local Colab storage...")
    shutil.copy2(TRAJECTORY_DRIVE_ARCHIVE, TRAJECTORY_LOCAL_ARCHIVE)
    copy_state_path.write_text(
        json.dumps(drive_signature, indent=2) + "\n",
        encoding="utf-8",
    )
else:
    print("Reusing the matching local trajectory ZIP copy.")

archive_hasher = hashlib.sha256()
with TRAJECTORY_LOCAL_ARCHIVE.open("rb") as archive_handle:
    for archive_chunk in iter(lambda: archive_handle.read(1024 * 1024), b""):
        archive_hasher.update(archive_chunk)
TRAJECTORY_ARCHIVE_SHA256 = archive_hasher.hexdigest()
BASE_DIRECTORY = RUN_DIRECTORY / "base_frames"
TRAJECTORY_SOURCE_DIRECTORY = BASE_DIRECTORY / "trajectory_sources"
TRAJECTORY_RECORDS = list(stage_regular_keyframes(
    TRAJECTORY_LOCAL_ARCHIVE,
    TRAJECTORY_SOURCE_DIRECTORY,
    keyframe_count=len(ACTIVE_BASE_STAGES),
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
    member_prefix=TRAJECTORY_MEMBER_PREFIX,
    frame_offset=TRAJECTORY_FRAME_OFFSET,
    reverse_order=TRAJECTORY_REVERSE_ORDER,
))
for trajectory_record, stage in zip(TRAJECTORY_RECORDS, ACTIVE_BASE_STAGES, strict=True):
    trajectory_record["stage_id"] = stage["id"]
    trajectory_record["science"] = stage["science"]
    trajectory_record["prompt"] = stage["prompt"]

TRAJECTORY_SELECTION_MANIFEST = (
    RUN_DIRECTORY / "metadata" / "trajectory_selection.json"
)
TRAJECTORY_SELECTION_MANIFEST.write_text(json.dumps({
    "archive_drive_path": str(TRAJECTORY_DRIVE_ARCHIVE),
    "archive_local_path": str(TRAJECTORY_LOCAL_ARCHIVE),
    "archive_sha256": TRAJECTORY_ARCHIVE_SHA256,
    "selection_rule": "floor(keyframe_index * image_count / keyframe_count)",
    "member_prefix": TRAJECTORY_MEMBER_PREFIX,
    "frame_offset": TRAJECTORY_FRAME_OFFSET,
    "reverse_order": TRAJECTORY_REVERSE_ORDER,
    "keyframe_count": len(TRAJECTORY_RECORDS),
    "records": TRAJECTORY_RECORDS,
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print({
    "trajectory_archive": str(TRAJECTORY_DRIVE_ARCHIVE),
    "archive_sha256": TRAJECTORY_ARCHIVE_SHA256,
    "selected_keyframes": len(TRAJECTORY_RECORDS),
    "selected_indices": [
        record["trajectory_index"] for record in TRAJECTORY_RECORDS
    ],
    "selection_manifest": str(TRAJECTORY_SELECTION_MANIFEST),
})
'''
anchor_order_marker = 'print("Anchor order:", " → ".join(ids), "→", ids[0])\n'
validation = replace_once(
    validation,
    anchor_order_marker,
    anchor_order_marker + dedent(staging).lstrip("\n"),
)
notebook["cells"][10]["source"] = validation.splitlines(keepends=True)

notebook["cells"][11]["source"] = lines(
    """
    ## 6. Load and fuse RIJKSOIL; optional strong-init trial and FlowMorph test

    The optional trial now uses the selected trajectory frame corresponding to the
    chosen prompt. It displays both the actual conditioning image and the regenerated
    painting so trajectory adherence can be checked before producing all anchors.
    """
)
model_cell = source(notebook["cells"][12])
model_cell = replace_once(
    model_cell,
    "from flowmorph_klein.lora import load_flux2_lora\n",
    "from flowmorph_klein.lora import load_flux2_lora\n"
    "from flowmorph_klein.trajectory import (\n"
    "    make_strong_trajectory_reference,\n"
    "    prepare_flux2_klein_img2img_inputs,\n"
    ")\n",
)
trial_start = model_cell.index("if RUN_TRIAL_KEYFRAME:\n")
model_cell = model_cell[:trial_start] + dedent(
    r'''
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
            trial_reference = make_strong_trajectory_reference(
                opened,
                detail_strength=TRAJECTORY_INIT_DETAIL_STRENGTH,
                blur_radius=TRAJECTORY_INIT_BLUR,
                grain_strength=TRAJECTORY_INIT_GRAIN_STRENGTH,
                grain_seed=trial_seed,
            )
        trial_generator = torch.Generator(device="cuda").manual_seed(trial_seed)
        trial_img2img = prepare_flux2_klein_img2img_inputs(
            FLUX_PIPE,
            trial_reference,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=IMAGE_INFERENCE_STEPS,
            strength=TRAJECTORY_DENOISE_STRENGTH,
            generator=trial_generator,
        )
        trial_generation_prompt = trajectory_generation_prompt(trial_stage["prompt"])
        trial_result = FLUX_PIPE(
            prompt=trial_generation_prompt,
            image=trial_reference,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            num_inference_steps=IMAGE_INFERENCE_STEPS,
            sigmas=list(trial_img2img.sigmas),
            latents=trial_img2img.latents,
            guidance_scale=IMAGE_GUIDANCE_SCALE,
            generator=trial_generator,
            output_type="pil",
        )
        trial_image = trial_result.images[0].convert("RGB")
        trial_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        trial_directory = RUN_DIRECTORY / "trials" / f"{trial_stamp}_{trial_stage['id']}_{trial_seed}"
        trial_directory.mkdir(parents=True, exist_ok=False)
        trial_path = trial_directory / "trial.png"
        trial_reference_path = trial_directory / "trajectory_reference.png"
        trial_image.save(trial_path)
        trial_reference.save(trial_reference_path)
        (trial_directory / "settings.json").write_text(json.dumps({
            "stage": trial_stage,
            "trajectory": trial_trajectory,
            "seed": trial_seed,
            "lora_scale": IMAGE_LORA_SCALE,
            "guidance_scale": IMAGE_GUIDANCE_SCALE,
            "inference_steps": IMAGE_INFERENCE_STEPS,
            "size": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "trajectory_init_detail_strength": TRAJECTORY_INIT_DETAIL_STRENGTH,
            "trajectory_init_blur": TRAJECTORY_INIT_BLUR,
            "trajectory_init_grain_strength": TRAJECTORY_INIT_GRAIN_STRENGTH,
            "trajectory_denoise_strength": TRAJECTORY_DENOISE_STRENGTH,
            "effective_start_sigma": trial_img2img.effective_start_sigma,
            "effective_denoising_steps": trial_img2img.denoising_steps,
            "generation_prompt": trial_generation_prompt,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        input_preview = trial_reference.copy()
        output_preview = trial_image.copy()
        input_preview.thumbnail((TRIAL_DISPLAY_MAX_WIDTH, TRIAL_DISPLAY_MAX_WIDTH))
        output_preview.thumbnail((TRIAL_DISPLAY_MAX_WIDTH, TRIAL_DISPLAY_MAX_WIDTH))
        display(Markdown(f"### Trial trajectory input: `{trial_trajectory['member']}`"))
        display(input_preview)
        display(Markdown(f"### Trial regenerated anchor: `{trial_stage['id']}`"))
        display(output_preview)
        print({
            "path": str(trial_path),
            "trajectory_reference": str(trial_reference_path),
            "seed": trial_seed,
            "prompt_index": trial_index,
        })
        del trial_result, trial_img2img, trial_image, trial_reference, input_preview, output_preview
    else:
        print("Trial skipped.")
    '''
).lstrip("\n")
notebook["cells"][12]["source"] = model_cell.splitlines(keepends=True)

notebook["cells"][13]["source"] = lines(
    """
    ## 7. Generate trajectory-conditioned cyclic anchor paintings

    Each active science prompt is paired with one regularly sampled ZIP frame. The
    selected image—not the preceding generated anchor—is used as the strong init.
    Every generated anchor and its exact reference are saved immediately to Drive.
    """
)
notebook["cells"][14]["source"] = lines(
    r'''
    from flowmorph_klein.trajectory import (
        make_strong_trajectory_reference,
        prepare_flux2_klein_img2img_inputs,
    )

    BASE_DIRECTORY = RUN_DIRECTORY / "base_frames"
    REFERENCE_DIRECTORY = BASE_DIRECTORY / "trajectory_references"
    BASE_MANIFEST_PATH = RUN_DIRECTORY / "metadata" / "base_manifest.json"
    BASE_RECORDS = []
    expected_anchor_contract = [
        {
            "uid": f"base_{index:03d}",
            "science": stage["science"],
            "prompt": stage["prompt"],
            "generation_prompt": trajectory_generation_prompt(stage["prompt"]),
            "trajectory_member": trajectory["member"],
            "trajectory_member_sha256": trajectory["member_sha256"],
            "trajectory_denoise_strength": TRAJECTORY_DENOISE_STRENGTH,
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
                "trajectory_denoise_strength": record.get(
                    "trajectory_denoise_strength"
                ),
            }
            for record in BASE_RECORDS
        ]
        if (
            resumed_contract != expected_anchor_contract
            or saved_base_manifest.get("trajectory_archive_sha256")
            != TRAJECTORY_ARCHIVE_SHA256
        ):
            raise RuntimeError(
                "The prompts or trajectory archive differ from this run's saved anchors. "
                "Set REGENERATE_BASE_FRAMES=True or resume the matching run."
            )
        print(f"Loaded {len(BASE_RECORDS)} existing trajectory-conditioned anchors.")
    else:
        REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for index, (stage, trajectory) in enumerate(
            zip(ACTIVE_BASE_STAGES, TRAJECTORY_RECORDS, strict=True)
        ):
            seed = BASE_SEED + index
            with Image.open(trajectory["path"]) as opened:
                reference = make_strong_trajectory_reference(
                    opened,
                    detail_strength=TRAJECTORY_INIT_DETAIL_STRENGTH,
                    blur_radius=TRAJECTORY_INIT_BLUR,
                    grain_strength=TRAJECTORY_INIT_GRAIN_STRENGTH,
                    grain_seed=seed,
                )
            reference_path = REFERENCE_DIRECTORY / f"reference_{index:03d}.png"
            if SAVE_TRAJECTORY_REFERENCES:
                reference.save(reference_path, format="PNG", compress_level=4)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            img2img = prepare_flux2_klein_img2img_inputs(
                FLUX_PIPE,
                reference,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                num_inference_steps=IMAGE_INFERENCE_STEPS,
                strength=TRAJECTORY_DENOISE_STRENGTH,
                generator=generator,
            )
            generation_prompt = trajectory_generation_prompt(stage["prompt"])
            result = FLUX_PIPE(
                prompt=generation_prompt,
                image=reference,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                num_inference_steps=IMAGE_INFERENCE_STEPS,
                sigmas=list(img2img.sigmas),
                latents=img2img.latents,
                guidance_scale=IMAGE_GUIDANCE_SCALE,
                generator=generator,
                output_type="pil",
            )
            if not result.images:
                raise RuntimeError(f"FLUX returned no image for {stage['id']}")
            image = result.images[0].convert("RGB")
            output_path = BASE_DIRECTORY / f"{index:03d}_{stage['id']}.png"
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
                "trajectory_index": trajectory["trajectory_index"],
                "trajectory_member": trajectory["member"],
                "trajectory_member_sha256": trajectory["member_sha256"],
                "trajectory_source_path": trajectory["path"],
                "trajectory_reference_path": (
                    str(reference_path) if SAVE_TRAJECTORY_REFERENCES else None
                ),
                "trajectory_init_detail_strength": TRAJECTORY_INIT_DETAIL_STRENGTH,
                "trajectory_init_blur": TRAJECTORY_INIT_BLUR,
                "trajectory_init_grain_strength": TRAJECTORY_INIT_GRAIN_STRENGTH,
                "trajectory_denoise_strength": TRAJECTORY_DENOISE_STRENGTH,
                "effective_start_sigma": img2img.effective_start_sigma,
                "effective_denoising_steps": img2img.denoising_steps,
            }
            BASE_RECORDS.append(record)
            BASE_MANIFEST_PATH.write_text(json.dumps({
                "trajectory_archive_sha256": TRAJECTORY_ARCHIVE_SHA256,
                "trajectory_selection_manifest": str(TRAJECTORY_SELECTION_MANIFEST),
                "records": BASE_RECORDS,
                "complete": len(BASE_RECORDS) == len(ACTIVE_BASE_STAGES),
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(
                f"Anchor {index + 1}/{len(ACTIVE_BASE_STAGES)} saved: "
                f"{output_path.name} ← {trajectory['member']}"
            )
            reference.close()
            image.close()
            del result, img2img

    if len(BASE_RECORDS) != len(ACTIVE_BASE_STAGES):
        raise RuntimeError("The anchor manifest is incomplete; regenerate or resume the correct run.")
    print(f"Prepared {len(BASE_RECORDS)} trajectory-conditioned anchors in {BASE_DIRECTORY}")
    '''
)
notebook["cells"][15]["source"] = lines(
    r'''
    from flowmorph_klein.visualization import make_contact_sheet

    trajectory_source_paths = [Path(item["trajectory_source_path"]) for item in BASE_RECORDS]
    generated_anchor_paths = [Path(item["path"]) for item in BASE_RECORDS]

    def load_contact_thumbnails(paths, size=192):
        images = []
        for path in paths:
            with Image.open(path) as opened:
                thumbnail = opened.convert("RGB")
                thumbnail.thumbnail((size, size))
                images.append(thumbnail)
        return images

    source_contact_sheet_path = RUN_DIRECTORY / "previews" / "trajectory_sources.png"
    source_images = load_contact_thumbnails(trajectory_source_paths)
    make_contact_sheet(
        source_images,
        source_contact_sheet_path,
        columns=min(CONTACT_SHEET_COLUMNS, len(source_images)),
        labels=[
            f"{index:02d} · src {record['trajectory_index']}"
            for index, record in enumerate(BASE_RECORDS)
        ],
    )
    for image in source_images:
        image.close()

    base_contact_sheet_path = RUN_DIRECTORY / "previews" / "base_contact_sheet.png"
    base_images = load_contact_thumbnails(generated_anchor_paths)
    make_contact_sheet(
        base_images,
        base_contact_sheet_path,
        columns=min(CONTACT_SHEET_COLUMNS, len(base_images)),
        labels=[item["uid"] for item in BASE_RECORDS],
    )
    for image in base_images:
        image.close()

    paired_contact_sheet_path = RUN_DIRECTORY / "previews" / "trajectory_source_anchor_pairs.png"
    paired_images = []
    paired_labels = []
    for index, record in enumerate(BASE_RECORDS):
        paired_images.extend(load_contact_thumbnails([
            Path(record["trajectory_source_path"]),
            Path(record["path"]),
        ]))
        paired_labels.extend([f"{index:02d} source", f"{index:02d} generated"])
    make_contact_sheet(
        paired_images,
        paired_contact_sheet_path,
        columns=2,
        labels=paired_labels,
    )
    for image in paired_images:
        image.close()

    for heading, path in (
        ("### Regularly sampled trajectory inputs", source_contact_sheet_path),
        ("### Regenerated anchor paintings", base_contact_sheet_path),
        ("### Source → generated trajectory audit", paired_contact_sheet_path),
    ):
        preview = Image.open(path).convert("RGB")
        preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
        display(Markdown(heading))
        display(preview)
        preview.close()
    print({
        "full_resolution_sources": str(trajectory_source_paths[0].parent),
        "full_resolution_anchors": str(BASE_DIRECTORY),
        "paired_audit": str(paired_contact_sheet_path),
    })
    '''
)

notebook["cells"][22]["source"] = lines(
    """
    ## 10. Assemble, preview, and audit the generated cyclic trajectory-following sequence

    The regularly sampled ZIP frames determine the anchor trajectory. Round 1 contributes
    one explicit FlowMorph midpoint per anchor gap, Round 2 contributes ten piecewise
    source→shared-prompt→target renders, and the final-to-first gap closes the loop.
    """
)
final_video = source(notebook["cells"][30])
final_video = final_video.replace(
    '"recursive_flowmorph_vision_rife_ssim_loop.mp4"',
    '"recursive_flowmorph_trajectory_rife_ssim_loop.mp4"',
)
notebook["cells"][30]["source"] = final_video.splitlines(keepends=True)
notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} clean cells")
