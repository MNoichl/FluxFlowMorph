"""Replace the SAGE paper renderer with the project's FLUX.2 Klein renderer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "StillLife_SAGE_Transition_Video.ipynb"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))
# Retain editable settings/prompts/FLUX anchor generation by heading rather than
# a brittle numeric slice. The mask section below is canonical and explicit.
cells = []
for existing_cell in notebook["cells"]:
    existing_source = "".join(existing_cell.get("source", []))
    if existing_cell.get("cell_type") == "markdown" and existing_source.lstrip().startswith(
        "## 8. Build or load foreground masks"
    ):
        break
    if existing_cell.get("cell_type") == "code" and not existing_source.strip():
        continue
    cells.append(copy.deepcopy(existing_cell))

# Upgrade only the SAGE-specific defaults in the retained editable settings
# cell. Prompts, filenames, project names, and the user's anchor edits remain
# byte-for-byte as they were.
for retained_cell in cells:
    retained_source = "".join(retained_cell.get("source", []))
    if retained_source.startswith("# Prompt-only FLUX.2 anchors"):
        retained_source = retained_source.replace(
            "5. uses each condition, both endpoint paintings, interpolated endpoint\n"
            "   prompt embeddings, and a smooth previous-frame reference to render the\n"
            "   inbetweens with **the same FLUX.2 Klein 9B + RIJKSOIL LoRA pipeline**.",
            "5. densely warps both detailed endpoint paintings to each SAGE geometry; and\n"
            "6. refines those warps with **FLUX.2 Klein 9B + RIJKSOIL**, a compatible\n"
            "   FLUX.2 Canny/reference control LoRA, and a soft structural latent lock.",
        ).replace(
            "Everything else above\n"
            "remains the SAGE structural pipeline. FLUX replaces the paper's FCVG/SVD\n"
            "renderer because no FLUX.2 Klein ControlNet exists. Exact anchors are\n"
            "retained between gaps and the last gap returns to the first anchor.",
            "The structural frontend remains SAGE. FLUX replaces the paper's FCVG/SVD\n"
            "renderer; the control stage is FLUX.2-native rather than an incompatible\n"
            "FLUX.1 ControlNet. Exact anchors are retained between gaps and the last gap\n"
            "returns to the first anchor. RIFE only densifies the finished FLUX frames.",
        )
        retained_cell["source"] = retained_source.splitlines(keepends=True)
        continue
    if "SAGE_GENERATED_FRAMES_PER_GAP" not in retained_source:
        continue
    old_render_settings = '''# FLUX-native rendering of the SAGE guides. The guide is injected into a
# true img2img initialization; both endpoints and that initialization are
# also supplied as FLUX.2 reference images.
SAGE_FLUX_INFERENCE_STEPS = IMAGE_INFERENCE_STEPS
SAGE_FLUX_GUIDANCE_SCALE = IMAGE_GUIDANCE_SCALE
SAGE_FLUX_IMG2IMG_STRENGTH = 0.72
SAGE_ENDPOINT_PALETTE_BLUR = 64.0
SAGE_PREVIOUS_FRAME_BLEND = 0.24
SAGE_PREVIOUS_FRAME_BLUR = 10.0
SAGE_STRUCTURE_INIT_STRENGTH = 0.24
SAGE_STRUCTURE_DILATION_PIXELS = 3
SAGE_INIT_GRAIN_STRENGTH = 0.025
SAGE_REUSE_PROMPT_EMBEDDINGS = True
'''
    new_render_settings = '''# FLUX-only SAGE renderer. Detailed endpoint warps provide dense img2img
# structure. A FLUX.2 Klein RefControl Canny LoRA receives the SAGE line map
# and warped canvas; a soft latent lock retains those contours during denoising.
SAGE_FLUX_INFERENCE_STEPS = 40
SAGE_FLUX_GUIDANCE_SCALE = IMAGE_GUIDANCE_SCALE
SAGE_FLUX_IMG2IMG_STRENGTH = 0.48
SAGE_WARP_MAX_CONTROL_LINES = 96
SAGE_WARP_BORDER_SAMPLES = 7
SAGE_WARP_EASED_BLEND = True
SAGE_INIT_GRAIN_STRENGTH = 0.012
SAGE_STRUCTURE_LOCK_STRENGTH = 0.50
SAGE_STRUCTURE_LOCK_DILATION_PIXELS = 14
SAGE_STRUCTURE_LOCK_FEATHER_RADIUS = 4.0
SAGE_REFCONTROL_ENABLED = True
SAGE_REFCONTROL_SOURCE = "thedeoxen/refcontrol-FLUX.2-klein-9B-reference-canny-lora"
SAGE_REFCONTROL_REVISION = "be1fed28213ee9a176198b5d077727198e3cb4f4"
SAGE_REFCONTROL_WEIGHT_NAME = "flux2_klein_9b_refcontrol_canny.safetensors"
SAGE_REFCONTROL_ADAPTER_NAME = "sage_refcontrol"
SAGE_REFCONTROL_SCALE = 0.85
SAGE_REFCONTROL_TRIGGER = "refcontrol"
SAGE_REFCONTROL_CANNY_DILATION_PIXELS = 1
SAGE_REUSE_PROMPT_EMBEDDINGS = True

# Five expensive FLUX keyframes plus RIFE 2x finishing gives approximately
# the original temporal density at a fraction of the FLUX transformer work.
RUN_RIFE_POSTPROCESS = True
RIFE_REPOSITORY_URL = "https://github.com/hzwer/Practical-RIFE.git"
RIFE_REPOSITORY_REVISION = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
RIFE_ROOT = "/content/Practical-RIFE"
RIFE_MODEL_REPOSITORY = "Bash2X/RIFE-Models"
RIFE_MODEL_REVISION = "feaf6d11238b4a1e9f015a5d18c18df152affd20"
RIFE_MODEL_FILENAME = "RIFE_v4.25.zip"
RIFE_MULTIPLIER = 2
RIFE_SCALE = 1.0
RIFE_BATCH_SIZE = 4
RIFE_USE_FP16 = True
'''
    retained_source = retained_source.replace(
        "SAGE_GENERATED_FRAMES_PER_GAP = 13",
        "SAGE_GENERATED_FRAMES_PER_GAP = 5",
    )
    if old_render_settings in retained_source:
        retained_source = retained_source.replace(old_render_settings, new_render_settings)
    retained_source = retained_source.replace(
        'if not 0 <= SAGE_PREVIOUS_FRAME_BLEND <= 1:\n    raise ValueError("SAGE_PREVIOUS_FRAME_BLEND must lie in [0, 1]")\nif not 0 <= SAGE_STRUCTURE_INIT_STRENGTH <= 1:\n    raise ValueError("SAGE_STRUCTURE_INIT_STRENGTH must lie in [0, 1]")',
        'if not 0 <= SAGE_STRUCTURE_LOCK_STRENGTH <= 1:\n    raise ValueError("SAGE_STRUCTURE_LOCK_STRENGTH must lie in [0, 1]")\nif not 0 <= SAGE_REFCONTROL_SCALE <= 2:\n    raise ValueError("SAGE_REFCONTROL_SCALE must lie in [0, 2]")',
    )
    retained_cell["source"] = retained_source.splitlines(keepends=True)

cells.extend([
    markdown(
        """
        ## 8. Build or load foreground masks and inspect them

        SAGE deliberately suppresses background line clutter by matching only
        lines intersecting the foreground mask. White means foreground. Automatic
        GrabCut is a convenience, not ground truth: replace any bad mask PNG and
        rerun from section 9 with `SAGE_MASK_REGENERATE=False`.
        """
    ),
    code(
        """
        import cv2
        import numpy as np
        from PIL import ImageOps

        SAGE_MASK_DIRECTORY = RUN_DIRECTORY / "sage" / "masks"
        SAGE_MASK_DIRECTORY.mkdir(parents=True, exist_ok=True)
        SAGE_MASK_RECORDS = []

        def automatic_grabcut_mask(image):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            height, width = bgr.shape[:2]
            margin_x = max(1, round(width * SAGE_GRABCUT_MARGIN_FRACTION))
            margin_y = max(1, round(height * SAGE_GRABCUT_MARGIN_FRACTION))
            rectangle = (
                margin_x,
                margin_y,
                max(2, width - 2 * margin_x),
                max(2, height - 2 * margin_y),
            )
            labels = np.zeros((height, width), dtype=np.uint8)
            background_model = np.zeros((1, 65), dtype=np.float64)
            foreground_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(
                bgr,
                labels,
                rectangle,
                background_model,
                foreground_model,
                5,
                cv2.GC_INIT_WITH_RECT,
            )
            foreground = np.isin(
                labels, [cv2.GC_FGD, cv2.GC_PR_FGD]
            ).astype(np.uint8)
            if SAGE_MASK_DILATION_PIXELS > 0:
                size = int(SAGE_MASK_DILATION_PIXELS) * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
                foreground = cv2.dilate(foreground, kernel, iterations=1)
            coverage = float(foreground.mean())
            if not SAGE_MASK_MIN_COVERAGE <= coverage <= SAGE_MASK_MAX_COVERAGE:
                print(
                    f"GrabCut coverage {coverage:.3f} is outside the requested range; "
                    "using full frame for this anchor."
                )
                foreground = np.ones((height, width), dtype=np.uint8)
            return Image.fromarray(foreground * 255, mode="L")

        for record in BASE_RECORDS:
            output_path = SAGE_MASK_DIRECTORY / f"{record['uid']}.png"
            if output_path.is_file() and not SAGE_MASK_REGENERATE:
                mask = Image.open(output_path).convert("L")
                source = "existing_run_mask"
            elif SAGE_MASK_MODE == "directory":
                input_path = (
                    Path(SAGE_MASK_SOURCE_DIRECTORY).expanduser()
                    / f"{record['uid']}.png"
                )
                if not input_path.is_file():
                    raise FileNotFoundError(f"Missing SAGE mask: {input_path}")
                with Image.open(input_path) as opened:
                    mask = opened.convert("L")
                source = str(input_path)
            elif SAGE_MASK_MODE == "full_frame":
                with Image.open(record["path"]) as opened:
                    mask = Image.new("L", opened.size, 255)
                source = "full_frame"
            else:
                with Image.open(record["path"]) as opened:
                    mask = automatic_grabcut_mask(opened)
                source = "automatic_grabcut"
            mask.save(output_path)
            coverage = float((np.asarray(mask, dtype=np.uint8) >= 128).mean())
            mask.close()
            SAGE_MASK_RECORDS.append({
                "uid": record["uid"],
                "mask_path": str(output_path),
                "source": source,
                "coverage": coverage,
            })

        mask_by_uid = {item["uid"]: item for item in SAGE_MASK_RECORDS}
        SAGE_ANCHOR_MANIFEST_PATH = (
            RUN_DIRECTORY / "metadata" / "sage_anchor_manifest.json"
        )
        SAGE_ANCHOR_MANIFEST = {
            "cyclic": True,
            "mask_mode": SAGE_MASK_MODE,
            "anchors": [
                {
                    **record,
                    "mask_path": mask_by_uid[record["uid"]]["mask_path"],
                    "mask_coverage": mask_by_uid[record["uid"]]["coverage"],
                }
                for record in BASE_RECORDS
            ],
        }
        SAGE_ANCHOR_MANIFEST_PATH.write_text(
            json.dumps(SAGE_ANCHOR_MANIFEST, indent=2, ensure_ascii=False) + "\\n",
            encoding="utf-8",
        )

        mask_contact_sheet_path = (
            RUN_DIRECTORY / "previews" / "sage_foreground_masks.png"
        )
        mask_previews = []
        for item in SAGE_MASK_RECORDS:
            with Image.open(item["mask_path"]) as opened:
                mask_previews.append(opened.convert("RGB"))
        make_contact_sheet(
            mask_previews,
            mask_contact_sheet_path,
            columns=min(CONTACT_SHEET_COLUMNS, len(mask_previews)),
            labels=[
                f"{item['uid']} ({item['coverage']:.0%})"
                for item in SAGE_MASK_RECORDS
            ],
        )
        for image in mask_previews:
            image.close()
        mask_preview = Image.open(mask_contact_sheet_path).convert("RGB")
        mask_preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
        display(Markdown(
            "### SAGE foreground masks — white structures will guide matching"
        ))
        display(mask_preview)
        mask_preview.close()
        print("Editable full-resolution masks:", SAGE_MASK_DIRECTORY)
        """
    ),
])

cells.extend([
    markdown(
        """
        ## 9. Install SAGE's structural frontend while retaining FLUX

        SAGE contributes GlueStick matching and spline-propagated line guides.
        The paper's Stable Video Diffusion/FCVG renderer is deliberately not
        installed or downloaded. `FLUX_PIPE` remains the sole image renderer,
        with the already fused RIJKSOIL LoRA.
        """
    ),
    code(
        """
        import gc
        import hashlib
        import shutil
        import urllib.request

        sage_import_probe = subprocess.run(
            [sys.executable, "-c", "import omegaconf, pytlsd; from pytlsd import lsd"],
            capture_output=True,
            text=True,
        )
        if sage_import_probe.returncode != 0:
            print("Repairing missing SAGE line-detector dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade",
                "setuptools>=69", "wheel", "pybind11>=2.10",
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "omegaconf==2.3.0", "pytlsd==0.0.2",
            ])
            import importlib
            importlib.invalidate_caches()
            subprocess.check_call([
                sys.executable, "-c",
                "import omegaconf, pytlsd; from pytlsd import lsd; print(pytlsd.__file__)",
            ])

        maybe_free = getattr(FLUX_PIPE, "maybe_free_model_hooks", None)
        if callable(maybe_free):
            maybe_free()
        gc.collect()
        torch.cuda.empty_cache()

        sage_repository = Path(SAGE_REPOSITORY_DIRECTORY)
        if not (sage_repository / ".git").is_dir():
            subprocess.check_call(["git", "clone", SAGE_REPOSITORY_URL, str(sage_repository)])
        subprocess.check_call([
            "git", "-C", str(sage_repository), "fetch", "origin", SAGE_REPOSITORY_COMMIT,
        ])
        subprocess.check_call([
            "git", "-C", str(sage_repository), "checkout", "--detach", SAGE_REPOSITORY_COMMIT,
        ])
        actual_sage_commit = subprocess.check_output(
            ["git", "-C", str(sage_repository), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_sage_commit != SAGE_REPOSITORY_COMMIT:
            raise RuntimeError("SAGE checkout does not match the pinned paper implementation")

        # SAGE vendors GlueStick's Python sources but omits the separate
        # SuperPoint detector checkpoint expected by its hard-coded relative path.
        SAGE_SUPERPOINT_CHECKPOINT = (
            sage_repository / "models" / "resources" / "weights" / "superpoint_v1.pth"
        )
        SAGE_SUPERPOINT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

        def local_sha256(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        superpoint_is_valid = (
            SAGE_SUPERPOINT_CHECKPOINT.is_file()
            and local_sha256(SAGE_SUPERPOINT_CHECKPOINT) == SAGE_SUPERPOINT_SHA256
        )
        if not superpoint_is_valid:
            temporary_path = SAGE_SUPERPOINT_CHECKPOINT.with_suffix(".download")
            urllib.request.urlretrieve(SAGE_SUPERPOINT_URL, temporary_path)
            downloaded_sha256 = local_sha256(temporary_path)
            if downloaded_sha256 != SAGE_SUPERPOINT_SHA256:
                raise RuntimeError(
                    "SuperPoint checkpoint checksum mismatch: "
                    f"found {downloaded_sha256}, expected {SAGE_SUPERPOINT_SHA256}"
                )
            temporary_path.replace(SAGE_SUPERPOINT_CHECKPOINT)
        print({
            "superpoint_checkpoint": str(SAGE_SUPERPOINT_CHECKPOINT),
            "superpoint_sha256": local_sha256(SAGE_SUPERPOINT_CHECKPOINT),
        })

        checkpoint_root = Path(HF_CACHE_DIR) / "sage_gluestick"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        SAGE_GLUESTICK_CHECKPOINT = checkpoint_root / "checkpoint_GlueStick_MD.tar"
        if not SAGE_GLUESTICK_CHECKPOINT.is_file():
            temporary_path = SAGE_GLUESTICK_CHECKPOINT.with_suffix(".download")
            urllib.request.urlretrieve(SAGE_GLUESTICK_URL, temporary_path)
            temporary_path.replace(SAGE_GLUESTICK_CHECKPOINT)

        print({
            "sage_commit": actual_sage_commit,
            "sage_component": "GlueStick + normalized matching + spline line propagation",
            "renderer": MODEL_ID,
            "renderer_lora": LORA_SOURCE,
            "lora_scale": IMAGE_LORA_SCALE,
            "stable_video_diffusion_downloaded": False,
            "cuda_reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        })
        """
    ),
    markdown(
        """
        ## 10. Prepare and inspect SAGE structural guides

        This inexpensive subprocess loads GlueStick once, matches each circular
        pair, and writes the true interior guide sequence. It does not load a
        second generative model. Inspect the overlays and middle conditions
        before starting the FLUX render.
        """
    ),
    code(
        """
        import os

        SAGE_OUTPUT_ROOT = RUN_DIRECTORY / "sage" / "one_round"
        SAGE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        SAGE_RUNNER = Path(PROJECT_ROOT) / "scripts" / "sage_still_sequence_runner.py"
        if not SAGE_RUNNER.is_file():
            raise FileNotFoundError(SAGE_RUNNER)

        SAGE_COMMAND = [
            sys.executable,
            str(SAGE_RUNNER),
            "--sage-repo", str(sage_repository),
            "--manifest", str(SAGE_ANCHOR_MANIFEST_PATH),
            "--output-root", str(SAGE_OUTPUT_ROOT),
            "--gluestick-checkpoint", str(SAGE_GLUESTICK_CHECKPOINT),
            "--width", str(SAGE_WIDTH),
            "--height", str(SAGE_HEIGHT),
            "--generated-frames", str(SAGE_GENERATED_FRAMES_PER_GAP),
            "--max-points", str(SAGE_MAX_POINTS),
            "--max-lines", str(SAGE_MAX_LINES),
            "--max-matched-lines", str(SAGE_MAX_MATCHED_LINES),
            "--minimum-matched-lines", str(SAGE_MINIMUM_MATCHED_LINES),
            "--line-width", str(SAGE_CONDITION_LINE_WIDTH),
            "--trajectory-bend", str(SAGE_TRAJECTORY_BEND),
            "--synthetic-flow-scale", str(SAGE_SYNTHETIC_FLOW_SCALE),
        ]
        if SAGE_REUSE_COMPLETED_GAPS:
            SAGE_COMMAND.append("--reuse")

        if callable(maybe_free):
            maybe_free()
        gc.collect()
        torch.cuda.empty_cache()
        SAGE_PREPARATION_LOG_PATH = (
            RUN_DIRECTORY / "metadata" / "sage_structure_preparation.log"
        )
        print("SAGE subprocess log:", SAGE_PREPARATION_LOG_PATH)
        sage_environment = os.environ.copy()
        sage_environment["PYTHONUNBUFFERED"] = "1"
        with SAGE_PREPARATION_LOG_PATH.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                SAGE_COMMAND,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=sage_environment,
            )
            assert process.stdout is not None
            captured_lines = []
            for output_line in process.stdout:
                print(output_line, end="")
                log_handle.write(output_line)
                log_handle.flush()
                captured_lines.append(output_line)
                if len(captured_lines) > 120:
                    captured_lines.pop(0)
            sage_return_code = process.wait()
        if sage_return_code != 0:
            failure_tail = "".join(captured_lines[-80:])
            raise RuntimeError(
                "SAGE structural preparation failed. The actual subprocess "
                f"exception is preserved at {SAGE_PREPARATION_LOG_PATH}.\\n\\n"
                "Last subprocess lines:\\n" + failure_tail
            )

        SAGE_PREPARATION_MANIFEST_PATH = SAGE_OUTPUT_ROOT / "sage_preparation_manifest.json"
        SAGE_PREPARATION = json.loads(
            SAGE_PREPARATION_MANIFEST_PATH.read_text(encoding="utf-8")
        )

        def display_path_contact_sheet(paths, labels, title, filename):
            images = [Image.open(path).convert("RGB") for path in paths]
            output_path = RUN_DIRECTORY / "previews" / filename
            make_contact_sheet(
                images,
                output_path,
                columns=min(CONTACT_SHEET_COLUMNS, len(images)),
                labels=labels,
            )
            for image in images:
                image.close()
            preview = Image.open(output_path).convert("RGB")
            preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
            display(Markdown(title))
            display(preview)
            preview.close()
            return output_path

        line_preview_paths = []
        condition_preview_paths = []
        for gap in SAGE_PREPARATION["gaps"]:
            gap_directory = Path(gap["gap_directory"])
            line_preview_paths.extend([
                gap_directory / "source_matched_lines.png",
                gap_directory / "target_matched_lines.png",
            ])
            condition_paths = [Path(path) for path in gap["condition_paths"]]
            condition_preview_paths.append(condition_paths[len(condition_paths) // 2])

        display_path_contact_sheet(
            line_preview_paths,
            [path.parent.name + "/" + path.stem for path in line_preview_paths],
            "### Matched foreground lines at both sides of every gap",
            "sage_matched_line_overlays.png",
        )
        display_path_contact_sheet(
            condition_preview_paths,
            [path.parent.parent.name for path in condition_preview_paths],
            "### Middle SAGE structural guide in every gap",
            "sage_middle_conditions.png",
        )
        print("Prepared SAGE guides:", SAGE_OUTPUT_ROOT)
        """
    ),
    markdown(
        """
        ## 11. Render the SAGE-guided round with FLUX.2 Klein + RIJKSOIL

        For every interior time step, the notebook builds a soft initialization
        from low-frequency endpoint color, the previous generated frame, seeded
        grain, and the current SAGE line guide. That initialization supplies true
        FLUX img2img latents. Both exact endpoints and the initialization are also
        passed as FLUX.2 reference images, while endpoint prompt embeddings are
        interpolated at the same time coordinate. Frames and metadata are saved
        immediately, and completed fingerprint-matching gaps are reusable.
        """
    ),
    code(
        """
        import hashlib
        from PIL import ImageEnhance, ImageOps
        from flowmorph_klein.conditioning import (
            encode_prompt_conditioning,
            interpolate_conditioning,
        )
        from flowmorph_klein.trajectory import prepare_flux2_klein_img2img_inputs

        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        def write_json_atomic(path, payload):
            path = Path(path)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\\n",
                encoding="utf-8",
            )
            temporary.replace(path)

        def fit_frame(value):
            if isinstance(value, Image.Image):
                opened = value.convert("RGB")
            else:
                with Image.open(value) as source:
                    opened = source.convert("RGB")
            fitted = ImageOps.fit(
                opened,
                (SAGE_WIDTH, SAGE_HEIGHT),
                method=Image.Resampling.LANCZOS,
            )
            if opened is not value:
                opened.close()
            return fitted

        def make_sage_flux_init(left, right, condition, previous, alpha, seed):
            left_low = left.filter(ImageFilter.GaussianBlur(SAGE_ENDPOINT_PALETTE_BLUR))
            right_low = right.filter(ImageFilter.GaussianBlur(SAGE_ENDPOINT_PALETTE_BLUR))
            canvas = Image.blend(left_low, right_low, float(alpha))
            left_low.close()
            right_low.close()

            if previous is not None and SAGE_PREVIOUS_FRAME_BLEND > 0:
                previous_low = previous.filter(
                    ImageFilter.GaussianBlur(SAGE_PREVIOUS_FRAME_BLUR)
                )
                canvas = Image.blend(canvas, previous_low, SAGE_PREVIOUS_FRAME_BLEND)
                previous_low.close()

            guide = condition.convert("L")
            if SAGE_STRUCTURE_DILATION_PIXELS > 0:
                kernel = SAGE_STRUCTURE_DILATION_PIXELS * 2 + 1
                guide = guide.filter(ImageFilter.MaxFilter(kernel))
            guide = guide.filter(ImageFilter.GaussianBlur(0.8))
            guide = guide.point(
                lambda value: int(round(value * SAGE_STRUCTURE_INIT_STRENGTH))
            )
            line_layer = ImageEnhance.Brightness(canvas).enhance(0.58)
            structured = Image.composite(line_layer, canvas, guide)
            line_layer.close()
            guide.close()
            canvas.close()

            if SAGE_INIT_GRAIN_STRENGTH > 0:
                array = np.asarray(structured, dtype=np.float32)
                rng = np.random.default_rng(seed)
                noise = rng.normal(
                    0.0,
                    255.0 * SAGE_INIT_GRAIN_STRENGTH,
                    size=(SAGE_HEIGHT, SAGE_WIDTH, 1),
                )
                array = np.clip(array + noise, 0, 255).astype(np.uint8)
                structured.close()
                structured = Image.fromarray(array, mode="RGB")
            return structured

        def ffmpeg_frames(input_directory, output_path):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(SAGE_OUTPUT_FPS),
                "-i", str(Path(input_directory) / "frame_%04d.png"),
                "-c:v", "libx264", "-preset", "slow", "-crf", str(SAGE_VIDEO_CRF),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
            ])

        RENDER_CONTRACT = {
            "method": "SAGE structure + FLUX.2 Klein img2img/reference rendering",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "lora_source": LORA_SOURCE,
            "lora_revision": LORA_REVISION,
            "lora_weight_name": LORA_WEIGHT_NAME,
            "lora_scale": IMAGE_LORA_SCALE,
            "inference_steps": SAGE_FLUX_INFERENCE_STEPS,
            "guidance_scale": SAGE_FLUX_GUIDANCE_SCALE,
            "img2img_strength": SAGE_FLUX_IMG2IMG_STRENGTH,
            "endpoint_palette_blur": SAGE_ENDPOINT_PALETTE_BLUR,
            "previous_frame_blend": SAGE_PREVIOUS_FRAME_BLEND,
            "previous_frame_blur": SAGE_PREVIOUS_FRAME_BLUR,
            "structure_init_strength": SAGE_STRUCTURE_INIT_STRENGTH,
            "structure_dilation_pixels": SAGE_STRUCTURE_DILATION_PIXELS,
            "grain_strength": SAGE_INIT_GRAIN_STRENGTH,
            "prompt_interpolation": "linear endpoint embedding interpolation",
            "reference_images": "left endpoint + right endpoint + per-frame SAGE init",
        }
        RENDER_CONTRACT_HASH = hashlib.sha256(
            json.dumps(RENDER_CONTRACT, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if RUN_SAGE_RENDER:
            print("Encoding each unique endpoint prompt once...")
            ANCHOR_CONDITIONING = {}
            for record in BASE_RECORDS:
                ANCHOR_CONDITIONING[record["uid"]] = encode_prompt_conditioning(
                    FLUX_PIPE,
                    record["prompt"],
                    device=FLUX_PIPE._execution_device,
                    max_sequence_length=FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
                ).cpu()
            if callable(maybe_free):
                maybe_free()
            gc.collect()
            torch.cuda.empty_cache()

            COMPLETED_SAGE_GAPS = []
            for gap in SAGE_PREPARATION["gaps"]:
                gap_directory = Path(gap["gap_directory"])
                rendered_directory = gap_directory / "flux_rendered"
                init_directory = gap_directory / "flux_inits"
                complete_directory = gap_directory / "flux_complete"
                for directory in (rendered_directory, init_directory, complete_directory):
                    directory.mkdir(parents=True, exist_ok=True)
                metadata_path = gap_directory / "flux_render_metadata.json"

                left_record = gap["left"]
                right_record = gap["right"]
                fingerprint = hashlib.sha256(json.dumps({
                    "render_contract_hash": RENDER_CONTRACT_HASH,
                    "sage_preparation_contract_hash": SAGE_PREPARATION["contract_hash"],
                    "gap_uid": gap["gap_uid"],
                    "endpoint_hashes": gap["endpoint_hashes"],
                    "left_prompt": left_record["prompt"],
                    "right_prompt": right_record["prompt"],
                    "condition_hashes": [sha256_file(path) for path in gap["condition_paths"]],
                }, sort_keys=True).encode("utf-8")).hexdigest()

                saved = {}
                if metadata_path.is_file():
                    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                saved_paths = [Path(path) for path in saved.get("rendered_frame_paths", [])]
                if not (
                    SAGE_REUSE_COMPLETED_GAPS
                    and saved.get("fingerprint") == fingerprint
                    and saved.get("complete")
                    and len(saved_paths) == SAGE_GENERATED_FRAMES_PER_GAP
                    and all(path.is_file() for path in saved_paths)
                ):
                    saved_paths = []

                left_image = fit_frame(left_record["path"])
                right_image = fit_frame(right_record["path"])
                previous_image = left_image.copy()
                rendered_paths = []
                alphas = gap["condition_alphas"]
                if len(alphas) != SAGE_GENERATED_FRAMES_PER_GAP:
                    raise RuntimeError("SAGE guide count does not match configured interior frames")

                for frame_index, (alpha, condition_path) in enumerate(
                    zip(alphas, gap["condition_paths"])
                ):
                    output_path = rendered_directory / f"flux_sage_{frame_index:04d}.png"
                    init_path = init_directory / f"init_{frame_index:04d}.png"
                    if frame_index < len(saved_paths) and output_path.is_file():
                        previous_image.close()
                        previous_image = fit_frame(output_path)
                        rendered_paths.append(str(output_path))
                        continue

                    with Image.open(condition_path) as opened_condition:
                        condition_image = fit_frame(opened_condition)
                    frame_seed = BASE_SEED + 500_000 + int(gap["gap_index"])
                    init_image = make_sage_flux_init(
                        left_image,
                        right_image,
                        condition_image,
                        previous_image,
                        float(alpha),
                        frame_seed + frame_index,
                    )
                    init_image.save(init_path, format="PNG", compress_level=4)
                    generator = torch.Generator(device="cuda").manual_seed(frame_seed)
                    img2img = prepare_flux2_klein_img2img_inputs(
                        FLUX_PIPE,
                        init_image,
                        width=SAGE_WIDTH,
                        height=SAGE_HEIGHT,
                        num_inference_steps=SAGE_FLUX_INFERENCE_STEPS,
                        strength=SAGE_FLUX_IMG2IMG_STRENGTH,
                        generator=generator,
                    )
                    prompt_package = interpolate_conditioning(
                        ANCHOR_CONDITIONING[left_record["uid"]],
                        ANCHOR_CONDITIONING[right_record["uid"]],
                        float(alpha),
                    ).to(FLUX_PIPE._execution_device, dtype=torch.bfloat16)
                    result = FLUX_PIPE(
                        image=[left_image, right_image, init_image],
                        prompt_embeds=prompt_package.prompt_embeds,
                        height=SAGE_HEIGHT,
                        width=SAGE_WIDTH,
                        num_inference_steps=SAGE_FLUX_INFERENCE_STEPS,
                        guidance_scale=SAGE_FLUX_GUIDANCE_SCALE,
                        generator=generator,
                        latents=img2img.latents,
                        sigmas=list(img2img.sigmas),
                        output_type="pil",
                        max_sequence_length=FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
                    )
                    generated = result.images[0].convert("RGB")
                    generated.save(output_path, format="PNG", compress_level=4)
                    rendered_paths.append(str(output_path))
                    previous_image.close()
                    previous_image = generated.copy()
                    generated.close()
                    init_image.close()
                    condition_image.close()
                    del result, img2img, prompt_package

                    partial = {
                        "gap_uid": gap["gap_uid"],
                        "gap_index": gap["gap_index"],
                        "left": left_record,
                        "right": right_record,
                        "fingerprint": fingerprint,
                        "render_contract": RENDER_CONTRACT,
                        "rendered_frame_paths": rendered_paths,
                        "complete": False,
                    }
                    write_json_atomic(metadata_path, partial)
                    print(
                        f"Saved FLUX SAGE frame {frame_index + 1}/"
                        f"{SAGE_GENERATED_FRAMES_PER_GAP} for {gap['gap_uid']}"
                    )

                complete_paths = []
                complete_images = [left_image]
                complete_images.extend(fit_frame(path) for path in rendered_paths)
                complete_images.append(right_image)
                for index, frame in enumerate(complete_images):
                    destination = complete_directory / f"frame_{index:04d}.png"
                    frame.save(destination, format="PNG", compress_level=4)
                    complete_paths.append(str(destination))
                clip_path = gap_directory / "flux_sage_transition.mp4"
                ffmpeg_frames(complete_directory, clip_path)
                completed = {
                    "gap_uid": gap["gap_uid"],
                    "gap_index": gap["gap_index"],
                    "left": left_record,
                    "right": right_record,
                    "fingerprint": fingerprint,
                    "render_contract": RENDER_CONTRACT,
                    "condition_alphas": alphas,
                    "condition_paths": gap["condition_paths"],
                    "rendered_frame_paths": rendered_paths,
                    "complete_frame_paths": complete_paths,
                    "clip_path": str(clip_path),
                    "complete": True,
                }
                write_json_atomic(metadata_path, completed)
                COMPLETED_SAGE_GAPS.append(completed)
                for frame in complete_images:
                    frame.close()
                previous_image.close()
                print("Completed FLUX-rendered gap:", clip_path)

            sequence_directory = SAGE_OUTPUT_ROOT / "flux_cyclic_frames"
            if sequence_directory.exists():
                shutil.rmtree(sequence_directory)
            sequence_directory.mkdir(parents=True, exist_ok=False)
            sequence_paths = []
            sequence_index = 0
            for gap in COMPLETED_SAGE_GAPS:
                for path in [Path(value) for value in gap["complete_frame_paths"]][:-1]:
                    destination = sequence_directory / f"frame_{sequence_index:04d}.png"
                    shutil.copy2(path, destination)
                    sequence_paths.append(str(destination))
                    sequence_index += 1
            SAGE_FINAL_VIDEO_PATH = RUN_DIRECTORY / "video" / "sage_flux2_klein_cyclic.mp4"
            ffmpeg_frames(sequence_directory, SAGE_FINAL_VIDEO_PATH)
            SAGE_SEQUENCE = {
                "method": "SAGE structure rendered by FLUX.2 Klein + RIJKSOIL",
                "paper": "https://arxiv.org/abs/2510.24667v2",
                "cyclic": True,
                "renderer": MODEL_ID,
                "lora_source": LORA_SOURCE,
                "render_contract": RENDER_CONTRACT,
                "gaps": COMPLETED_SAGE_GAPS,
                "sequence_frame_paths": sequence_paths,
                "frame_count": len(sequence_paths),
                "fps": SAGE_OUTPUT_FPS,
                "final_video_path": str(SAGE_FINAL_VIDEO_PATH),
            }
            SAGE_SEQUENCE_MANIFEST_PATH = SAGE_OUTPUT_ROOT / "sage_sequence_manifest.json"
            write_json_atomic(SAGE_SEQUENCE_MANIFEST_PATH, SAGE_SEQUENCE)
        else:
            print("FLUX SAGE render intentionally stopped after guide inspection.")
        """
    ),
    markdown(
        """
        ## 12. Preview the FLUX-rendered SAGE result
        """
    ),
    code(
        """
        from IPython.display import Video

        if RUN_SAGE_RENDER:
            midpoint_paths = []
            for gap in SAGE_SEQUENCE["gaps"]:
                rendered = [Path(path) for path in gap["rendered_frame_paths"]]
                midpoint_paths.append(rendered[len(rendered) // 2])
                if SAGE_DISPLAY_EACH_GAP_VIDEO:
                    display(Markdown(
                        f"### `{gap['left']['uid']}` → `{gap['right']['uid']}`"
                    ))
                    display(Video(
                        str(gap["clip_path"]),
                        embed=False,
                        width=SAGE_VIDEO_DISPLAY_WIDTH,
                    ))

            display_path_contact_sheet(
                midpoint_paths,
                [gap["gap_uid"] for gap in SAGE_SEQUENCE["gaps"]],
                "### FLUX.2 Klein + RIJKSOIL midpoint from every SAGE gap",
                "sage_flux_generated_midpoints.png",
            )
            display(Markdown("## Final one-round cyclic SAGE/FLUX video"))
            display(Video(
                str(SAGE_FINAL_VIDEO_PATH),
                embed=False,
                width=SAGE_VIDEO_DISPLAY_WIDTH,
                html_attributes="controls loop muted",
            ))
            print({
                "final_video": str(SAGE_FINAL_VIDEO_PATH),
                "frames": SAGE_SEQUENCE["frame_count"],
                "fps": SAGE_SEQUENCE["fps"],
                "renderer": SAGE_SEQUENCE["renderer"],
                "lora": SAGE_SEQUENCE["lora_source"],
                "cyclic": SAGE_SEQUENCE["cyclic"],
            })
            if DOWNLOAD_FINAL_VIDEO:
                from google.colab import files
                files.download(str(SAGE_FINAL_VIDEO_PATH))
        """
    ),
    markdown(
        """
        ## Interpretation

        This is a SAGE **structure adaptation**, not a claim that FLUX.2 Klein
        natively implements the paper's FCVG renderer. SAGE determines the
        matched line topology and its spline trajectory. FLUX.2 Klein 9B with
        RIJKSOIL renders every visible interior image from those guides. Because
        the current FLUX ControlNet implementation targets FLUX.1 rather than
        FLUX.2 Klein, the guide enters through a saved, inspectable img2img
        initialization plus FLUX.2 multi-reference conditioning.
        """
    ),
])

# Replace the first FLUX rendering adaptation with the dense-warp/RefControl
# implementation. Keeping this override close to notebook serialization makes
# the builder idempotent even when it is run against an older generated file.
section_11_index = next(
    index
    for index, cell in enumerate(cells)
    if cell.get("cell_type") == "markdown"
    and "".join(cell.get("source", [])).lstrip().startswith("## 11.")
)
cells = cells[:section_11_index]
cells.extend([
    markdown(
        """
        ## 11. Render detailed SAGE warps with FLUX.2 Klein control

        The previous experimental renderer blurred both endpoints by 64 pixels
        and then autoregressively blurred the preceding result. That destroyed
        detail before FLUX ever saw it. This version instead warps both complete
        endpoint paintings to every interpolated SAGE geometry and blends those
        detailed warps.

        FLUX receives three coordinated controls: true img2img latents from the
        dense warp, a soft latent lock around the SAGE trajectories, and an
        optional FLUX.2 Klein RefControl Canny LoRA whose first reference is the
        black/white SAGE structure and whose second is the warped painting. The
        RIJKSOIL adapter remains fused in the base model. No Stable Diffusion,
        SVD, FCVG, or FLUX.1 component is loaded.
        """
    ),
    code(
        """
        import hashlib
        from huggingface_hub import hf_hub_download
        from PIL import ImageOps
        from flowmorph_klein.conditioning import (
            encode_prompt_conditioning,
            interpolate_conditioning,
        )
        from flowmorph_klein.sage_control import (
            make_canny_reference,
            make_structure_lock_mask,
            warp_sage_endpoints,
        )
        from flowmorph_klein.trajectory import prepare_flux2_klein_spatial_lock_inputs

        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        def write_json_atomic(path, payload):
            path = Path(path)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\\n",
                encoding="utf-8",
            )
            temporary.replace(path)

        def fit_frame(value):
            if isinstance(value, Image.Image):
                source = value.convert("RGB")
            else:
                with Image.open(value) as opened:
                    source = opened.convert("RGB")
            fitted = ImageOps.fit(
                source,
                (SAGE_WIDTH, SAGE_HEIGHT),
                method=Image.Resampling.LANCZOS,
            )
            source.close()
            return fitted

        def ffmpeg_frames(input_directory, output_path, fps=None):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps or SAGE_OUTPUT_FPS),
                "-i", str(Path(input_directory) / "frame_%04d.png"),
                "-c:v", "libx264", "-preset", "slow", "-crf", str(SAGE_VIDEO_CRF),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
            ])

        if "FLUX_PIPE" not in globals():
            FLUX_PIPE, LORA_REPORT = load_flux_pipeline()
            FLUX_PIPE_LORA_SCALE = float(IMAGE_LORA_SCALE)

        REFCONTROL_LOCAL_PATH = None
        if SAGE_REFCONTROL_ENABLED:
            REFCONTROL_LOCAL_PATH = Path(hf_hub_download(
                repo_id=SAGE_REFCONTROL_SOURCE,
                filename=SAGE_REFCONTROL_WEIGHT_NAME,
                revision=SAGE_REFCONTROL_REVISION,
                cache_dir=HF_CACHE_DIR,
            ))
            listed = FLUX_PIPE.get_list_adapters() or {}
            transformer_adapters = set(listed.get("transformer", []))
            if SAGE_REFCONTROL_ADAPTER_NAME not in transformer_adapters:
                FLUX_PIPE.load_lora_weights(
                    str(REFCONTROL_LOCAL_PATH),
                    adapter_name=SAGE_REFCONTROL_ADAPTER_NAME,
                    use_safetensors=True,
                )
            # RIJKSOIL is already fused into the transformer. This sets only the
            # remaining unfused structural adapter and does not replace it.
            FLUX_PIPE.set_adapters(
                SAGE_REFCONTROL_ADAPTER_NAME,
                adapter_weights=float(SAGE_REFCONTROL_SCALE),
            )
            for parameter in FLUX_PIPE.transformer.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            print({
                "flux2_refcontrol": SAGE_REFCONTROL_SOURCE,
                "revision": SAGE_REFCONTROL_REVISION,
                "scale": SAGE_REFCONTROL_SCALE,
                "active_adapters": FLUX_PIPE.get_active_adapters(),
            })

        RENDER_CONTRACT = {
            "method": "SAGE piecewise-affine dense warp + FLUX.2 Klein spatial lock + RefControl",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "style_lora": LORA_SOURCE,
            "style_lora_revision": LORA_REVISION,
            "style_lora_scale": IMAGE_LORA_SCALE,
            "refcontrol_enabled": SAGE_REFCONTROL_ENABLED,
            "refcontrol_source": SAGE_REFCONTROL_SOURCE if SAGE_REFCONTROL_ENABLED else None,
            "refcontrol_revision": SAGE_REFCONTROL_REVISION if SAGE_REFCONTROL_ENABLED else None,
            "refcontrol_scale": SAGE_REFCONTROL_SCALE if SAGE_REFCONTROL_ENABLED else 0.0,
            "inference_steps": SAGE_FLUX_INFERENCE_STEPS,
            "guidance_scale": SAGE_FLUX_GUIDANCE_SCALE,
            "img2img_strength": SAGE_FLUX_IMG2IMG_STRENGTH,
            "warp_max_control_lines": SAGE_WARP_MAX_CONTROL_LINES,
            "warp_border_samples": SAGE_WARP_BORDER_SAMPLES,
            "warp_eased_blend": SAGE_WARP_EASED_BLEND,
            "structure_lock_strength": SAGE_STRUCTURE_LOCK_STRENGTH,
            "structure_lock_dilation": SAGE_STRUCTURE_LOCK_DILATION_PIXELS,
            "structure_lock_feather": SAGE_STRUCTURE_LOCK_FEATHER_RADIUS,
            "grain_strength": SAGE_INIT_GRAIN_STRENGTH,
            "prompt_interpolation": "linear endpoint embedding interpolation",
            "reference_order": "SAGE black/white line map, detailed warped endpoint blend",
            "previous_frame_autoregression": False,
            "endpoint_blur": False,
        }
        RENDER_CONTRACT_HASH = hashlib.sha256(
            json.dumps(RENDER_CONTRACT, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if RUN_SAGE_RENDER:
            prompt_prefix = f"{SAGE_REFCONTROL_TRIGGER}. " if SAGE_REFCONTROL_ENABLED else ""
            print("Encoding each unique endpoint prompt once...")
            ANCHOR_CONDITIONING = {
                record["uid"]: encode_prompt_conditioning(
                    FLUX_PIPE,
                    prompt_prefix + record["prompt"],
                    device=FLUX_PIPE._execution_device,
                    max_sequence_length=FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
                ).cpu()
                for record in BASE_RECORDS
            }
            if callable(maybe_free):
                maybe_free()
            gc.collect()
            torch.cuda.empty_cache()

            COMPLETED_SAGE_GAPS = []
            for gap in SAGE_PREPARATION["gaps"]:
                gap_directory = Path(gap["gap_directory"])
                rendered_directory = gap_directory / "flux_rendered_v2"
                init_directory = gap_directory / "dense_warp_inits"
                control_directory = gap_directory / "flux_controls"
                complete_directory = gap_directory / "flux_complete_v2"
                for directory in (
                    rendered_directory, init_directory, control_directory, complete_directory
                ):
                    directory.mkdir(parents=True, exist_ok=True)
                metadata_path = gap_directory / "flux_render_metadata_v2.json"
                structure_data_path = Path(gap["structure_data_path"])
                with np.load(structure_data_path) as structure_data:
                    matched_source_lines = structure_data["matched_source_lines"].copy()
                    matched_target_lines = structure_data["matched_target_lines"].copy()
                    intermediate_lines = structure_data["intermediate_lines"].copy()

                left_record, right_record = gap["left"], gap["right"]
                fingerprint = hashlib.sha256(json.dumps({
                    "render_contract_hash": RENDER_CONTRACT_HASH,
                    "sage_preparation_contract_hash": SAGE_PREPARATION["contract_hash"],
                    "gap_uid": gap["gap_uid"],
                    "endpoint_hashes": gap["endpoint_hashes"],
                    "left_prompt": left_record["prompt"],
                    "right_prompt": right_record["prompt"],
                    "structure_data_sha256": gap["structure_data_sha256"],
                }, sort_keys=True).encode("utf-8")).hexdigest()

                saved = {}
                if metadata_path.is_file():
                    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                saved_paths = [Path(path) for path in saved.get("rendered_frame_paths", [])]
                reuse_gap = (
                    SAGE_REUSE_COMPLETED_GAPS
                    and saved.get("fingerprint") == fingerprint
                    and saved.get("complete")
                    and len(saved_paths) == SAGE_GENERATED_FRAMES_PER_GAP
                    and all(path.is_file() for path in saved_paths)
                )
                if reuse_gap:
                    COMPLETED_SAGE_GAPS.append(saved)
                    print("Reusing completed FLUX gap:", gap["gap_uid"])
                    continue

                left_image = fit_frame(left_record["path"])
                right_image = fit_frame(right_record["path"])
                rendered_paths = []
                init_paths = []
                control_paths = []
                lock_paths = []
                alphas = gap["condition_alphas"]
                if len(alphas) != len(intermediate_lines):
                    raise RuntimeError("SAGE geometry count does not match configured frames")

                for frame_index, (alpha, condition_path, middle_lines) in enumerate(
                    zip(alphas, gap["condition_paths"], intermediate_lines)
                ):
                    output_path = rendered_directory / f"flux_sage_{frame_index:04d}.png"
                    init_path = init_directory / f"warp_{frame_index:04d}.png"
                    canny_path = control_directory / f"canny_{frame_index:04d}.png"
                    lock_path = control_directory / f"lock_{frame_index:04d}.png"
                    frame_seed = BASE_SEED + 500_000 + int(gap["gap_index"])
                    with Image.open(condition_path) as opened_condition:
                        condition_image = fit_frame(opened_condition)
                    warp_result = warp_sage_endpoints(
                        left_image,
                        right_image,
                        matched_source_lines,
                        matched_target_lines,
                        middle_lines,
                        float(alpha),
                        maximum_control_lines=SAGE_WARP_MAX_CONTROL_LINES,
                        border_samples_per_edge=SAGE_WARP_BORDER_SAMPLES,
                        eased_blend=SAGE_WARP_EASED_BLEND,
                        grain_strength=SAGE_INIT_GRAIN_STRENGTH,
                        seed=frame_seed + frame_index,
                    )
                    init_image = warp_result.image
                    canny_image = make_canny_reference(
                        condition_image,
                        dilation_pixels=SAGE_REFCONTROL_CANNY_DILATION_PIXELS,
                    )
                    lock_mask = make_structure_lock_mask(
                        condition_image,
                        dilation_pixels=SAGE_STRUCTURE_LOCK_DILATION_PIXELS,
                        feather_radius=SAGE_STRUCTURE_LOCK_FEATHER_RADIUS,
                        strength=SAGE_STRUCTURE_LOCK_STRENGTH,
                    )
                    init_image.save(init_path, format="PNG", compress_level=4)
                    canny_image.save(canny_path, format="PNG", compress_level=4)
                    lock_mask.save(lock_path, format="PNG", compress_level=4)
                    init_paths.append(str(init_path))
                    control_paths.append(str(canny_path))
                    lock_paths.append(str(lock_path))

                    generator = torch.Generator(device="cuda").manual_seed(frame_seed)
                    spatial = prepare_flux2_klein_spatial_lock_inputs(
                        FLUX_PIPE,
                        init_image,
                        lock_mask,
                        width=SAGE_WIDTH,
                        height=SAGE_HEIGHT,
                        num_inference_steps=SAGE_FLUX_INFERENCE_STEPS,
                        strength=SAGE_FLUX_IMG2IMG_STRENGTH,
                        generator=generator,
                    )
                    prompt_package = interpolate_conditioning(
                        ANCHOR_CONDITIONING[left_record["uid"]],
                        ANCHOR_CONDITIONING[right_record["uid"]],
                        float(alpha),
                    ).to(FLUX_PIPE._execution_device, dtype=torch.bfloat16)
                    call_kwargs = dict(
                        prompt_embeds=prompt_package.prompt_embeds,
                        height=SAGE_HEIGHT,
                        width=SAGE_WIDTH,
                        num_inference_steps=SAGE_FLUX_INFERENCE_STEPS,
                        guidance_scale=SAGE_FLUX_GUIDANCE_SCALE,
                        generator=generator,
                        latents=spatial.latents,
                        sigmas=list(spatial.sigmas),
                        callback_on_step_end=spatial.callback_on_step_end,
                        callback_on_step_end_tensor_inputs=["latents"],
                        output_type="pil",
                        max_sequence_length=FLUX_PROMPT_MAX_SEQUENCE_LENGTH,
                    )
                    if SAGE_REFCONTROL_ENABLED:
                        call_kwargs["image"] = [canny_image, init_image]
                    result = FLUX_PIPE(**call_kwargs)
                    generated = result.images[0].convert("RGB")
                    generated.save(output_path, format="PNG", compress_level=4)
                    rendered_paths.append(str(output_path))

                    partial = {
                        "gap_uid": gap["gap_uid"],
                        "gap_index": gap["gap_index"],
                        "left": left_record,
                        "right": right_record,
                        "fingerprint": fingerprint,
                        "render_contract": RENDER_CONTRACT,
                        "rendered_frame_paths": rendered_paths,
                        "dense_warp_init_paths": init_paths,
                        "canny_control_paths": control_paths,
                        "structure_lock_paths": lock_paths,
                        "complete": False,
                    }
                    write_json_atomic(metadata_path, partial)
                    print(
                        f"Saved controlled FLUX frame {frame_index + 1}/"
                        f"{SAGE_GENERATED_FRAMES_PER_GAP} for {gap['gap_uid']} "
                        f"({spatial.denoising_steps} denoising steps)"
                    )
                    if frame_index in {0, len(alphas) // 2, len(alphas) - 1}:
                        progress = generated.copy()
                        progress.thumbnail((512, 512))
                        display(progress)
                        progress.close()
                    generated.close()
                    condition_image.close()
                    canny_image.close()
                    lock_mask.close()
                    warp_result.warped_left.close()
                    warp_result.warped_right.close()
                    init_image.close()
                    del result, spatial, prompt_package, call_kwargs

                complete_paths = []
                complete_images = [left_image]
                complete_images.extend(fit_frame(path) for path in rendered_paths)
                complete_images.append(right_image)
                for index, frame in enumerate(complete_images):
                    destination = complete_directory / f"frame_{index:04d}.png"
                    frame.save(destination, format="PNG", compress_level=4)
                    complete_paths.append(str(destination))
                clip_path = gap_directory / "flux_sage_transition_v2.mp4"
                ffmpeg_frames(complete_directory, clip_path)
                completed = {
                    **partial,
                    "condition_alphas": alphas,
                    "condition_paths": gap["condition_paths"],
                    "complete_frame_paths": complete_paths,
                    "clip_path": str(clip_path),
                    "complete": True,
                }
                write_json_atomic(metadata_path, completed)
                COMPLETED_SAGE_GAPS.append(completed)
                for frame in complete_images:
                    frame.close()
                print("Completed controlled FLUX gap:", clip_path)

            sparse_sequence_directory = SAGE_OUTPUT_ROOT / "flux_cyclic_keyframes"
            if sparse_sequence_directory.exists():
                shutil.rmtree(sparse_sequence_directory)
            sparse_sequence_directory.mkdir(parents=True, exist_ok=False)
            sparse_sequence_paths = []
            sequence_index = 0
            for gap in COMPLETED_SAGE_GAPS:
                for path in [Path(value) for value in gap["complete_frame_paths"]][:-1]:
                    destination = sparse_sequence_directory / f"frame_{sequence_index:04d}.png"
                    shutil.copy2(path, destination)
                    sparse_sequence_paths.append(str(destination))
                    sequence_index += 1
            SAGE_SPARSE_VIDEO_PATH = RUN_DIRECTORY / "video" / "sage_flux2_controlled_keyframes.mp4"
            ffmpeg_frames(sparse_sequence_directory, SAGE_SPARSE_VIDEO_PATH)
        else:
            print("FLUX SAGE render intentionally stopped after guide inspection.")
        """
    ),
    markdown(
        """
        ## 12. Finish the controlled keyframes with Practical-RIFE

        FLUX now renders only five structurally meaningful interiors per gap.
        RIFE fills each neighboring pair after the expensive FLUX pipeline is
        released. The opening image is appended once so the final wraparound
        pair is interpolated explicitly; that duplicate is removed afterward.
        """
    ),
    code(
        """
        import zipfile

        if RUN_SAGE_RENDER and RUN_RIFE_POSTPROCESS:
            release_flux_pipeline()
            gc.collect()
            torch.cuda.empty_cache()

            rife_root = Path(RIFE_ROOT)
            if not rife_root.is_dir():
                subprocess.check_call([
                    "git", "clone", "--filter=blob:none", RIFE_REPOSITORY_URL, str(rife_root)
                ])
            installed_revision = subprocess.check_output(
                ["git", "-C", str(rife_root), "rev-parse", "HEAD"], text=True
            ).strip()
            if installed_revision != RIFE_REPOSITORY_REVISION:
                subprocess.check_call([
                    "git", "-C", str(rife_root), "fetch", "origin", RIFE_REPOSITORY_REVISION
                ])
                subprocess.check_call([
                    "git", "-C", str(rife_root), "checkout", "--detach", RIFE_REPOSITORY_REVISION
                ])

            rife_archive = Path(hf_hub_download(
                repo_id=RIFE_MODEL_REPOSITORY,
                filename=RIFE_MODEL_FILENAME,
                revision=RIFE_MODEL_REVISION,
                cache_dir=HF_CACHE_DIR,
            ))
            rife_model_root = (
                Path(HF_CACHE_DIR) / "flowmorph_rife_models" / RIFE_MODEL_FILENAME.removesuffix(".zip")
            )
            flownet_candidates = list(rife_model_root.rglob("flownet.pkl")) if rife_model_root.exists() else []
            if not flownet_candidates:
                rife_model_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(rife_archive) as archive:
                    archive.extractall(rife_model_root)
                flownet_candidates = list(rife_model_root.rglob("flownet.pkl"))
            if len(flownet_candidates) != 1:
                raise RuntimeError(f"Expected one RIFE flownet.pkl, found {len(flownet_candidates)}")
            rife_model_directory = flownet_candidates[0].parent

            rife_work = SAGE_OUTPUT_ROOT / "rife_work"
            rife_input = rife_work / "cyclic_input"
            rife_dense = rife_work / "dense_with_duplicate"
            for path in (rife_input, rife_dense):
                if path.exists():
                    shutil.rmtree(path)
            rife_input.mkdir(parents=True, exist_ok=False)
            for index, source_path in enumerate(sparse_sequence_paths):
                shutil.copy2(source_path, rife_input / f"{index:07d}.png")
            shutil.copy2(
                sparse_sequence_paths[0],
                rife_input / f"{len(sparse_sequence_paths):07d}.png",
            )
            rife_runner = Path(PROJECT_ROOT) / "scripts" / "rife_pair_sequence_runner.py"
            command = [
                sys.executable, "-u", str(rife_runner),
                "--repo", str(rife_root),
                "--model", str(rife_model_directory),
                "--input", str(rife_input),
                "--output", str(rife_dense),
                "--multi", str(RIFE_MULTIPLIER),
                "--scale", str(RIFE_SCALE),
                "--batch-size", str(RIFE_BATCH_SIZE),
            ]
            if RIFE_USE_FP16:
                command.append("--fp16")
            return_code = subprocess.call(command)
            if return_code and "--fp16" in command:
                print("RIFE fp16 failed; retrying in fp32.")
                shutil.rmtree(rife_dense, ignore_errors=True)
                command.remove("--fp16")
                subprocess.check_call(command)
            elif return_code:
                raise RuntimeError(f"RIFE failed with exit code {return_code}")

            dense_paths = sorted(rife_dense.glob("*.png"), key=lambda path: int(path.stem))
            if len(dense_paths) < 2:
                raise RuntimeError("RIFE produced fewer than two frames")
            first_hash = sha256_file(dense_paths[0])
            last_hash = sha256_file(dense_paths[-1])
            if first_hash != last_hash:
                raise RuntimeError("RIFE terminal frame is not the expected cyclic duplicate")
            final_sequence_directory = SAGE_OUTPUT_ROOT / "rife_cyclic_frames"
            if final_sequence_directory.exists():
                shutil.rmtree(final_sequence_directory)
            final_sequence_directory.mkdir(parents=True, exist_ok=False)
            sequence_paths = []
            for index, source_path in enumerate(dense_paths[:-1]):
                destination = final_sequence_directory / f"frame_{index:04d}.png"
                shutil.copy2(source_path, destination)
                sequence_paths.append(str(destination))
            SAGE_FINAL_VIDEO_PATH = RUN_DIRECTORY / "video" / "sage_flux2_controlled_rife_cyclic.mp4"
            ffmpeg_frames(final_sequence_directory, SAGE_FINAL_VIDEO_PATH)
        elif RUN_SAGE_RENDER:
            sequence_paths = sparse_sequence_paths
            SAGE_FINAL_VIDEO_PATH = SAGE_SPARSE_VIDEO_PATH

        if RUN_SAGE_RENDER:
            SAGE_SEQUENCE = {
                "method": "SAGE dense warps + FLUX.2 Klein RefControl/spatial lock + RIFE",
                "paper": "https://arxiv.org/abs/2510.24667v2",
                "cyclic": True,
                "renderer": MODEL_ID,
                "style_lora": LORA_SOURCE,
                "control_lora": SAGE_REFCONTROL_SOURCE if SAGE_REFCONTROL_ENABLED else None,
                "render_contract": RENDER_CONTRACT,
                "gaps": COMPLETED_SAGE_GAPS,
                "sparse_frame_paths": sparse_sequence_paths,
                "sequence_frame_paths": sequence_paths,
                "frame_count": len(sequence_paths),
                "fps": SAGE_OUTPUT_FPS,
                "rife_multiplier": RIFE_MULTIPLIER if RUN_RIFE_POSTPROCESS else 1,
                "final_video_path": str(SAGE_FINAL_VIDEO_PATH),
            }
            SAGE_SEQUENCE_MANIFEST_PATH = SAGE_OUTPUT_ROOT / "sage_sequence_manifest_v2.json"
            write_json_atomic(SAGE_SEQUENCE_MANIFEST_PATH, SAGE_SEQUENCE)
            print({
                "sparse_flux_frames": len(sparse_sequence_paths),
                "final_frames": len(sequence_paths),
                "final_video": str(SAGE_FINAL_VIDEO_PATH),
            })
        """
    ),
    markdown("""## 13. Preview the controlled FLUX/SAGE result"""),
    code(
        """
        from IPython.display import Video

        if RUN_SAGE_RENDER:
            midpoint_paths = []
            init_midpoint_paths = []
            for gap in SAGE_SEQUENCE["gaps"]:
                rendered = [Path(path) for path in gap["rendered_frame_paths"]]
                inits = [Path(path) for path in gap["dense_warp_init_paths"]]
                midpoint_paths.append(rendered[len(rendered) // 2])
                init_midpoint_paths.append(inits[len(inits) // 2])
                if SAGE_DISPLAY_EACH_GAP_VIDEO:
                    display(Video(str(gap["clip_path"]), embed=False, width=SAGE_VIDEO_DISPLAY_WIDTH))
            display_path_contact_sheet(
                init_midpoint_paths,
                [gap["gap_uid"] for gap in SAGE_SEQUENCE["gaps"]],
                "### Detailed warped initialization at each gap midpoint",
                "sage_dense_warp_midpoints.png",
            )
            display_path_contact_sheet(
                midpoint_paths,
                [gap["gap_uid"] for gap in SAGE_SEQUENCE["gaps"]],
                "### Controlled FLUX.2 Klein result at each gap midpoint",
                "sage_flux_controlled_midpoints.png",
            )
            display(Markdown("## Final cyclic SAGE/FLUX/RIFE video"))
            display(Video(
                str(SAGE_FINAL_VIDEO_PATH),
                embed=False,
                width=SAGE_VIDEO_DISPLAY_WIDTH,
                html_attributes="controls loop muted",
            ))
            if DOWNLOAD_FINAL_VIDEO:
                from google.colab import files
                files.download(str(SAGE_FINAL_VIDEO_PATH))
        """
    ),
    markdown(
        """
        ## What “control” means here

        Diffusers currently exposes its trained `FluxControlNetModel` pipeline
        for FLUX.1, not FLUX.2 Klein. Loading one of those weights into Klein
        would be architecturally invalid. This notebook instead uses a learned
        FLUX.2 Klein Canny/reference control LoRA, trained for this exact Base 9B
        model, plus an inspectable SAGE-derived dense warp and latent trajectory
        lock. It is therefore entirely FLUX.2 Klein, while preserving the part
        of SAGE that actually supplies geometric motion.
        """
    ),
])

for index, cell in enumerate(cells):
    cell["id"] = f"sage-flux-{index:02d}"
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook["cells"] = cells
notebook["nbformat"] = 4
notebook["nbformat_minor"] = 5
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
