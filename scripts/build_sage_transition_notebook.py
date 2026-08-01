"""Build the prompt-only FLUX anchor -> one-round SAGE video notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Prompt_Only.ipynb"
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


source_notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
prompt_cell = copy.deepcopy(source_notebook["cells"][4])
prompt_cell["execution_count"] = None
prompt_cell["outputs"] = []

cells = [
    markdown(
        """
        # Prompt-only FLUX.2 anchors → one-round SAGE transition video

        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MNoichl/FluxFlowMorph/blob/main/notebooks/StillLife_SAGE_Transition_Video.ipynb)

        This experimental notebook keeps the established editable anchor-prompt,
        RIJKSOIL LoRA, weak blurred/grained continuity, numbered Drive run,
        resumability, and cyclic-video setup. It replaces FlowMorph with
        **SAGE: Structure-Aware Generative Video Transitions** for one round.
        It is pinned to the [v2 paper](https://arxiv.org/html/2510.24667v2)
        and the authors' [official code](https://github.com/kan32501/SAGE).

        For each circular anchor gap, it:

        1. finds foreground line structures with GlueStick;
        2. matches them in mask-normalized coordinates with Hungarian assignment;
        3. propagates them along a smooth cubic global trajectory while locally
           interpolating each matched segment;
        4. rasterizes one structural condition per time step; and
        5. gives both endpoint paintings and those conditions to FCVG to synthesize
           the SAGE transition frames.

        Important adaptation: the paper starts from **moving clips** and obtains
        endpoint motion with SEA-RAFT. These inputs are still paintings, so real
        clip flow does not exist. The notebook therefore uses a deterministic,
        exposed synthetic-flow bend for the global spline. Everything else above
        remains the SAGE structural/generative pipeline. Exact anchors are retained
        between gaps and the last gap returns to the first anchor.
        """
    ),
    markdown(
        """
        ## 1. Editable anchor-generation and SAGE settings

        The SAGE block is deliberately separate from the FLUX block. Start with
        576×576 and 13 generated frames on a 24 GiB Colab GPU; increase resolution
        only after one complete gap succeeds.
        """
    ),
    code(
        """
        PROJECT_ROOT = "/content/FlowMorphKlein9B"
        REPOSITORY_URL = "https://github.com/MNoichl/FluxFlowMorph.git"
        UPDATE_REPOSITORY = True
        PROJECT_NAME = "science_path_sage_transition"
        LOCAL_ASSET_ROOT = "/content/sage_transition_art"
        HF_CACHE_DIR = "/content/hf_cache"

        # Drive persistence and gated Stable Video Diffusion access.
        MOUNT_DRIVE = True
        DRIVE_PROJECT_BASE = "/content/drive/MyDrive/FluxFlowMorphArt"
        HF_TOKEN_FILENAME = "hftoken.txt"  # Optional fallback inside DRIVE_PROJECT_BASE.
        RESUME_RUN_DIRECTORY = None

        # Editable anchor selection.
        BASE_PROMPT_COUNT = None  # None uses every BASE_STAGES entry; any 3..len(BASE_STAGES) works.
        REGENERATE_BASE_FRAMES = True

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

        # Weak anchor continuity: blurred/grained previous image, no beige canvas.
        BASE_CONTINUITY_ENABLED = True
        BASE_REFERENCE_BLUR = 16.0
        BASE_REFERENCE_GRAIN_STRENGTH = 0.035
        BASE_REFERENCE_DENOISE_STRENGTH = 0.75
        SAVE_SOFT_REFERENCES = True
        FLUX_PROMPT_MAX_SEQUENCE_LENGTH = 512

        # Optional cheap anchor trial.
        RUN_TRIAL_KEYFRAME = True
        TRIAL_KEYFRAME_INDEX = None
        TRIAL_SEED = None
        TRIAL_DISPLAY_MAX_WIDTH = 768
        CONTACT_SHEET_COLUMNS = 6
        CONTACT_SHEET_DISPLAY_MAX_WIDTH = 1100

        # Automatic foreground masks. Replace individual generated PNGs and rerun
        # from section 9 if GrabCut misses the composition.
        SAGE_MASK_MODE = "grabcut"  # "grabcut", "full_frame", or "directory"
        SAGE_MASK_SOURCE_DIRECTORY = None  # For directory mode: one <anchor_uid>.png per anchor.
        SAGE_MASK_REGENERATE = True
        SAGE_GRABCUT_MARGIN_FRACTION = 0.035
        SAGE_MASK_DILATION_PIXELS = 7
        SAGE_MASK_MIN_COVERAGE = 0.04
        SAGE_MASK_MAX_COVERAGE = 0.96

        # Pinned paper implementation and pretrained components.
        SAGE_REPOSITORY_URL = "https://github.com/kan32501/SAGE.git"
        SAGE_REPOSITORY_COMMIT = "5a30e6bfb035e2c243d90d4804ebda2addecacf4"
        SAGE_REPOSITORY_DIRECTORY = "/content/SAGE"
        SAGE_VENV_DIRECTORY = "/content/sage_runtime_venv"
        SAGE_GLUESTICK_URL = "https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar"
        SAGE_FCVG_REPOSITORY = "melmass/FCVG"
        SAGE_FCVG_REVISION = "e03fde54104b2e1a9642b671bcf8d8f8d7bf34d4"
        SAGE_BASE_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        SAGE_BASE_MODEL_REVISION = "043843887ccd51926e3efed36270444a838e7861"
        SAGE_PERSIST_MODEL_CACHE_TO_DRIVE = True

        # One SAGE round per cyclic anchor gap.
        SAGE_WIDTH = 576
        SAGE_HEIGHT = 576
        SAGE_GENERATED_FRAMES_PER_GAP = 13
        SAGE_INFERENCE_STEPS = 25
        SAGE_CONTROL_WEIGHT = 1.0
        SAGE_MIN_GUIDANCE = 3.0
        SAGE_MAX_GUIDANCE = 3.0
        SAGE_MOTION_BUCKET_ID = 127.0
        SAGE_NOISE_AUG_STRENGTH = 0.02
        SAGE_FRAMES_PER_BATCH = 13
        SAGE_OVERLAP = 5
        SAGE_DECODE_CHUNK_SIZE = 2
        SAGE_MAX_POINTS = 1000
        SAGE_MAX_LINES = 200
        SAGE_MAX_MATCHED_LINES = 160
        SAGE_MINIMUM_MATCHED_LINES = 8
        SAGE_CONDITION_LINE_WIDTH = 2

        # Still-image substitute for unavailable clip flow. 0 gives a direct
        # center path; small values curve the global foreground trajectory.
        SAGE_SYNTHETIC_FLOW_SCALE = 0.16
        SAGE_TRAJECTORY_BEND = 0.04

        # Output, reuse, and display.
        SAGE_REUSE_COMPLETED_GAPS = True
        SAGE_OUTPUT_FPS = 12.0
        SAGE_VIDEO_CRF = 16
        SAGE_DISPLAY_EACH_GAP_VIDEO = False
        SAGE_VIDEO_DISPLAY_WIDTH = 768
        RUN_SAGE_RENDER = True  # Set False to stop after mask/line-guide inspection.
        DOWNLOAD_FINAL_VIDEO = False
        DELETE_LOCAL_FLUX_CACHE_BEFORE_SAGE = True
        """
    ),
    markdown(
        """
        ## 2. Editable anchor sciences and prompts

        These are copied from the current prompt-only notebook. Edit them here in
        exactly the same way; each prompt must contain `RIJKSOIL` exactly once.
        """
    ),
    prompt_cell,
    markdown(
        """
        ## 3. GPU, repository, and FLUX dependencies

        SAGE itself later runs in an isolated venv because the authors pin
        diffusers 0.27 while FLUX.2 needs a newer implementation.
        """
    ),
    code(
        """
        import platform
        import subprocess
        import sys
        from pathlib import Path

        print({"python": sys.version, "platform": platform.platform()})
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is missing; use a Colab GPU runtime.") from error
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU runtime is required.")
        print({"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda})

        project_path = Path(PROJECT_ROOT)
        if not (project_path / "pyproject.toml").is_file():
            subprocess.check_call(["git", "clone", "--depth", "1", REPOSITORY_URL, PROJECT_ROOT])
        elif UPDATE_REPOSITORY:
            subprocess.check_call(["git", "-C", PROJECT_ROOT, "pull", "--ff-only"])

        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import cv2, numpy, scipy, transformers, pydantic; "
                    "from diffusers import Flux2KleinPipeline"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            print(probe.stderr[-2000:])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r",
                str(project_path / "requirements-colab.txt"),
            ])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", PROJECT_ROOT])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless>=4.9,<5"])
            raise RuntimeError(
                "Dependencies installed successfully. Restart the kernel once, then rerun from section 1."
            )

        import importlib
        package_source = str(project_path / "src")
        if package_source not in sys.path:
            sys.path.insert(0, package_source)
        importlib.invalidate_caches()
        import flowmorph_klein
        from diffusers import Flux2KleinPipeline

        project_commit = subprocess.check_output(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"], text=True
        ).strip()
        print({
            "repository_commit": project_commit,
            "flowmorph_source": flowmorph_klein.__file__,
        })
        """
    ),
    markdown(
        """
        ## 4. Mount Drive, reserve a persistent run, and resolve `HF_TOKEN`

        Stable Video Diffusion 1.1 is gated. Accept its Hugging Face license once,
        then provide the token through Colab Secrets (`HF_TOKEN`), the process
        environment, your existing Hugging Face login, or `hftoken.txt` in the
        configured Drive base. The credential value is never printed or recorded.
        """
    ),
    code(
        """
        import getpass
        import json
        import os
        import re
        from datetime import datetime, timezone
        from huggingface_hub import get_token

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", PROJECT_NAME):
            raise ValueError("PROJECT_NAME may contain only letters, numbers, underscores, and hyphens")

        DRIVE_ENABLED = False
        if MOUNT_DRIVE:
            from google.colab import drive
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

        for child in ("base_frames", "trials", "previews", "metadata", "sage", "video"):
            (RUN_DIRECTORY / child).mkdir(parents=True, exist_ok=True)
        Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

        hf_token = os.environ.get("HF_TOKEN") or get_token()
        authentication_source = "environment_or_huggingface_login" if hf_token else None
        if not hf_token:
            try:
                from google.colab import userdata
                hf_token = userdata.get("HF_TOKEN")
                if hf_token:
                    authentication_source = "colab_secret"
            except Exception:
                pass
        if not hf_token and drive_base is not None:
            token_path = drive_base / HF_TOKEN_FILENAME
            if token_path.is_file():
                hf_token = token_path.read_text(encoding="utf-8").strip()
                authentication_source = "drive_token_file"
        if not hf_token:
            hf_token = getpass.getpass("Hugging Face access token (hidden): ").strip()
            authentication_source = "interactive_hidden_prompt"
        if len(hf_token) < 20 or any(character.isspace() for character in hf_token):
            raise ValueError("HF_TOKEN is empty or malformed")
        os.environ["HF_TOKEN"] = hf_token

        if SAGE_PERSIST_MODEL_CACHE_TO_DRIVE:
            if not DRIVE_ENABLED:
                raise RuntimeError("Persistent SAGE model cache requires MOUNT_DRIVE=True")
            SAGE_EFFECTIVE_CACHE_DIRECTORY = drive_base / "_model_cache" / "sage"
        else:
            SAGE_EFFECTIVE_CACHE_DIRECTORY = Path(HF_CACHE_DIR) / "sage"
        SAGE_EFFECTIVE_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

        run_identity = {
            "project": PROJECT_NAME,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "persistent": DRIVE_ENABLED,
            "run_directory": str(RUN_DIRECTORY),
            "authentication_source": authentication_source,
            "credential_value_recorded": False,
            "sage_model_cache": str(SAGE_EFFECTIVE_CACHE_DIRECTORY),
        }
        (RUN_DIRECTORY / "metadata" / "run_identity.json").write_text(
            json.dumps(run_identity, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8"
        )
        print("HF access resolved without displaying the credential.")
        print("Run directory:", RUN_DIRECTORY)
        print("SAGE model cache:", SAGE_EFFECTIVE_CACHE_DIRECTORY)
        """
    ),
    markdown(
        """
        ## 5. Validate the creative and SAGE contracts
        """
    ),
    code(
        """
        if BASE_PROMPT_COUNT is None:
            BASE_PROMPT_COUNT = len(BASE_STAGES)
        elif not 3 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):
            raise ValueError(f"BASE_PROMPT_COUNT must be between 3 and {len(BASE_STAGES)}")
        ACTIVE_BASE_STAGES = BASE_STAGES[:BASE_PROMPT_COUNT]

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

        ids = [item["id"] for item in ACTIVE_BASE_STAGES]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_]+", item) for item in ids):
            raise ValueError("Anchor IDs must be unique lowercase snake_case values")
        for item in ACTIVE_BASE_STAGES:
            if not item["science"].strip() or not item["prompt"].strip():
                raise ValueError(f"Blank science or prompt in {item['id']}")
            if item["prompt"].casefold().count(LORA_TRIGGER.casefold()) != 1:
                raise ValueError(f"{item['id']} must contain the LoRA trigger exactly once")

        if SAGE_MASK_MODE not in {"grabcut", "full_frame", "directory"}:
            raise ValueError("SAGE_MASK_MODE must be grabcut, full_frame, or directory")
        if SAGE_MASK_MODE == "directory" and not SAGE_MASK_SOURCE_DIRECTORY:
            raise ValueError("directory mask mode requires SAGE_MASK_SOURCE_DIRECTORY")
        if not 0 < SAGE_MASK_MIN_COVERAGE < SAGE_MASK_MAX_COVERAGE <= 1:
            raise ValueError("Invalid SAGE mask coverage interval")
        if SAGE_WIDTH % 64 or SAGE_HEIGHT % 64:
            raise ValueError("SAGE_WIDTH and SAGE_HEIGHT must be divisible by 64")
        if SAGE_GENERATED_FRAMES_PER_GAP < 5:
            raise ValueError("SAGE needs at least five generated frames per gap")
        if not 1 <= SAGE_FRAMES_PER_BATCH <= SAGE_GENERATED_FRAMES_PER_GAP:
            raise ValueError("Invalid SAGE_FRAMES_PER_BATCH")
        if not 0 <= SAGE_OVERLAP < SAGE_FRAMES_PER_BATCH:
            raise ValueError("SAGE_OVERLAP must be smaller than the frame batch")
        if not 0 < SAGE_CONTROL_WEIGHT <= 3:
            raise ValueError("SAGE_CONTROL_WEIGHT must lie in (0, 3]")

        print({
            "anchors": BASE_PROMPT_COUNT,
            "cyclic_gaps": BASE_PROMPT_COUNT,
            "sage_generated_frames_per_gap": SAGE_GENERATED_FRAMES_PER_GAP,
            "exact_anchor_plus_sage_frames_per_gap": SAGE_GENERATED_FRAMES_PER_GAP + 1,
            "final_cyclic_frames": BASE_PROMPT_COUNT * (SAGE_GENERATED_FRAMES_PER_GAP + 1),
            "size": [SAGE_WIDTH, SAGE_HEIGHT],
            "still_motion_fallback": {
                "synthetic_flow_scale": SAGE_SYNTHETIC_FLOW_SCALE,
                "trajectory_bend": SAGE_TRAJECTORY_BEND,
            },
        })
        print("Anchor order:", " → ".join(ids), "→", ids[0])
        """
    ),
    markdown(
        """
        ## 6. Load RIJKSOIL and optionally render one anchor trial
        """
    ),
    copy.deepcopy(source_notebook["cells"][12]),
    markdown(
        """
        ## 7. Generate the softly related cyclic anchor paintings

        As in the prompt-only workflow, each later anchor can use a blurred,
        grained previous painting as weak latent img2img initialization. There is
        no mask canvas and no fixed beige background construction.
        """
    ),
    copy.deepcopy(source_notebook["cells"][14]),
    copy.deepcopy(source_notebook["cells"][15]),
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
            foreground = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
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
                input_path = Path(SAGE_MASK_SOURCE_DIRECTORY).expanduser() / f"{record['uid']}.png"
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
        SAGE_ANCHOR_MANIFEST_PATH = RUN_DIRECTORY / "metadata" / "sage_anchor_manifest.json"
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

        mask_contact_sheet_path = RUN_DIRECTORY / "previews" / "sage_foreground_masks.png"
        mask_previews = []
        for item in SAGE_MASK_RECORDS:
            with Image.open(item["mask_path"]) as opened:
                mask_previews.append(opened.convert("RGB"))
        make_contact_sheet(
            mask_previews,
            mask_contact_sheet_path,
            columns=min(CONTACT_SHEET_COLUMNS, len(mask_previews)),
            labels=[f"{item['uid']} ({item['coverage']:.0%})" for item in SAGE_MASK_RECORDS],
        )
        for image in mask_previews:
            image.close()
        mask_preview = Image.open(mask_contact_sheet_path).convert("RGB")
        mask_preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
        display(Markdown("### SAGE foreground masks — white structures will guide matching"))
        display(mask_preview)
        mask_preview.close()
        print("Editable full-resolution masks:", SAGE_MASK_DIRECTORY)
        """
    ),
    markdown(
        """
        ## 9. Release FLUX, install the isolated SAGE runtime, and fetch pinned weights

        The base SVD model is gated. If this cell reports 401/403, accept the
        model's license on Hugging Face for the account owning `HF_TOKEN`.
        The persistent Drive cache avoids downloading the multi-gigabyte FCVG/SVD
        weights on every Colab session; the first load from Drive is slower.
        """
    ),
    code(
        """
        import gc
        import hashlib
        import shutil
        import urllib.request
        from huggingface_hub import hf_hub_download

        release_flux_pipeline()
        for stale_name in (
            "FLUX_PROMPT_TOKENIZER",
            "LORA_REPORT",
            "LOCAL_LORA_PATH",
            "downloaded_lora",
        ):
            globals().pop(stale_name, None)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        if DELETE_LOCAL_FLUX_CACHE_BEFORE_SAGE:
            flux_cache = Path(HF_CACHE_DIR) / ("models--" + MODEL_ID.replace("/", "--"))
            if flux_cache.is_dir() and str(flux_cache).startswith(str(Path(HF_CACHE_DIR))):
                shutil.rmtree(flux_cache)
                print("Removed released local FLUX weights:", flux_cache)

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

        runtime_contract = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "packages": [
                "numpy==1.26.4",
                "scipy==1.11.4",
                "diffusers==0.27.0",
                "transformers==4.37.2",
                "huggingface_hub==0.25.2",
                "tokenizers==0.15.2",
                "safetensors==0.4.5",
                "omegaconf==2.3.0",
                "pytlsd==0.0.2",
                "einops==0.8.0",
                "opencv-python-headless==4.9.0.80",
                "accelerate>=0.27,<2",
            ],
        }
        runtime_hash = hashlib.sha256(
            json.dumps(runtime_contract, sort_keys=True).encode("utf-8")
        ).hexdigest()
        venv_directory = Path(SAGE_VENV_DIRECTORY)
        venv_python = venv_directory / "bin" / "python"
        marker_path = venv_directory / "flowmorph_sage_runtime.json"
        marker_matches = False
        if marker_path.is_file() and venv_python.is_file():
            marker_matches = json.loads(marker_path.read_text(encoding="utf-8")).get("hash") == runtime_hash
        if not marker_matches:
            if venv_directory.exists():
                shutil.rmtree(venv_directory)
            subprocess.check_call([
                sys.executable, "-m", "venv", "--system-site-packages", str(venv_directory),
            ])
            subprocess.check_call([
                str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "Cython<3",
            ])
            subprocess.check_call([
                str(venv_python), "-m", "pip", "install", *runtime_contract["packages"],
            ])
            marker_path.write_text(
                json.dumps({"hash": runtime_hash, **runtime_contract}, indent=2) + "\\n",
                encoding="utf-8",
            )
        subprocess.check_call([
            str(venv_python),
            "-c",
            (
                "import cv2, diffusers, huggingface_hub, numpy, omegaconf, pytlsd, scipy, torch, transformers; "
                "assert diffusers.__version__ == '0.27.0'; "
                "print({'sage_diffusers': diffusers.__version__, 'torch': torch.__version__})"
            ),
        ])

        checkpoint_root = SAGE_EFFECTIVE_CACHE_DIRECTORY / "paper_checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        SAGE_GLUESTICK_CHECKPOINT = checkpoint_root / "checkpoint_GlueStick_MD.tar"
        if not SAGE_GLUESTICK_CHECKPOINT.is_file():
            temporary_path = SAGE_GLUESTICK_CHECKPOINT.with_suffix(".download")
            urllib.request.urlretrieve(SAGE_GLUESTICK_URL, temporary_path)
            temporary_path.replace(SAGE_GLUESTICK_CHECKPOINT)

        SAGE_CONTROLNEXT_CHECKPOINT = Path(hf_hub_download(
            repo_id=SAGE_FCVG_REPOSITORY,
            filename="controlnext.safetensors",
            revision=SAGE_FCVG_REVISION,
            cache_dir=str(SAGE_EFFECTIVE_CACHE_DIRECTORY),
        ))
        SAGE_UNET_CHECKPOINT = Path(hf_hub_download(
            repo_id=SAGE_FCVG_REPOSITORY,
            filename="unet.safetensors",
            revision=SAGE_FCVG_REVISION,
            cache_dir=str(SAGE_EFFECTIVE_CACHE_DIRECTORY),
        ))
        # Tiny gated-file probe gives a clear failure before FCVG is loaded.
        hf_hub_download(
            repo_id=SAGE_BASE_MODEL_ID,
            filename="model_index.json",
            revision=SAGE_BASE_MODEL_REVISION,
            cache_dir=str(SAGE_EFFECTIVE_CACHE_DIRECTORY),
            token=hf_token,
        )

        disk = shutil.disk_usage("/content")
        print({
            "sage_commit": actual_sage_commit,
            "sage_python": str(venv_python),
            "checkpoint_cache": str(SAGE_EFFECTIVE_CACHE_DIRECTORY),
            "local_disk_free_gib": round(disk.free / 1024**3, 2),
            "cuda_reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        })
        """
    ),
    markdown(
        """
        ## 10. Prepare SAGE line matches and structural guides (cheap, inspectable phase)

        This phase loads GlueStick once, prepares every circular gap, saves the
        matched-line overlays and all frame-wise conditions, then exits before
        FCVG is loaded. Inspect the contact sheets. If a gap is nonsensical,
        correct its mask or tune the two still-trajectory controls before paying
        for the diffusion render.
        """
    ),
    code(
        """
        SAGE_OUTPUT_ROOT = RUN_DIRECTORY / "sage" / "one_round"
        SAGE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        SAGE_RUNNER = Path(PROJECT_ROOT) / "scripts" / "sage_still_sequence_runner.py"
        if not SAGE_RUNNER.is_file():
            raise FileNotFoundError(SAGE_RUNNER)

        SAGE_COMMAND_BASE = [
            str(venv_python),
            str(SAGE_RUNNER),
            "--sage-repo", str(sage_repository),
            "--manifest", str(SAGE_ANCHOR_MANIFEST_PATH),
            "--output-root", str(SAGE_OUTPUT_ROOT),
            "--gluestick-checkpoint", str(SAGE_GLUESTICK_CHECKPOINT),
            "--controlnext-checkpoint", str(SAGE_CONTROLNEXT_CHECKPOINT),
            "--unet-checkpoint", str(SAGE_UNET_CHECKPOINT),
            "--base-model-id", SAGE_BASE_MODEL_ID,
            "--base-model-revision", SAGE_BASE_MODEL_REVISION,
            "--hf-cache", str(SAGE_EFFECTIVE_CACHE_DIRECTORY),
            "--width", str(SAGE_WIDTH),
            "--height", str(SAGE_HEIGHT),
            "--generated-frames", str(SAGE_GENERATED_FRAMES_PER_GAP),
            "--inference-steps", str(SAGE_INFERENCE_STEPS),
            "--control-weight", str(SAGE_CONTROL_WEIGHT),
            "--min-guidance", str(SAGE_MIN_GUIDANCE),
            "--max-guidance", str(SAGE_MAX_GUIDANCE),
            "--motion-bucket-id", str(SAGE_MOTION_BUCKET_ID),
            "--noise-aug-strength", str(SAGE_NOISE_AUG_STRENGTH),
            "--frames-per-batch", str(SAGE_FRAMES_PER_BATCH),
            "--overlap", str(SAGE_OVERLAP),
            "--decode-chunk-size", str(SAGE_DECODE_CHUNK_SIZE),
            "--max-points", str(SAGE_MAX_POINTS),
            "--max-lines", str(SAGE_MAX_LINES),
            "--max-matched-lines", str(SAGE_MAX_MATCHED_LINES),
            "--minimum-matched-lines", str(SAGE_MINIMUM_MATCHED_LINES),
            "--line-width", str(SAGE_CONDITION_LINE_WIDTH),
            "--trajectory-bend", str(SAGE_TRAJECTORY_BEND),
            "--synthetic-flow-scale", str(SAGE_SYNTHETIC_FLOW_SCALE),
            "--seed", str(BASE_SEED + 500_000),
            "--fps", str(SAGE_OUTPUT_FPS),
            "--crf", str(SAGE_VIDEO_CRF),
        ]
        if SAGE_REUSE_COMPLETED_GAPS:
            SAGE_COMMAND_BASE.append("--reuse")

        sage_environment = os.environ.copy()
        sage_environment["HF_TOKEN"] = hf_token
        sage_environment["HF_HOME"] = str(SAGE_EFFECTIVE_CACHE_DIRECTORY)
        subprocess.check_call(
            [*SAGE_COMMAND_BASE, "--phase", "prepare"],
            env=sage_environment,
        )

        SAGE_PREPARATION_MANIFEST_PATH = SAGE_OUTPUT_ROOT / "sage_preparation_manifest.json"
        SAGE_PREPARATION = json.loads(
            SAGE_PREPARATION_MANIFEST_PATH.read_text(encoding="utf-8")
        )
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

        display_path_contact_sheet(
            line_preview_paths,
            [path.parent.name + "/" + path.stem for path in line_preview_paths],
            "### Matched foreground lines at both sides of every gap",
            "sage_matched_line_overlays.png",
        )
        display_path_contact_sheet(
            condition_preview_paths,
            [path.parent.parent.name for path in condition_preview_paths],
            "### Middle structural condition in every gap",
            "sage_middle_conditions.png",
        )
        print("Prepared SAGE guides:", SAGE_OUTPUT_ROOT)
        """
    ),
    markdown(
        """
        ## 11. Render one SAGE round and assemble the circular video

        FCVG is loaded once and retained across all gaps. Each completed gap is
        written immediately, with a fingerprint over endpoints, masks, model
        revisions, checkpoints, and settings. Reruns reuse only exact matches.
        The final sequence stores each exact left anchor followed by its 13 SAGE
        frames; the target anchor is the next gap's exact left anchor, avoiding
        duplicates. The last gap returns to anchor zero.
        """
    ),
    code(
        """
        from IPython.display import Video

        if RUN_SAGE_RENDER:
            subprocess.check_call(
                [*SAGE_COMMAND_BASE, "--phase", "render"],
                env=sage_environment,
            )
        else:
            print("SAGE render intentionally stopped after guide inspection.")

        SAGE_SEQUENCE_MANIFEST_PATH = SAGE_OUTPUT_ROOT / "sage_sequence_manifest.json"
        if SAGE_SEQUENCE_MANIFEST_PATH.is_file():
            SAGE_SEQUENCE = json.loads(
                SAGE_SEQUENCE_MANIFEST_PATH.read_text(encoding="utf-8")
            )
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
                "### Generated SAGE midpoint from every circular gap",
                "sage_generated_midpoints.png",
            )
            SAGE_FINAL_VIDEO_PATH = Path(SAGE_SEQUENCE["final_video_path"])
            display(Markdown("## Final one-round cyclic SAGE video"))
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
                "cyclic": SAGE_SEQUENCE["cyclic"],
                "still_image_adaptation": SAGE_SEQUENCE["still_image_adaptation"],
            })
            if DOWNLOAD_FINAL_VIDEO:
                from google.colab import files
                files.download(str(SAGE_FINAL_VIDEO_PATH))
        elif RUN_SAGE_RENDER:
            raise FileNotFoundError(SAGE_SEQUENCE_MANIFEST_PATH)
        """
    ),
    markdown(
        """
        ## What this trial does—and does not—claim

        This is a faithful SAGE **structural synthesis** trial for our prompt-only
        still workflow, but not an exact reproduction of the paper's clip-to-clip
        motion input. With actual clips, SAGE uses SEA-RAFT over nearby frames to
        determine the two global flow directions. A still image has no such
        measurement, so this notebook records its synthetic spline control in
        every manifest rather than presenting it as inferred motion.

        The direct SAGE frames are already video-diffusion outputs, so this first
        notebook does not add RIFE. That keeps the experiment diagnostic: any
        pacing or structural behavior seen in the MP4 belongs to SAGE/FCVG rather
        than a second interpolator.
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"sage-{index:02d}"
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook = {
    "cells": cells,
    "metadata": copy.deepcopy(source_notebook.get("metadata", {})),
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
