"""Build the local recursive vision-interpolation art notebook.

The generated notebook is intentionally ignored by Git. Keeping the builder
tracked makes the notebook reproducible without publishing keys, outputs, or
the user's editable working copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "StillLife_Recursive_Vision_Interpolation.ipynb"


def _lines(source: str) -> list[str]:
    clean = dedent(source).strip("\n") + "\n"
    return clean.splitlines(keepends=True)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


cells = [
    markdown(
        r"""
        # Recursive science still-life loop — image-aware prompt interpolation

        This local working notebook generates a closed sequence without an uploaded prompt JSON and without the old twenty-prompt FlowMorph schedules.

        1. Edit the anchor sciences and prompts directly in section 2.
        2. Generate the anchor paintings with weak continuity conditioning.
        3. For every cyclic neighbor pair, send both generated paintings plus their science descriptions to an OpenAI vision model. It returns one or more literal, self-contained midpoint prompts that interpolate the subject matter, object correspondences, layout, light, color, and materials.
        4. Generate those midpoint paintings from the new prompts and a very soft blurred structural mixture of the two endpoint images.
        5. Repeat the midpoint process on the denser sequence. Defaults: 15 → 30 → 60 images.
        6. Finish the duplicate-free cyclic sequence with Practical-RIFE, circular SSIM motion equalization, and H.264 export.

        Google Drive is mounted first. The OpenAI key is read from a standalone text file in the Drive project base directory, never printed, never placed in an environment variable, and never copied into run outputs. When Drive persistence is enabled, the auto-numbered timestamped run directory itself lives on Drive, so every completed image and manifest is persistent as soon as it is written.
        """
    ),
    markdown(
        r"""
        ## 1. Editable run, model, API, image, and video settings

        The recursive count grows quickly. With `N` anchors, `M` inserted images per gap, and `R` rounds, the final sequence contains `N × (M + 1)^R` images. The default is `15 × 2² = 60` FLUX images and 45 OpenAI vision calls.

        `MIDPOINT_REFERENCE_STRENGTH` is deliberately low. If generated images look double-exposed or alpha-blended, lower it toward `0.04`, increase the blur, or disable midpoint image conditioning; the LLM prompt still carries the visual correspondence.
        """
    ),
    code(
        r"""
        PROJECT_ROOT = "/content/FlowMorphKlein9B"
        REPOSITORY_URL = "https://github.com/MNoichl/FluxFlowMorph.git"
        UPDATE_REPOSITORY = True
        PROJECT_NAME = "science_path_recursive_vision"
        LOCAL_ASSET_ROOT = "/content/flowmorph_recursive_art"
        HF_CACHE_DIR = "/content/hf_cache"

        # Drive and secret. Put only the token text in this file (one line).
        MOUNT_DRIVE = True
        DRIVE_PROJECT_BASE = "/content/drive/MyDrive/FluxFlowMorphArt"
        OPENAI_KEY_FILENAME = "openai_api_key.txt"
        RESUME_RUN_DIRECTORY = None  # Example: "/content/drive/MyDrive/FluxFlowMorphArt/science_path_recursive_vision/..."

        # OpenAI vision prompt generation.
        OPENAI_MODEL = "gpt-5.6"
        OPENAI_REASONING_EFFORT = "medium"
        OPENAI_IMAGE_DETAIL = "high"  # "low", "high", "original", or "auto"
        # Includes hidden reasoning plus visible structured JSON. 5000 avoids
        # cutting a valid proposal off in the middle of its prompt string.
        OPENAI_MAX_OUTPUT_TOKENS = 5000
        OPENAI_MAX_ATTEMPTS = 3
        VISION_IMAGE_MAX_SIDE = 1024
        VISION_JPEG_QUALITY = 90

        # Editable anchor selection and recursive insertion.
        BASE_PROMPT_COUNT = 15
        INTERPOLATION_ROUNDS = 2
        MIDPOINTS_PER_GAP = 1
        REGENERATE_BASE_FRAMES = True
        REUSE_EXISTING_MIDPOINTS = True

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
        IMAGE_INFERENCE_STEPS = 28
        IMAGE_GUIDANCE_SCALE = 4.0
        IMAGE_LORA_SCALE = 1.0
        BASE_SEED = 1729

        # Weak continuity for anchors; pair conditioning for recursive midpoints.
        BASE_CONTINUITY_ENABLED = True
        BASE_REFERENCE_STRENGTH = 0.12
        BASE_REFERENCE_BLUR = 16.0
        BASE_REFERENCE_GRAIN_STRENGTH = 0.035  # Normalized monochrome noise sigma; 0 disables.
        MIDPOINT_CONDITIONING_ENABLED = True
        MIDPOINT_REFERENCE_STRENGTH = 0.08
        MIDPOINT_REFERENCE_BLUR = 18.0
        REFERENCE_BACKGROUND = (116, 105, 91)
        SAVE_SOFT_REFERENCES = False

        # Trial and notebook display.
        RUN_TRIAL_KEYFRAME = True
        TRIAL_KEYFRAME_INDEX = None  # None chooses randomly; otherwise 0..BASE_PROMPT_COUNT-1.
        TRIAL_SEED = None
        TRIAL_DISPLAY_MAX_WIDTH = 768
        CONTACT_SHEET_COLUMNS = 8
        CONTACT_SHEET_DISPLAY_MAX_WIDTH = 1100
        LOOP_PREVIEW_DISPLAY_WIDTH = 768
        LOOP_PREVIEW_RENDER_MAX_SIDE = 512  # Reduced streaming preview; source PNGs stay untouched.

        # Cyclic sequence and RIFE/SSIM finishing.
        SOURCE_SEQUENCE_FPS = 12.0
        LOOP_AUTO_ROTATE_TO_QUIETEST_CUT = True
        LOOP_SEAM_ANALYSIS_SIZE = 192
        RUN_RIFE_POSTPROCESS = True
        RIFE_REPOSITORY_URL = "https://github.com/hzwer/Practical-RIFE.git"
        RIFE_REPOSITORY_REVISION = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
        RIFE_ROOT = "/content/Practical-RIFE"
        RIFE_MODEL_REPOSITORY = "Bash2X/RIFE-Models"
        RIFE_MODEL_REVISION = "feaf6d11238b4a1e9f015a5d18c18df152affd20"
        RIFE_MODEL_FILENAME = "RIFE_v4.25.zip"
        RIFE_MULTIPLIER = 4
        RIFE_SCALE = 1.0
        RIFE_USE_FP16 = True
        RIFE_RETRY_WITH_FP32 = True  # Automatically retry once when the fp16 runner fails.
        RIFE_FINAL_FPS = 24.0
        RIFE_SSIM_ANALYSIS_SIZE = 192
        RIFE_SSIM_WEIGHT_FLOOR = 1e-6
        RIFE_VIDEO_CRF = 16
        RIFE_KEEP_WORK_FRAMES = False
        RIFE_DISPLAY_WIDTH = 768
        DOWNLOAD_FINAL_VIDEO = False
        """
    ),
    markdown(
        r"""
        ## 2. Editable anchor sciences and prompts

        Edit these dictionaries directly. `science` is sent to the vision model as conceptual context; `prompt` is sent to FLUX. Every prompt must be a literal visual description and must contain the LoRA trigger `RIJKSOIL`. Avoid production-language such as “bridge frame,” “keep,” “same,” or “transition.”
        """
    ),
    code(
        r"""
        BASE_STAGES = [
            {
                "id": "nuclear_atomic_optical_physics",
                "science": "nuclear and high-energy physics; atomic and molecular physics; optics",
                "prompt": "RIJKSOIL, a medium-wide low three-quarter Dutch Baroque still life rising diagonally from a black laboratory plinth into a stone alcove; a brass cloud chamber beneath a misted glass bell with pale particle tracks; a dark ore specimen in a dull lead cradle; a cut-glass prism catching a narrow muted spectrum; paired brass lenses, a sealed vapor ampoule, an ivory counter dial and a loose arc of copper detector wire; cold upper-left light answered by a low amber glow, pronounced tenebrism, soot black, lead gray, oxidized brass, luminous glass, layered oil glazes and restrained impasto; no people, no readable text.",
            },
            {
                "id": "electronic_magnetic_materials",
                "science": "electronic, optical and magnetic materials; materials chemistry",
                "prompt": "RIJKSOIL, a medium-wide lateral Baroque arrangement of broad concentric arcs on a polished slate shelf; a cobalt silicon wafer tilted against a low brass rest; an enamelled copper coil encircling a dark horseshoe magnet; translucent calcite balanced by stepped ferrite tiles; a short fiber-optic strand releasing a few pale points; dark teal silk falling in monumental folds, cool light gathering into warm copper reflections, sculptural chiaroscuro, mineral surfaces, broad brushwork and glazed highlights; no people, no readable text.",
            },
            {
                "id": "mechanics_ocean_aerospace_control",
                "science": "mechanics and computational mechanics; ocean engineering; aerospace, electrical and control systems engineering",
                "prompt": "RIJKSOIL, a medium-wide Baroque workshop composition swept by a wing-shaped diagonal above a shallow pewter basin; a brass gyroscope inside its circular gimbal, a small airfoil raised on pins, a steel gear crossed by calipers, a copper-wound servo coupled to a feedback pendulum, and a carved wave crest beside a rolled salt-stained chart; storm-blue canvas and charcoal wool form large shadowed planes; hard left light traces rivets, wet pewter, scratched steel and oil-dark brass with vigorous loaded brushwork; no people, no readable text.",
            },
            {
                "id": "manufacturing_networks",
                "science": "industrial and manufacturing engineering; computer networks and communications",
                "prompt": "RIJKSOIL, a medium-wide low Baroque composition carrying a chain of mechanisms across an oil-darkened cast-iron plate; an articulated gripper poised over a precision gear train ending in a polished bearing; a punched brass card joined to woven copper cable; cream ceramic signal insulators rhythmically crossing the rear edge; a heavy brown curtain billows into cavernous shadow while a high left glint breaks across steel, oily brass, woven wire and chalky ceramic, coarse impasto and monumental repetition; no people, no readable text.",
            },
            {
                "id": "mathematics_computation",
                "science": "mathematics; computational theory; geometry and topology",
                "prompt": "RIJKSOIL, a medium-wide cabinet-like scholarly Baroque still life unfolding around open brass compasses on a broad chalk-dusted slate; a wooden polyhedron, faint non-readable geometric diagrams, ivory counting rods, exposed calculator wheels and a dark topology loop over folded graph parchment; moss-green baize and aged parchment form quiet vertical layers; angled candlelight, measured geometry, slate black, ivory, worn brass and wood, contemplative chiaroscuro and softly glazed surfaces; no people, no readable text.",
            },
            {
                "id": "computer_science_ai_vision",
                "science": "computer science; artificial intelligence; computer vision and pattern recognition; information systems",
                "prompt": "RIJKSOIL, a medium-wide symmetrical Baroque nocturne built around an antique camera lens like a mechanical eye; layered cobalt circuit boards rise behind its toothed blackened-brass housing; cream punched cards meet a glass field of restrained square lights while branching gold conductors spread across the lower plane; a cool square illumination from the left balances one warm copper gleam, lacquer blue, amber glass, centralized drama, deep glazing and luminous accents; no people, no readable text.",
            },
            {
                "id": "operations_economics",
                "science": "management science and operations research; economics and econometrics; accounting",
                "prompt": "RIJKSOIL, a medium-wide Dutch Golden Age merchant-table composition ascending from coin stacks to a brass balance beam; an oxblood leather ledger lies open on a shallow writing slope beside a dark abacus, cargo miniatures, a clear sand timer and folded sheets bearing non-readable curves; tobacco-brown drapery gathers into one generous fold; warm candlelight multiplies across tarnished silver, copper, rubbed leather, paper and dark wood in pyramidal order and sober chiaroscuro; no people, no readable text.",
            },
            {
                "id": "strategy_politics_relations",
                "science": "strategy and management; political science and international relations",
                "prompt": "RIJKSOIL, a medium-wide courtly Baroque still life leading opposing ebony and ivory chess pieces toward a small terrestrial globe; a brass compass opens over an unreadable coastal chart on a cherrywood campaign box; treaty ribbons, red sealing wax and restrained crimson threads connect colored map pins; dark carmine damask swells behind the globe, theatrical left candlelight catches wax, silk and brass, dramatic diagonals and sumptuous glazing; no people, no readable text.",
            },
            {
                "id": "sociology_philosophy",
                "science": "sociology; political science; philosophy, knowledge and ethics",
                "prompt": "RIJKSOIL, a medium-wide civic vanitas arranged around a shallow pewter bowl of voting tokens on a cracked black-marble ledge; clustered wooden figures of varied heights stand among census tally sticks, three linked rings and an open illegible leather book weighted by a river stone; a dark convex mirror and small brass balance catch one severe beeswax candle; smoke-gray linen and olive velvet descend into enveloping shadow, worn wood, fibrous paper, dull pewter and grave translucent glazes; no people, no readable text.",
            },
            {
                "id": "psychology_cognitive_science",
                "science": "clinical and social psychology; psychiatry and mental health; cognitive neuroscience",
                "prompt": "RIJKSOIL, a medium-wide asymmetrical Baroque arrangement orbiting a pale ivory wax brain and a reflected theatrical mask; a wooden maze aligns with a slender metronome, ambiguous ink cards scatter among memory beads, and a silver tuning fork crosses the foreground; plum felt, pale maple and a dusky violet curtain open onto a narrow black recess; soft divided light, theatrical doubling, velvety shadows and layered oil color; no people, no readable text.",
            },
            {
                "id": "public_environmental_health",
                "science": "public, environmental and occupational health; epidemiology; general health professions",
                "prompt": "RIJKSOIL, a medium-wide field-kit Baroque still life spreading practical instruments in a calm arc from an opened galvanized case; a brass air-sampling pump and pleated filter, a small respirator, worn leather glove, clear water vial, silver thermometer and an epidemiological map with colored pins but no labels; deep green canvas rises behind them with dust and one water stain; clear left window light reveals particles across metal, fabric and glass, earthy realism and weighty forms; no people, no readable text.",
            },
            {
                "id": "neuroscience_physiology_cardiovascular",
                "science": "neuroscience and neurology; physiology; endocrinology, diabetes and metabolism; cardiology and cardiovascular medicine",
                "prompt": "RIJKSOIL, a medium-wide anatomical Baroque arc joining an ivory wax brain to refined wax models of a heart and paired lungs; a delicate electrode crown sends red and blue nerve threads toward a coiled brass stethoscope, a clear insulin vial, a reflex hammer and a ruby pulse watch; indigo cloth crosses a burgundy leather case under warm silver light, non-gory sculptural modeling, deep recession, luminous glass and humane layered oil glazes; no people, no readable text.",
            },
            {
                "id": "oncology_immunity_pathology",
                "science": "cancer research and oncology; hematology; immunology; pathology and forensic medicine",
                "prompt": "RIJKSOIL, a medium-wide non-gory laboratory vanitas rising from a ruby glass dish toward an angled brass microscope; translucent red droplets, pathology slides, branching ivory antibody forms and pale cell spheres gather around a closed black specimen box; clear glass rests over dark crimson cloth against a black-burgundy recess; sharp left light turns the microscope rim gold and the slides luminous, cavernous shadow, transparent glazes and precise impasto; no people, no readable text.",
            },
            {
                "id": "genetics_evolution_ecology",
                "science": "molecular and cell biology; genetics; infectious diseases; evolution, ecology, behavior, food and plant science",
                "prompt": "RIJKSOIL, a medium-wide naturalist's Baroque crescent sweeping from a glass double helix and abstract petri colonies toward a fossil ammonite, spiral shells, a pressed fern, seed pods and a sliced heritage pear; translucent cell vesicles mingle with a pale finch skull and dark beetle on a weathered sandstone shelf; forest-brown and green-black drapery frames cool glass and autumnal fruit, tactile bone, ribbed shell, leaf, seed and moist flesh in layered glazes; no people, no readable text.",
            },
            {
                "id": "toxicology_chemistry_sustainable_materials",
                "science": "health, toxicology and mutagenesis; chemistry and spectroscopy; biomaterials; polymers; water science; renewable energy and sustainability; materials chemistry",
                "prompt": "RIJKSOIL, a medium-wide vertical alchemical Baroque still life rising through a coiled glass alembic with amber reagent drops and descending across a spectroscopy prism, charred leaf, clear polymer film, porous biomaterial mesh, blue solar cell, copper battery plate and pure-water vial; pale soapstone bears old amber rings beneath burnt-orange fabric and a tar-black wall; firelit left illumination refracts through glass and oxidized copper, rich layered paint and fiery chiaroscuro; no people, no readable text.",
            },
        ]
        """
    ),
    markdown(
        r"""
        ## 3. GPU, repository, and compatible dependencies

        This checks the actual imports needed by the notebook. It does not reject a healthy Diffusers install merely because editable-install provenance metadata is absent. A core reinstall happens only if a clean Python process cannot import the FLUX.2 Klein pipeline and required packages; in that case restart the kernel once after installation.
        """
    ),
    code(
        r"""
        import platform
        import subprocess
        import sys
        from pathlib import Path

        print({"python": sys.version, "platform": platform.platform()})
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is missing; use a Colab GPU runtime and rerun this cell.") from error
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU runtime is required.")
        print({"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda})

        project_path = Path(PROJECT_ROOT)
        if not (project_path / "pyproject.toml").is_file():
            subprocess.check_call(["git", "clone", "--depth", "1", REPOSITORY_URL, PROJECT_ROOT])
        elif UPDATE_REPOSITORY:
            subprocess.check_call(["git", "-C", PROJECT_ROOT, "pull", "--ff-only"])

        core_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import numpy, scipy, transformers, pydantic; from diffusers import Flux2KleinPipeline",
            ],
            capture_output=True,
            text=True,
        )
        if core_probe.returncode != 0:
            print("Installing the notebook's pinned FLUX environment because the clean import probe failed:")
            print(core_probe.stderr[-2000:])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r",
                str(project_path / "requirements-colab.txt"),
            ])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", PROJECT_ROOT])
            raise RuntimeError(
                "Dependencies installed successfully. Restart the notebook kernel once, then rerun from section 1."
            )

        try:
            import openai
            from openai import OpenAI
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=2,<3"])
            import openai
            from openai import OpenAI

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
            "openai_sdk": openai.__version__,
        })
        """
    ),
    markdown(
        r"""
        ## 4. Mount Drive, reserve the run directory, and load the API key

        Create `openai_api_key.txt` directly inside `DRIVE_PROJECT_BASE`. The file should contain only the API key and a final newline is optional. It is read into the OpenAI client and then the temporary string is deleted. The notebook prints the path it read, never the key or any key fragment.
        """
    ),
    code(
        r"""
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

        for child in ("base_frames", "trials", "rounds", "previews", "video", "metadata"):
            (RUN_DIRECTORY / child).mkdir(parents=True, exist_ok=True)
        Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

        if not DRIVE_ENABLED:
            raise RuntimeError(
                "This notebook's OpenAI key workflow expects Google Drive. Set MOUNT_DRIVE=True."
            )
        OPENAI_KEY_PATH = drive_base / OPENAI_KEY_FILENAME
        if not OPENAI_KEY_PATH.is_file():
            raise FileNotFoundError(
                f"Create {OPENAI_KEY_PATH} with only your OpenAI API key, then rerun this cell."
            )
        _openai_key = OPENAI_KEY_PATH.read_text(encoding="utf-8").strip()
        if len(_openai_key) < 20 or any(character.isspace() for character in _openai_key):
            raise ValueError("The Drive key file is empty or malformed; expected one token with no spaces.")
        OPENAI_CLIENT = OpenAI(api_key=_openai_key)
        del _openai_key

        run_identity = {
            "project": PROJECT_NAME,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "persistent": DRIVE_ENABLED,
            "run_directory": str(RUN_DIRECTORY),
            "openai_model": OPENAI_MODEL,
            "key_file": str(OPENAI_KEY_PATH),
            "key_value_recorded": False,
        }
        (RUN_DIRECTORY / "metadata" / "run_identity.json").write_text(
            json.dumps(run_identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("OpenAI client initialized from the Drive key file (credential value not displayed).")
        print("Run directory:", RUN_DIRECTORY)
        print("Every generated image and manifest is written directly into this persistent directory.")
        """
    ),
    markdown(
        r"""
        ## 5. Validate settings and preview the recursive cost

        The validation also catches accidental missing or duplicated LoRA triggers in anchor prompts. It does not rewrite your text.
        """
    ),
    code(
        r"""
        if not 3 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):
            raise ValueError(f"BASE_PROMPT_COUNT must be between 3 and {len(BASE_STAGES)}")
        if not 0 <= INTERPOLATION_ROUNDS <= 4:
            raise ValueError("INTERPOLATION_ROUNDS must be between 0 and 4")
        if not 1 <= MIDPOINTS_PER_GAP <= 4:
            raise ValueError("MIDPOINTS_PER_GAP must be between 1 and 4")
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
        for name, strength in (
            ("BASE_REFERENCE_STRENGTH", BASE_REFERENCE_STRENGTH),
            ("MIDPOINT_REFERENCE_STRENGTH", MIDPOINT_REFERENCE_STRENGTH),
        ):
            if not 0 < strength <= 0.35:
                raise ValueError(f"{name} must lie in (0, 0.35]")
        if not 0 <= BASE_REFERENCE_GRAIN_STRENGTH <= 0.25:
            raise ValueError("BASE_REFERENCE_GRAIN_STRENGTH must lie in [0, 0.25]")
        if OPENAI_IMAGE_DETAIL not in {"low", "high", "original", "auto"}:
            raise ValueError("OPENAI_IMAGE_DETAIL must be low, high, original, or auto")

        ACTIVE_BASE_STAGES = BASE_STAGES[:BASE_PROMPT_COUNT]
        ids = [item["id"] for item in ACTIVE_BASE_STAGES]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_]+", item) for item in ids):
            raise ValueError("Anchor IDs must be unique lowercase snake_case values")
        for item in ACTIVE_BASE_STAGES:
            if not item["science"].strip() or not item["prompt"].strip():
                raise ValueError(f"Blank science or prompt in {item['id']}")
            trigger_count = item["prompt"].casefold().count(LORA_TRIGGER.casefold())
            if trigger_count != 1:
                raise ValueError(f"{item['id']} must contain the LoRA trigger exactly once")

        round_counts = [BASE_PROMPT_COUNT]
        for _ in range(INTERPOLATION_ROUNDS):
            round_counts.append(round_counts[-1] * (MIDPOINTS_PER_GAP + 1))
        openai_calls = round_counts[-1] - round_counts[0]
        print({
            "anchor_images": BASE_PROMPT_COUNT,
            "sequence_counts": round_counts,
            "openai_vision_calls": openai_calls,
            "total_flux_images": round_counts[-1],
            "cyclic_gaps_per_round": round_counts[:-1],
        })
        print("Anchor order:", " → ".join(ids), "→", ids[0])
        """
    ),
    markdown(
        r"""
        ## 6. Load and fuse the RIJKSOIL LoRA; optional trial image

        LoRA weights are fused into the transformer before CPU offload. This avoids the CPU/CUDA matrix mismatch that can occur when PEFT adapter matrices remain attached during repeated offloaded calls. Rerunning this cell reuses a same-scale pipeline and rebuilds it if the scale changed.
        """
    ),
    code(
        r"""
        import gc
        import os
        import random
        import shutil
        from huggingface_hub import hf_hub_download
        from IPython.display import Markdown, display
        from PIL import Image, ImageFilter
        from flowmorph_klein.lora import load_flux2_lora

        try:
            import peft.tuners.lora.torchao as peft_torchao_dispatch
        except ImportError:
            peft_torchao_dispatch = None
        else:
            peft_torchao_dispatch.is_torchao_available = lambda: False

        downloaded_lora = Path(hf_hub_download(
            repo_id=LORA_SOURCE,
            filename=LORA_WEIGHT_NAME,
            revision=LORA_REVISION,
            cache_dir=HF_CACHE_DIR,
        ))
        lora_stage_directory = Path(HF_CACHE_DIR) / "flowmorph_lora_files" / LORA_REVISION[:12]
        lora_stage_directory.mkdir(parents=True, exist_ok=True)
        LOCAL_LORA_PATH = lora_stage_directory / LORA_WEIGHT_NAME
        if not LOCAL_LORA_PATH.is_file():
            try:
                os.link(downloaded_lora.resolve(), LOCAL_LORA_PATH)
            except OSError:
                shutil.copy2(downloaded_lora, LOCAL_LORA_PATH)
        if LOCAL_LORA_PATH.stat().st_size != downloaded_lora.stat().st_size:
            raise RuntimeError(f"Staged LoRA size mismatch at {LOCAL_LORA_PATH}")

        def release_flux_pipeline():
            previous = globals().pop("FLUX_PIPE", None)
            globals().pop("FLUX_PIPE_LORA_SCALE", None)
            if previous is not None:
                maybe_free = getattr(previous, "maybe_free_model_hooks", None)
                if callable(maybe_free):
                    maybe_free()
                del previous
                gc.collect()
                torch.cuda.empty_cache()

        def load_flux_pipeline():
            pipeline = Flux2KleinPipeline.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=HF_CACHE_DIR,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            report = load_flux2_lora(
                pipeline,
                str(LOCAL_LORA_PATH),
                adapter_name=LORA_ADAPTER_NAME,
                scale=IMAGE_LORA_SCALE,
                require_base_9b_provenance=False,
                allow_distilled_9b=True,
            )
            pipeline.fuse_lora(
                components=["transformer"],
                lora_scale=1.0,
                safe_fusing=True,
                adapter_names=[LORA_ADAPTER_NAME],
            )
            pipeline.unload_lora_weights()
            remaining = [
                name for name, _ in pipeline.transformer.named_parameters()
                if "lora_" in name.casefold() or ".lora" in name.casefold()
            ]
            if remaining:
                raise RuntimeError("LoRA fusion left runtime parameters: " + ", ".join(remaining[:5]))
            pipeline.enable_model_cpu_offload()
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
            return pipeline, report

        if "FLUX_PIPE" in globals() and globals().get("FLUX_PIPE_LORA_SCALE") != float(IMAGE_LORA_SCALE):
            print("LoRA scale changed; rebuilding the fused pipeline.")
            release_flux_pipeline()
        if "FLUX_PIPE" not in globals():
            FLUX_PIPE, LORA_REPORT = load_flux_pipeline()
            FLUX_PIPE_LORA_SCALE = float(IMAGE_LORA_SCALE)
            print("Loaded a device-safe fused-LoRA pipeline.")
        else:
            print("Reusing the fused pipeline at the current LoRA scale.")

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
            trial_result = FLUX_PIPE(
                prompt=trial_stage["prompt"],
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                num_inference_steps=IMAGE_INFERENCE_STEPS,
                guidance_scale=IMAGE_GUIDANCE_SCALE,
                generator=torch.Generator(device="cuda").manual_seed(trial_seed),
                output_type="pil",
            )
            trial_image = trial_result.images[0].convert("RGB")
            trial_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            trial_directory = RUN_DIRECTORY / "trials" / f"{trial_stamp}_{trial_stage['id']}_{trial_seed}"
            trial_directory.mkdir(parents=True, exist_ok=False)
            trial_path = trial_directory / "trial.png"
            trial_image.save(trial_path)
            (trial_directory / "settings.json").write_text(json.dumps({
                "stage": trial_stage,
                "seed": trial_seed,
                "lora_scale": IMAGE_LORA_SCALE,
                "guidance_scale": IMAGE_GUIDANCE_SCALE,
                "inference_steps": IMAGE_INFERENCE_STEPS,
                "size": [IMAGE_WIDTH, IMAGE_HEIGHT],
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            preview = trial_image.copy()
            preview.thumbnail((TRIAL_DISPLAY_MAX_WIDTH, TRIAL_DISPLAY_MAX_WIDTH))
            display(Markdown(f"### Trial anchor: `{trial_stage['id']}`"))
            display(preview)
            print({"path": str(trial_path), "seed": trial_seed, "prompt_index": trial_index})
            del trial_result, trial_image, preview
        else:
            print("Trial skipped.")
        """
    ),
    markdown(
        r"""
        ## 7. Generate the cyclic anchor paintings

        The first anchor is text-to-image. Later anchors receive only a faint blurred trace of the previous anchor. Images are not individually displayed; the following cell renders one compact contact sheet.
        """
    ),
    code(
        r"""
        from flowmorph_klein.art_loop import make_soft_reference

        BASE_DIRECTORY = RUN_DIRECTORY / "base_frames"
        BASE_MANIFEST_PATH = RUN_DIRECTORY / "metadata" / "base_manifest.json"
        BASE_RECORDS = []

        if not REGENERATE_BASE_FRAMES and BASE_MANIFEST_PATH.is_file():
            BASE_RECORDS = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))["records"]
            missing = [item["path"] for item in BASE_RECORDS if not Path(item["path"]).is_file()]
            if missing:
                raise FileNotFoundError("Missing resumed anchor images: " + ", ".join(missing))
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
                    "The editable anchor prompts differ from this run's saved anchors. "
                    "Set REGENERATE_BASE_FRAMES=True or resume the matching notebook settings."
                )
            print(f"Loaded {len(BASE_RECORDS)} existing anchor records.")
        else:
            previous = None
            for index, stage in enumerate(ACTIVE_BASE_STAGES):
                seed = BASE_SEED + index
                kwargs = {
                    "prompt": stage["prompt"],
                    "height": IMAGE_HEIGHT,
                    "width": IMAGE_WIDTH,
                    "num_inference_steps": IMAGE_INFERENCE_STEPS,
                    "guidance_scale": IMAGE_GUIDANCE_SCALE,
                    "generator": torch.Generator(device="cuda").manual_seed(seed),
                    "output_type": "pil",
                }
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
                    kwargs["image"] = reference
                    if SAVE_SOFT_REFERENCES:
                        reference_directory = BASE_DIRECTORY / "soft_references"
                        reference_directory.mkdir(parents=True, exist_ok=True)
                        reference_path = reference_directory / f"reference_{index:03d}.png"
                        reference.save(reference_path)
                result = FLUX_PIPE(**kwargs)
                if not result.images:
                    raise RuntimeError(f"FLUX returned no image for {stage['id']}")
                image = result.images[0].convert("RGB")
                output_path = BASE_DIRECTORY / f"{index:03d}_{stage['id']}.png"
                image.save(output_path, compress_level=4)
                record = {
                    "uid": f"base_{index:03d}",
                    "kind": "base",
                    "round": 0,
                    "science": stage["science"],
                    "prompt": stage["prompt"],
                    "seed": seed,
                    "path": str(output_path),
                    "soft_reference_path": str(reference_path) if reference_path else None,
                    "soft_reference_grain_strength": BASE_REFERENCE_GRAIN_STRENGTH,
                    "soft_reference_grain_seed": seed if previous is not None else None,
                }
                BASE_RECORDS.append(record)
                BASE_MANIFEST_PATH.write_text(json.dumps({
                    "records": BASE_RECORDS,
                    "complete": len(BASE_RECORDS) == len(ACTIVE_BASE_STAGES),
                }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                previous = image
                print(f"Anchor {index + 1}/{len(ACTIVE_BASE_STAGES)} saved: {output_path.name}")
                del result
            del previous

        if len(BASE_RECORDS) != len(ACTIVE_BASE_STAGES):
            raise RuntimeError("The anchor manifest is incomplete; regenerate or select the correct resume run.")
        print(f"Prepared {len(BASE_RECORDS)} cyclic anchors in {BASE_DIRECTORY}")
        """
    ),
    code(
        r"""
        from flowmorph_klein.visualization import make_contact_sheet

        base_contact_sheet_path = RUN_DIRECTORY / "previews" / "base_contact_sheet.png"
        base_images = [Image.open(item["path"]).convert("RGB") for item in BASE_RECORDS]
        make_contact_sheet(
            base_images,
            base_contact_sheet_path,
            columns=min(CONTACT_SHEET_COLUMNS, len(base_images)),
            labels=[item["uid"] for item in BASE_RECORDS],
        )
        for image in base_images:
            image.close()
        base_preview = Image.open(base_contact_sheet_path).convert("RGB")
        base_preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
        display(Markdown("### Anchor paintings — compact contact sheet"))
        display(base_preview)
        del base_preview, base_images
        print("Full-resolution anchors and contact sheet:", BASE_DIRECTORY)
        """
    ),
    markdown(
        r"""
        ## 8. Define the image-aware midpoint prompt contract

        Each API call receives both actual endpoint images, both literal prompts, both science descriptions, and the requested fractional position. The model returns structured fields, but only its standalone descriptive `prompt` is sent to FLUX. Explanatory fields are saved for audit.

        Semantic validation rejects common failure modes from the earlier hand-authored JSONs: production jargon, instructions to keep things “the same,” and missing/duplicated LoRA triggers. Failed semantic outputs are retried with a concise correction.
        """
    ),
    code(
        r'''
        import base64
        import hashlib
        import io
        import time
        from pydantic import BaseModel, Field, ValidationError

        class MidpointProposal(BaseModel):
            science_connection: str = Field(min_length=20, max_length=800)
            visual_correspondence: str = Field(min_length=20, max_length=1200)
            prompt: str = Field(min_length=300, max_length=2600)

        MIDPOINT_SYSTEM_PROMPT = f"""
        Role: You are an art director writing one prompt for FLUX.2 Klein with the {LORA_TRIGGER} oil-painting LoRA.

        Goal: Given two endpoint still-life paintings, their prompts, and their science descriptions, write the literal visual description of a painting at the requested fractional position from A to B. It must be a plausible interdisciplinary scientific still life and a genuine visual midpoint.

        Success criteria:
        - Inspect both images, not merely their text. Map major objects by position, silhouette, scale, orientation, material, color, lighting, negative space, and support geometry.
        - Transform each important correspondence by one small intelligible step. A vessel may change proportion/material/content toward the paired object; folds, arcs, lenses, coils, branches, handles, bowls, organs, instruments, and shadows may become thematically appropriate intermediate forms.
        - Connect the named sciences with concrete objects or processes. Do not invent a third unrelated scene.
        - Produce one self-contained descriptive image prompt. Repeat every visual fact that should appear; do not refer back to either endpoint.
        - Preserve the established medium-wide seventeenth-century Dutch Baroque still-life language, theatrical chiaroscuro, material specificity, layered oil glazes, restrained impasto, no people, and no readable text.
        - The prompt begins exactly with "{LORA_TRIGGER}," and contains that trigger exactly once.

        Prompt prohibitions: Do not use the words bridge, transition, intermediate, halfway, keep, retain, preserve, unchanged, same, source image, target image, left image, right image, endpoint, frame, interpolation, or morph. Do not issue editing commands. Do not introduce generic walnut tables or other stock furniture unless the visible supports in both images justify it.

        Output: science_connection briefly states the interdisciplinary logic; visual_correspondence briefly states the concrete A-to-B object and composition mappings; prompt contains only the final literal image description.
        """.strip()

        FORBIDDEN_PROMPT_TERMS = (
            "bridge", "transition", "intermediate", "halfway", "keep", "retain", "preserve",
            "unchanged", "same", "source image", "target image", "left image", "right image",
            "endpoint", "frame", "interpolation", "morph",
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

        def midpoint_request_fingerprint(left, right, fraction):
            contract = {
                "model": OPENAI_MODEL,
                "reasoning_effort": OPENAI_REASONING_EFFORT,
                "image_detail": OPENAI_IMAGE_DETAIL,
                "system_prompt": MIDPOINT_SYSTEM_PROMPT,
                "fraction": fraction,
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

        def extract_parsed_proposal(response):
            parsed = getattr(response, "output_parsed", None)
            if parsed is not None:
                return parsed
            refusal_messages = []
            for output in response.output:
                if output.type != "message":
                    continue
                for item in output.content:
                    if item.type == "refusal":
                        refusal_messages.append(item.refusal)
                    elif getattr(item, "parsed", None) is not None:
                        return item.parsed
            if refusal_messages:
                raise RuntimeError("OpenAI refused the midpoint request: " + " | ".join(refusal_messages))
            raise RuntimeError("OpenAI response contained no parsed midpoint proposal")

        def validate_midpoint_prompt(prompt):
            clean = " ".join(prompt.split())
            if not clean.startswith(f"{LORA_TRIGGER},"):
                raise ValueError(f"Prompt must begin exactly with {LORA_TRIGGER},")
            if clean.casefold().count(LORA_TRIGGER.casefold()) != 1:
                raise ValueError("Prompt must contain the LoRA trigger exactly once")
            found = [term for term in FORBIDDEN_PROMPT_TERMS if re.search(rf"\b{re.escape(term)}\b", clean, re.I)]
            if found:
                raise ValueError("Prompt contains production-language terms: " + ", ".join(found))
            return clean

        def propose_midpoint(left, right, fraction):
            request_text = f"""
            Requested position: {fraction:.6f} from painting A toward painting B.

            Painting A sciences: {left['science']}
            Painting A generation prompt: {left['prompt']}

            Painting B sciences: {right['science']}
            Painting B generation prompt: {right['prompt']}

            Inspect both attached paintings and return the structured proposal.
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
                            {"role": "system", "content": MIDPOINT_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": request_text + correction},
                                    {"type": "input_text", "text": "Painting A:"},
                                    {
                                        "type": "input_image",
                                        "image_url": image_data_url(left["path"]),
                                        "detail": OPENAI_IMAGE_DETAIL,
                                    },
                                    {"type": "input_text", "text": "Painting B:"},
                                    {
                                        "type": "input_image",
                                        "image_url": image_data_url(right["path"]),
                                        "detail": OPENAI_IMAGE_DETAIL,
                                    },
                                ],
                            },
                        ],
                        text_format=MidpointProposal,
                    )
                    proposal = extract_parsed_proposal(response)
                except (ValidationError, json.JSONDecodeError) as error:
                    last_error = error
                    correction = (
                        "\n\nThe previous response was truncated or was not complete valid JSON. "
                        "Return a shorter complete response: concise audit fields and a literal image "
                        "prompt below 1,200 characters. Close every JSON string and object."
                    )
                    if attempt < OPENAI_MAX_ATTEMPTS:
                        time.sleep(min(2 ** (attempt - 1), 4))
                        continue
                    raise RuntimeError(
                        f"OpenAI returned incomplete structured JSON after {attempt} attempts"
                    ) from error
                try:
                    clean_prompt = validate_midpoint_prompt(proposal.prompt)
                except ValueError as error:
                    last_error = error
                    correction = (
                        f"\n\nThe previous result failed semantic validation: {error}. "
                        "Return a newly written literal prompt satisfying every prohibition."
                    )
                    if attempt < OPENAI_MAX_ATTEMPTS:
                        time.sleep(min(2 ** (attempt - 1), 4))
                        continue
                    raise
                proposal.prompt = clean_prompt
                return proposal, response
            raise RuntimeError(f"Midpoint generation failed: {last_error}")

        def pair_soft_reference(left_path, right_path, fraction):
            with Image.open(left_path) as opened:
                left = opened.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            with Image.open(right_path) as opened:
                right = opened.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            left = left.filter(ImageFilter.GaussianBlur(MIDPOINT_REFERENCE_BLUR))
            right = right.filter(ImageFilter.GaussianBlur(MIDPOINT_REFERENCE_BLUR))
            structural_mix = Image.blend(left, right, fraction)
            background = Image.new("RGB", structural_mix.size, REFERENCE_BACKGROUND)
            return Image.blend(background, structural_mix, MIDPOINT_REFERENCE_STRENGTH)

        print("Image-aware structured midpoint contract ready.")
        '''
    ),
    markdown(
        r"""
        ## 9. Run recursive midpoint prompt and image generation

        Every round treats the sequence as circular, including the final-to-first gap. Round 2 therefore sees the actual round-1 paintings and descriptions, not just the original anchors. Each successful proposal is saved before FLUX generation; if generation is interrupted, rerunning with `REUSE_EXISTING_MIDPOINTS=True` avoids paying for completed API calls again.
        """
    ),
    code(
        r"""
        def usage_payload(response):
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            return usage.model_dump(mode="json") if hasattr(usage, "model_dump") else str(usage)

        CURRENT_RECORDS = list(BASE_RECORDS)
        ROUND_MANIFESTS = []
        for round_number in range(1, INTERPOLATION_ROUNDS + 1):
            round_directory = RUN_DIRECTORY / "rounds" / f"round_{round_number:02d}"
            image_directory = round_directory / "images"
            proposal_directory = round_directory / "proposals"
            reference_directory = round_directory / "soft_references"
            for directory in (image_directory, proposal_directory):
                directory.mkdir(parents=True, exist_ok=True)
            if SAVE_SOFT_REFERENCES:
                reference_directory.mkdir(parents=True, exist_ok=True)

            incoming = list(CURRENT_RECORDS)
            outgoing = []
            gap_count = len(incoming)
            for gap_index, left in enumerate(incoming):
                right = incoming[(gap_index + 1) % gap_count]
                outgoing.append(left)
                for midpoint_index in range(1, MIDPOINTS_PER_GAP + 1):
                    fraction = midpoint_index / (MIDPOINTS_PER_GAP + 1)
                    uid = f"r{round_number:02d}_g{gap_index:04d}_m{midpoint_index:02d}"
                    proposal_path = proposal_directory / f"{uid}.json"
                    output_path = image_directory / f"{uid}.png"
                    request_fingerprint, request_contract = midpoint_request_fingerprint(
                        left, right, fraction
                    )
                    reused_proposal = False

                    if REUSE_EXISTING_MIDPOINTS and proposal_path.is_file():
                        saved = json.loads(proposal_path.read_text(encoding="utf-8"))
                        if saved.get("request_fingerprint") == request_fingerprint:
                            proposal = MidpointProposal.model_validate(saved["proposal"])
                            response_id = saved.get("openai_response_id")
                            usage = saved.get("usage")
                            reused_proposal = True
                            print(f"Reusing endpoint-verified saved prompt {uid}")
                        else:
                            print(f"Endpoint or prompt contract changed; regenerating {uid}")
                    if not reused_proposal:
                        proposal, response = propose_midpoint(left, right, fraction)
                        response_id = response.id
                        usage = usage_payload(response)
                        proposal_path.write_text(json.dumps({
                            "uid": uid,
                            "round": round_number,
                            "gap_index": gap_index,
                            "fraction": fraction,
                            "left_uid": left["uid"],
                            "right_uid": right["uid"],
                            "left_science": left["science"],
                            "right_science": right["science"],
                            "left_prompt": left["prompt"],
                            "right_prompt": right["prompt"],
                            "request_fingerprint": request_fingerprint,
                            "request_contract": request_contract,
                            "proposal": proposal.model_dump(mode="json"),
                            "openai_model": OPENAI_MODEL,
                            "openai_response_id": response_id,
                            "usage": usage,
                            "image_inputs_stored_in_manifest": False,
                        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                        print(f"OpenAI prompt {gap_index + 1}/{gap_count}, midpoint {midpoint_index}/{MIDPOINTS_PER_GAP}: {uid}")

                    seed = BASE_SEED + round_number * 100000 + gap_index * 100 + midpoint_index
                    if not (reused_proposal and output_path.is_file()):
                        kwargs = {
                            "prompt": proposal.prompt,
                            "height": IMAGE_HEIGHT,
                            "width": IMAGE_WIDTH,
                            "num_inference_steps": IMAGE_INFERENCE_STEPS,
                            "guidance_scale": IMAGE_GUIDANCE_SCALE,
                            "generator": torch.Generator(device="cuda").manual_seed(seed),
                            "output_type": "pil",
                        }
                        reference_path = None
                        if MIDPOINT_CONDITIONING_ENABLED:
                            reference = pair_soft_reference(left["path"], right["path"], fraction)
                            kwargs["image"] = reference
                            if SAVE_SOFT_REFERENCES:
                                reference_path = reference_directory / f"{uid}.png"
                                reference.save(reference_path)
                        result = FLUX_PIPE(**kwargs)
                        if not result.images:
                            raise RuntimeError(f"FLUX returned no midpoint image for {uid}")
                        result.images[0].convert("RGB").save(output_path, compress_level=4)
                        del result
                        print(f"Saved midpoint image: {output_path.name}")
                    else:
                        reference_path = None
                        print(f"Reusing saved midpoint image {uid}")

                    outgoing.append({
                        "uid": uid,
                        "kind": "midpoint",
                        "round": round_number,
                        "fraction": fraction,
                        "left_uid": left["uid"],
                        "right_uid": right["uid"],
                        "science": proposal.science_connection,
                        "visual_correspondence": proposal.visual_correspondence,
                        "prompt": proposal.prompt,
                        "seed": seed,
                        "path": str(output_path),
                        "proposal_path": str(proposal_path),
                        "soft_reference_path": str(reference_path) if reference_path else None,
                        "openai_response_id": response_id,
                        "usage": usage,
                    })

            CURRENT_RECORDS = outgoing
            round_manifest_path = round_directory / "sequence_manifest.json"
            round_manifest_path.write_text(json.dumps({
                "round": round_number,
                "cyclic": True,
                "input_count": len(incoming),
                "midpoints_per_gap": MIDPOINTS_PER_GAP,
                "output_count": len(outgoing),
                "records": outgoing,
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            ROUND_MANIFESTS.append(str(round_manifest_path))

            round_contact_sheet = round_directory / "contact_sheet.png"
            round_images = [Image.open(item["path"]).convert("RGB") for item in outgoing]
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
            display(Markdown(f"### Recursive round {round_number}: {len(outgoing)} cyclic images"))
            display(preview)
            del preview, round_images

        FINAL_RECORDS = CURRENT_RECORDS
        FINAL_SEQUENCE_MANIFEST = RUN_DIRECTORY / "metadata" / "final_recursive_sequence.json"
        FINAL_SEQUENCE_MANIFEST.write_text(json.dumps({
            "project": PROJECT_NAME,
            "cyclic": True,
            "anchor_count": len(BASE_RECORDS),
            "interpolation_rounds": INTERPOLATION_ROUNDS,
            "midpoints_per_gap": MIDPOINTS_PER_GAP,
            "final_count": len(FINAL_RECORDS),
            "round_manifests": ROUND_MANIFESTS,
            "records": FINAL_RECORDS,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print({"final_images": len(FINAL_RECORDS), "manifest": str(FINAL_SEQUENCE_MANIFEST)})
        release_flux_pipeline()
        """
    ),
    markdown(
        r"""
        ## 10. Assemble, preview, and audit the generated cyclic sequence

        No endpoint is duplicated. The measured wrap edge is part of the cycle, and optional rotation places the playback boundary at the quietest neighboring pair. This changes only where playback starts, not the circular order.
        """
    ),
    code(
        r"""
        import imageio_ffmpeg
        import numpy as np
        import shutil
        import subprocess
        import tempfile
        from IPython.display import Video

        # A Colab reconnect clears Python variables while completed manifests and
        # images remain on Drive. Recover the final sequence before auditing it.
        if "FINAL_RECORDS" not in globals():
            if "RUN_DIRECTORY" not in globals():
                raise RuntimeError(
                    "RUN_DIRECTORY is not initialized. Set RESUME_RUN_DIRECTORY to the "
                    "completed Drive run, then rerun the setup cells before section 10."
                )
            restored_manifest_path = Path(
                globals().get(
                    "FINAL_SEQUENCE_MANIFEST",
                    RUN_DIRECTORY / "metadata" / "final_recursive_sequence.json",
                )
            )
            if not restored_manifest_path.is_file():
                raise RuntimeError(
                    "FINAL_RECORDS is not in memory and the saved sequence manifest was "
                    f"not found at {restored_manifest_path}. Set RESUME_RUN_DIRECTORY to "
                    "the completed Drive run and rerun the setup cells."
                )
            restored_payload = json.loads(restored_manifest_path.read_text(encoding="utf-8"))
            FINAL_RECORDS = restored_payload["records"]
            FINAL_SEQUENCE_MANIFEST = restored_manifest_path
            print({
                "restored_final_sequence": True,
                "frames": len(FINAL_RECORDS),
                "manifest": str(FINAL_SEQUENCE_MANIFEST),
            })

        if len(FINAL_RECORDS) < 3:
            raise RuntimeError("A cyclic preview needs at least three images")
        if LOOP_PREVIEW_RENDER_MAX_SIDE < 128:
            raise ValueError("LOOP_PREVIEW_RENDER_MAX_SIDE must be at least 128")

        canonical_paths = [Path(item["path"]) for item in FINAL_RECORDS]

        def metric_array(path, size):
            with Image.open(path) as opened:
                sample = opened.convert("RGB")
                sample.thumbnail((size, size))
                return np.asarray(sample, dtype=np.uint8).copy()

        def mean_absolute_delta(left, right):
            difference = left.astype(np.int16) - right.astype(np.int16)
            return float(np.mean(np.abs(difference)) / 255.0)

        print(f"Reading {len(canonical_paths)} small seam-analysis thumbnails (not full frames)...")
        metric_frames = [metric_array(path, LOOP_SEAM_ANALYSIS_SIZE) for path in canonical_paths]
        edge_scores = [
            mean_absolute_delta(metric_frames[index], metric_frames[index - 1])
            for index in range(len(metric_frames))
        ]
        quietest_cut_index = int(np.argmin(edge_scores))
        export_cut_index = quietest_cut_index if LOOP_AUTO_ROTATE_TO_QUIETEST_CUT else 0
        EXPORT_FRAME_PATHS = canonical_paths[export_cut_index:] + canonical_paths[:export_cut_index]
        EXPORT_RECORDS = FINAL_RECORDS[export_cut_index:] + FINAL_RECORDS[:export_cut_index]
        export_metrics = metric_frames[export_cut_index:] + metric_frames[:export_cut_index]

        seam_delta = mean_absolute_delta(export_metrics[0], export_metrics[-1])
        median_delta = float(np.median(edge_scores))
        seam_ratio = seam_delta / median_delta if median_delta else 0.0
        incoming_motion = export_metrics[0].astype(np.int16) - export_metrics[-1].astype(np.int16)
        outgoing_motion = export_metrics[1].astype(np.int16) - export_metrics[0].astype(np.int16)
        motion_mismatch = float(np.mean(np.abs(outgoing_motion - incoming_motion)) / 255.0)

        preview_directory = RUN_DIRECTORY / "previews" / "generated_loop"
        preview_directory.mkdir(parents=True, exist_ok=True)
        preview_video_path = preview_directory / "generated_loop_reduced.mp4"
        preview_stage = Path(tempfile.mkdtemp(prefix="flowmorph_preview_"))
        try:
            for index, source_path in enumerate(EXPORT_FRAME_PATHS):
                staged_path = preview_stage / f"{index:07d}.png"
                try:
                    staged_path.symlink_to(source_path.resolve())
                except OSError:
                    shutil.copy2(source_path, staged_path)
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.check_call([
                ffmpeg, "-y", "-framerate", str(SOURCE_SEQUENCE_FPS),
                "-i", str(preview_stage / "%07d.png"),
                "-vf", (
                    f"scale={LOOP_PREVIEW_RENDER_MAX_SIDE}:{LOOP_PREVIEW_RENDER_MAX_SIDE}:"
                    "force_original_aspect_ratio=decrease:force_divisible_by=2"
                ),
                "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(preview_video_path),
            ])
        finally:
            shutil.rmtree(preview_stage, ignore_errors=True)

        seam_sheet_path = preview_directory / "seam_audit.png"
        seam_images = []
        try:
            for path in (EXPORT_FRAME_PATHS[-1], EXPORT_FRAME_PATHS[0], EXPORT_FRAME_PATHS[1]):
                with Image.open(path) as opened:
                    seam_images.append(opened.convert("RGB"))
            make_contact_sheet(
                seam_images,
                seam_sheet_path,
                columns=3,
                labels=["last before wrap", "playback start", "first after start"],
            )
        finally:
            for image in seam_images:
                image.close()
        seam_report = {
            "cyclic": True,
            "duplicate_terminal_frame": False,
            "frame_count": len(EXPORT_FRAME_PATHS),
            "auto_rotate": LOOP_AUTO_ROTATE_TO_QUIETEST_CUT,
            "cut_index": export_cut_index,
            "quietest_cut_index": quietest_cut_index,
            "seam_mean_absolute_delta": seam_delta,
            "median_edge_mean_absolute_delta": median_delta,
            "seam_ratio_to_median": seam_ratio,
            "seam_motion_mismatch": motion_mismatch,
            "ordered_uids": [item["uid"] for item in EXPORT_RECORDS],
        }
        seam_report_path = preview_directory / "seam_audit.json"
        seam_report_path.write_text(json.dumps(seam_report, indent=2) + "\n", encoding="utf-8")

        seam_preview = Image.open(seam_sheet_path).convert("RGB")
        seam_preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
        display(Markdown("### Loop seam: last → playback start → next"))
        display(seam_preview)
        del seam_preview
        display(Markdown("### Generated-image loop before RIFE"))
        display(Video(
            str(preview_video_path),
            embed=False,
            width=LOOP_PREVIEW_DISPLAY_WIDTH,
            html_attributes="controls loop muted playsinline",
        ))
        del metric_frames, export_metrics, incoming_motion, outgoing_motion
        print({
            "frames": len(EXPORT_FRAME_PATHS),
            "cut_index": export_cut_index,
            "seam_vs_median": round(seam_ratio, 4),
            "motion_mismatch": round(motion_mismatch, 6),
            "preview_video": str(preview_video_path),
        })
        """
    ),
    markdown(
        r"""
        ## 11. Prepare pinned Practical-RIFE and its v4.25 model

        RIFE operates on the lossless generated PNG sequence, including a final-to-first pair. Its large temporary dense-frame lattice stays on local Colab storage; only diagnostics and the final video are written to the persistent run directory.
        """
    ),
    code(
        r"""
        import subprocess
        import zipfile

        if not RUN_RIFE_POSTPROCESS:
            print("RIFE post-processing disabled in section 1.")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("RIFE requires CUDA")
            if RIFE_MULTIPLIER < 2:
                raise ValueError("RIFE_MULTIPLIER must be at least 2")
            if RIFE_SCALE not in {0.25, 0.5, 1.0, 2.0, 4.0}:
                raise ValueError("RIFE_SCALE must be 0.25, 0.5, 1.0, 2.0, or 4.0")
            if RIFE_FINAL_FPS <= 0 or SOURCE_SEQUENCE_FPS <= 0:
                raise ValueError("Source and final FPS must be positive")
            if RIFE_FINAL_FPS / SOURCE_SEQUENCE_FPS > RIFE_MULTIPLIER:
                raise ValueError("RIFE_MULTIPLIER must be at least RIFE_FINAL_FPS / SOURCE_SEQUENCE_FPS")

            rife_root = Path(RIFE_ROOT)
            if not (rife_root / ".git").is_dir():
                if rife_root.exists():
                    raise RuntimeError(f"RIFE_ROOT exists but is not a Git checkout: {rife_root}")
                subprocess.check_call(["git", "clone", "--filter=blob:none", RIFE_REPOSITORY_URL, str(rife_root)])
            installed_revision = subprocess.check_output(
                ["git", "-C", str(rife_root), "rev-parse", "HEAD"], text=True
            ).strip()
            if installed_revision != RIFE_REPOSITORY_REVISION:
                subprocess.check_call([
                    "git", "-C", str(rife_root), "fetch", "--depth", "1",
                    "origin", RIFE_REPOSITORY_REVISION,
                ])
                subprocess.check_call([
                    "git", "-C", str(rife_root), "checkout", "--detach", RIFE_REPOSITORY_REVISION
                ])
            installed_revision = subprocess.check_output(
                ["git", "-C", str(rife_root), "rev-parse", "HEAD"], text=True
            ).strip()
            if installed_revision != RIFE_REPOSITORY_REVISION:
                raise RuntimeError("Practical-RIFE checkout did not resolve to the pinned revision")

            rife_archive = Path(hf_hub_download(
                repo_id=RIFE_MODEL_REPOSITORY,
                filename=RIFE_MODEL_FILENAME,
                revision=RIFE_MODEL_REVISION,
                cache_dir=HF_CACHE_DIR,
            ))
            rife_model_root = Path(HF_CACHE_DIR) / "flowmorph_rife_models" / RIFE_MODEL_FILENAME.removesuffix(".zip")
            candidates = list(rife_model_root.rglob("flownet.pkl")) if rife_model_root.exists() else []
            if not candidates:
                rife_model_root.mkdir(parents=True, exist_ok=True)
                root_resolved = rife_model_root.resolve()
                with zipfile.ZipFile(rife_archive) as archive:
                    for member in archive.infolist():
                        destination = (rife_model_root / member.filename).resolve()
                        if not destination.is_relative_to(root_resolved):
                            raise RuntimeError(f"Unsafe path in RIFE archive: {member.filename}")
                    archive.extractall(rife_model_root)
                candidates = list(rife_model_root.rglob("flownet.pkl"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one RIFE flownet.pkl, found {candidates}")
            RIFE_MODEL_DIRECTORY = candidates[0].parent

            RIFE_RUNNER_SOURCE = r'''
        import argparse
        import shutil
        import sys
        from pathlib import Path

        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        parser = argparse.ArgumentParser()
        parser.add_argument("--repo", required=True)
        parser.add_argument("--model", required=True)
        parser.add_argument("--input", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--multi", type=int, required=True)
        parser.add_argument("--scale", type=float, required=True)
        parser.add_argument("--fp16", action="store_true")
        args = parser.parse_args()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for RIFE")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        sys.path.insert(0, str(Path(args.model).resolve().parent))
        sys.path.insert(0, str(Path(args.repo).resolve()))
        import train_log.IFNet_HDv3 as rife_ifnet_module
        from train_log.IFNet_HDv3 import IFNet

        # Practical-RIFE's pinned warplayer builds its cached sampling grid in
        # float32. torch.grid_sample requires input and grid to share a dtype,
        # so that implementation fails when the network runs in fp16. Replace
        # the function in IFNet's module namespace with a device/dtype-aware
        # equivalent while retaining the pinned model and weights.
        rife_grid_cache = {}

        def dtype_safe_warp(tensor_input, tensor_flow):
            cache_key = (
                str(tensor_flow.device),
                str(tensor_flow.dtype),
                tuple(tensor_flow.shape),
            )
            if cache_key not in rife_grid_cache:
                horizontal = torch.linspace(
                    -1.0, 1.0, tensor_flow.shape[3],
                    device=tensor_flow.device, dtype=tensor_flow.dtype,
                ).view(1, 1, 1, tensor_flow.shape[3]).expand(
                    tensor_flow.shape[0], -1, tensor_flow.shape[2], -1
                )
                vertical = torch.linspace(
                    -1.0, 1.0, tensor_flow.shape[2],
                    device=tensor_flow.device, dtype=tensor_flow.dtype,
                ).view(1, 1, tensor_flow.shape[2], 1).expand(
                    tensor_flow.shape[0], -1, -1, tensor_flow.shape[3]
                )
                rife_grid_cache[cache_key] = torch.cat((horizontal, vertical), dim=1)
            normalized_flow = torch.cat((
                tensor_flow[:, 0:1] / ((tensor_input.shape[3] - 1.0) / 2.0),
                tensor_flow[:, 1:2] / ((tensor_input.shape[2] - 1.0) / 2.0),
            ), dim=1)
            grid = (rife_grid_cache[cache_key] + normalized_flow).permute(0, 2, 3, 1)
            return F.grid_sample(
                tensor_input,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

        rife_ifnet_module.warp = dtype_safe_warp

        input_paths = sorted(Path(args.input).glob("*.png"), key=lambda path: int(path.stem))
        if len(input_paths) < 2:
            raise ValueError("RIFE input needs at least two numbered PNG files")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)

        class InferenceModel:
            def __init__(self, model_directory):
                # Load on CPU first, then move the complete initialized module to
                # one explicit CUDA device. This avoids mixed CPU/CUDA parameters
                # with newer torch/checkpoint combinations.
                self.flownet = IFNet()
                state = torch.load(
                    str(Path(model_directory) / "flownet.pkl"), map_location="cpu", weights_only=True
                )
                state = {key.removeprefix("module."): value for key, value in state.items()}
                load_result = self.flownet.load_state_dict(state, strict=False)
                if load_result.missing_keys:
                    raise RuntimeError(f"RIFE checkpoint missing keys: {load_result.missing_keys}")
                self.flownet.to(device).eval()

            def inference(self, image0, image1, timestep, scale):
                inputs = torch.cat((image0, image1), dim=1)
                scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
                _, _, merged = self.flownet(inputs, timestep, scale_list)
                return merged[-1]

        model = InferenceModel(args.model)
        if args.fp16:
            model.flownet.half()
        model_dtype = next(model.flownet.parameters()).dtype
        model_device = next(model.flownet.parameters()).device
        if model_device != device:
            raise RuntimeError(f"RIFE model resolved to {model_device}, expected {device}")
        print({
            "cuda_device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "model_device": str(model_device),
            "model_dtype": str(model_dtype),
            "input_pairs": len(input_paths) - 1,
        }, flush=True)
        first_image = Image.open(input_paths[0]).convert("RGB")
        height, width = first_image.height, first_image.width
        block = max(128, int(128 / args.scale))
        padded_height = ((height - 1) // block + 1) * block
        padded_width = ((width - 1) // block + 1) * block
        padding = (0, padded_width - width, 0, padded_height - height)

        def load_tensor(path):
            image = Image.open(path).convert("RGB")
            if image.size != (width, height):
                raise ValueError(f"Mismatched input dimensions at {path}: {image.size}")
            array = np.asarray(image, dtype=np.uint8).copy()
            tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=model_dtype) / 255.0
            return F.pad(tensor, padding)

        def save_tensor(tensor, path):
            array = (tensor[0, :, :height, :width].float().clamp(0, 1) * 255.0).round().byte()
            array = array.permute(1, 2, 0).cpu().numpy()
            Image.fromarray(array, mode="RGB").save(path, compress_level=4)

        output_index = 0
        shutil.copy2(input_paths[0], output / f"{output_index:07d}.png")
        output_index += 1
        report_every = max(1, (len(input_paths) - 1) // 20)
        with torch.inference_mode():
            left = load_tensor(input_paths[0])
            for pair_index, right_path in enumerate(input_paths[1:], start=1):
                right = load_tensor(right_path)
                for step in range(1, args.multi):
                    middle = model.inference(left, right, timestep=step / args.multi, scale=args.scale)
                    save_tensor(middle, output / f"{output_index:07d}.png")
                    output_index += 1
                shutil.copy2(right_path, output / f"{output_index:07d}.png")
                output_index += 1
                left = right
                if pair_index % report_every == 0 or pair_index == len(input_paths) - 1:
                    print(f"RIFE pairs: {pair_index}/{len(input_paths) - 1}; frames: {output_index}", flush=True)
        print(f"RIFE complete: {output_index} PNG frames")
        '''
            RIFE_RUNNER_PATH = Path(LOCAL_ASSET_ROOT) / PROJECT_NAME / "rife_pair_sequence_runner.py"
            RIFE_RUNNER_PATH.parent.mkdir(parents=True, exist_ok=True)
            RIFE_RUNNER_PATH.write_text(RIFE_RUNNER_SOURCE.strip() + "\n", encoding="utf-8")
            print({
                "rife_revision": installed_revision,
                "model_directory": str(RIFE_MODEL_DIRECTORY),
                "multiplier": RIFE_MULTIPLIER,
                "scale": RIFE_SCALE,
                "runner": str(RIFE_RUNNER_PATH),
            })
        """
    ),
    markdown(
        r"""
        ## 12. Interpolate the circular PNG sequence with RIFE

        The opening image is appended once as a temporary terminal input so RIFE explicitly processes the wraparound pair. The exact duplicate is verified and removed before resampling and export.
        """
    ),
    code(
        r"""
        if RUN_RIFE_POSTPROCESS:
            # A Colab reconnect clears Python variables even though completed images and
            # manifests remain on Drive. Restore the ordered export paths when needed.
            if "EXPORT_FRAME_PATHS" not in globals():
                restored_records = globals().get("FINAL_RECORDS")
                if not restored_records:
                    restored_manifest_path = Path(
                        globals().get(
                            "FINAL_SEQUENCE_MANIFEST",
                            RUN_DIRECTORY / "metadata" / "final_recursive_sequence.json",
                        )
                    )
                    if not restored_manifest_path.is_file():
                        raise RuntimeError(
                            "The generated sequence is not available in memory and its saved "
                            f"manifest was not found at {restored_manifest_path}. Set "
                            "RESUME_RUN_DIRECTORY to the completed Drive run and rerun the "
                            "setup/assembly cells before RIFE."
                        )
                    restored_payload = json.loads(restored_manifest_path.read_text(encoding="utf-8"))
                    restored_records = restored_payload["records"]

                restored_cut_index = 0
                restored_seam_report_path = (
                    RUN_DIRECTORY / "previews" / "generated_loop" / "seam_audit.json"
                )
                if restored_seam_report_path.is_file():
                    restored_seam_report = json.loads(
                        restored_seam_report_path.read_text(encoding="utf-8")
                    )
                    restored_cut_index = int(restored_seam_report.get("cut_index", 0))
                elif LOOP_AUTO_ROTATE_TO_QUIETEST_CUT:
                    print(
                        "No saved seam report was found; using canonical circular order. "
                        "Run section 10 first if you want automatic quietest-cut rotation."
                    )

                restored_cut_index %= len(restored_records)
                EXPORT_RECORDS = (
                    restored_records[restored_cut_index:] + restored_records[:restored_cut_index]
                )
                EXPORT_FRAME_PATHS = [Path(item["path"]) for item in EXPORT_RECORDS]
                missing_export_paths = [path for path in EXPORT_FRAME_PATHS if not path.is_file()]
                if missing_export_paths:
                    raise FileNotFoundError(
                        "The restored sequence references missing image files; first missing path: "
                        f"{missing_export_paths[0]}"
                    )
                print({
                    "restored_export_sequence": True,
                    "frames": len(EXPORT_FRAME_PATHS),
                    "cut_index": restored_cut_index,
                })

            postprocess_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            RIFE_WORK_DIRECTORY = Path(LOCAL_ASSET_ROOT) / PROJECT_NAME / "rife_work" / postprocess_stamp
            RIFE_INPUT_DIRECTORY = RIFE_WORK_DIRECTORY / "cyclic_input"
            RIFE_DENSE_DIRECTORY = RIFE_WORK_DIRECTORY / "dense_frames"
            RIFE_RESULTS_DIRECTORY = RUN_DIRECTORY / "video" / postprocess_stamp
            RIFE_INPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
            RIFE_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=False)

            for index, source_path in enumerate(EXPORT_FRAME_PATHS):
                shutil.copy2(source_path, RIFE_INPUT_DIRECTORY / f"{index:07d}.png")
            shutil.copy2(
                RIFE_INPUT_DIRECTORY / "0000000.png",
                RIFE_INPUT_DIRECTORY / f"{len(EXPORT_FRAME_PATHS):07d}.png",
            )
            command = [
                sys.executable, "-u", str(RIFE_RUNNER_PATH),
                "--repo", str(rife_root),
                "--model", str(RIFE_MODEL_DIRECTORY),
                "--input", str(RIFE_INPUT_DIRECTORY),
                "--output", str(RIFE_DENSE_DIRECTORY),
                "--multi", str(RIFE_MULTIPLIER),
                "--scale", str(RIFE_SCALE),
            ]
            if RIFE_USE_FP16:
                command.append("--fp16")
            print(f"Interpolating {len(EXPORT_FRAME_PATHS)} cyclic pairs at {RIFE_MULTIPLIER}× density...")

            def run_rife(command_to_run, label):
                print(f"RIFE attempt: {label}", flush=True)
                process = subprocess.Popen(
                    command_to_run,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                log_lines = []
                assert process.stdout is not None
                for line in process.stdout:
                    log_lines.append(line)
                    print(line, end="", flush=True)
                return_code = process.wait()
                log_text = "".join(log_lines) or "(RIFE produced no subprocess output)"
                if not log_lines:
                    print(log_text)
                return return_code, log_text

            return_code, rife_log = run_rife(
                command,
                "fp16" if "--fp16" in command else "fp32",
            )
            if return_code != 0 and "--fp16" in command and RIFE_RETRY_WITH_FP32:
                print("The fp16 RIFE attempt failed; retrying once in fp32.")
                RIFE_DENSE_DIRECTORY = RIFE_WORK_DIRECTORY / "dense_frames_fp32_retry"
                retry_command = [item for item in command if item != "--fp16"]
                output_index = retry_command.index("--output") + 1
                retry_command[output_index] = str(RIFE_DENSE_DIRECTORY)
                return_code, rife_log = run_rife(retry_command, "fp32 fallback")
            if return_code != 0:
                raise RuntimeError(
                    "RIFE failed after the available attempt(s). The complete child-process "
                    "traceback is printed immediately above. Last output:\n" + rife_log[-6000:]
                )

            dense_with_duplicate = sorted(
                RIFE_DENSE_DIRECTORY.glob("*.png"), key=lambda path: int(path.stem)
            )
            expected = len(EXPORT_FRAME_PATHS) * RIFE_MULTIPLIER + 1
            if len(dense_with_duplicate) != expected:
                raise RuntimeError(f"RIFE wrote {len(dense_with_duplicate)} frames; expected {expected}")
            with Image.open(dense_with_duplicate[0]) as opened:
                first_array = np.asarray(opened.convert("RGB"))
            with Image.open(dense_with_duplicate[-1]) as opened:
                last_array = np.asarray(opened.convert("RGB"))
            if not np.array_equal(first_array, last_array):
                raise RuntimeError("RIFE terminal image is not pixel-identical to the opening image")
            RIFE_DENSE_PATHS = dense_with_duplicate[:-1]
            print({
                "base_cyclic_frames": len(EXPORT_FRAME_PATHS),
                "dense_unique_frames": len(RIFE_DENSE_PATHS),
                "removed_exact_terminal_duplicate": True,
                "local_work_directory": str(RIFE_WORK_DIRECTORY),
                "persistent_results_directory": str(RIFE_RESULTS_DIRECTORY),
            })
        """
    ),
    markdown(
        r"""
        ## 13. Circular SSIM motion equalization and final H.264 loop

        Dense RIFE frames are weighted by circular `1 − SSIM`. Monotonic unique selection places more final frames where visual motion is larger while preserving duration and the cyclic wrap. The MP4, report, and motion plot are written directly to Drive; large temporary PNG lattices are deleted only after successful export when `RIFE_KEEP_WORK_FRAMES=False`.
        """
    ),
    code(
        r"""
        if RUN_RIFE_POSTPROCESS:
            import imageio_ffmpeg
            import matplotlib.pyplot as plt
            from IPython.display import Video
            from skimage.metrics import structural_similarity

            def ssim_luma(path):
                with Image.open(path) as opened:
                    gray = opened.convert("L")
                    gray.thumbnail((RIFE_SSIM_ANALYSIS_SIZE, RIFE_SSIM_ANALYSIS_SIZE))
                    return np.asarray(gray, dtype=np.uint8)

            print(f"Computing circular SSIM weights for {len(RIFE_DENSE_PATHS)} dense frames...")
            dense_luma = [ssim_luma(path) for path in RIFE_DENSE_PATHS]
            circular_ssim = np.asarray([
                structural_similarity(dense_luma[index - 1], dense_luma[index], data_range=255)
                for index in range(len(dense_luma))
            ], dtype=np.float64)
            motion_weights = np.maximum(RIFE_SSIM_WEIGHT_FLOOR, 1.0 - circular_ssim)
            frame_positions = np.zeros(len(RIFE_DENSE_PATHS), dtype=np.float64)
            frame_positions[1:] = np.cumsum(motion_weights[1:])
            total_motion = float(frame_positions[-1] + motion_weights[0])

            canonical_duration = len(EXPORT_FRAME_PATHS) / float(SOURCE_SEQUENCE_FPS)
            target_frame_count = int(round(canonical_duration * RIFE_FINAL_FPS))
            if target_frame_count < 3:
                raise ValueError("Final video needs at least three frames")
            if target_frame_count > len(RIFE_DENSE_PATHS):
                raise ValueError("Increase RIFE_MULTIPLIER or lower RIFE_FINAL_FPS")
            targets = np.linspace(0.0, total_motion, target_frame_count, endpoint=False)

            selected_indices = []
            previous_index = -1
            dense_count = len(RIFE_DENSE_PATHS)
            for order, target in enumerate(targets):
                minimum = previous_index + 1
                maximum = dense_count - (target_frame_count - order)
                insertion = int(np.searchsorted(frame_positions, target, side="left"))
                candidates = {
                    min(max(insertion, minimum), maximum),
                    min(max(insertion - 1, minimum), maximum),
                }
                chosen = min(candidates, key=lambda index: abs(frame_positions[index] - target))
                selected_indices.append(chosen)
                previous_index = chosen
            selected_indices[0] = 0
            selected_indices[-1] = dense_count - 1
            if len(set(selected_indices)) != target_frame_count:
                raise RuntimeError("SSIM resampling produced duplicate selections")

            selected_directory = RIFE_WORK_DIRECTORY / "ssim_resampled_frames"
            selected_directory.mkdir(parents=True, exist_ok=True)
            # This is a generated, run-local staging directory. Clear numbered
            # frames so rerunning this cell cannot leave stale PNGs behind when
            # timing settings or the selected frame count change.
            for stale_frame in selected_directory.glob("*.png"):
                stale_frame.unlink()
            for output_index, dense_index in enumerate(selected_indices):
                source = RIFE_DENSE_PATHS[dense_index]
                destination = selected_directory / f"{output_index:07d}.png"
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)

            RIFE_FINAL_VIDEO_PATH = RIFE_RESULTS_DIRECTORY / "recursive_vision_rife_ssim_loop.mp4"
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.check_call([
                ffmpeg, "-y", "-framerate", str(RIFE_FINAL_FPS),
                "-i", str(selected_directory / "%07d.png"),
                "-an", "-c:v", "libx264", "-preset", "slow",
                "-crf", str(RIFE_VIDEO_CRF), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(RIFE_FINAL_VIDEO_PATH),
            ])
            if not RIFE_FINAL_VIDEO_PATH.is_file() or RIFE_FINAL_VIDEO_PATH.stat().st_size == 0:
                raise RuntimeError("FFmpeg did not create the final video")

            selected_ssim = np.asarray([
                structural_similarity(
                    dense_luma[selected_indices[index - 1]],
                    dense_luma[selected_indices[index]],
                    data_range=255,
                )
                for index in range(target_frame_count)
            ], dtype=np.float64)
            selected_motion = 1.0 - selected_ssim
            report = {
                "method": "cyclic PNGs -> Practical-RIFE -> circular 1-SSIM equal-motion sampling -> H.264",
                "rife_repository": RIFE_REPOSITORY_URL,
                "rife_revision": RIFE_REPOSITORY_REVISION,
                "rife_model_repository": RIFE_MODEL_REPOSITORY,
                "rife_model_revision": RIFE_MODEL_REVISION,
                "rife_model_filename": RIFE_MODEL_FILENAME,
                "rife_multiplier": RIFE_MULTIPLIER,
                "rife_scale": RIFE_SCALE,
                "rife_fp16": RIFE_USE_FP16,
                "base_frames": len(EXPORT_FRAME_PATHS),
                "dense_unique_frames": len(RIFE_DENSE_PATHS),
                "final_unique_frames": target_frame_count,
                "source_fps": SOURCE_SEQUENCE_FPS,
                "final_fps": RIFE_FINAL_FPS,
                "duration_seconds": target_frame_count / RIFE_FINAL_FPS,
                "dense_ssim": {
                    "mean": float(circular_ssim.mean()),
                    "median": float(np.median(circular_ssim)),
                    "minimum": float(circular_ssim.min()),
                    "wraparound": float(circular_ssim[0]),
                },
                "resampled_ssim": {
                    "mean": float(selected_ssim.mean()),
                    "median": float(np.median(selected_ssim)),
                    "minimum": float(selected_ssim.min()),
                    "wraparound": float(selected_ssim[0]),
                    "motion_coefficient_of_variation": (
                        float(selected_motion.std() / selected_motion.mean())
                        if selected_motion.mean() else 0.0
                    ),
                },
                "selected_dense_indices": selected_indices,
                "terminal_duplicate_in_video": False,
                "video": str(RIFE_FINAL_VIDEO_PATH),
            }
            RIFE_REPORT_PATH = RIFE_RESULTS_DIRECTORY / "rife_ssim_report.json"
            RIFE_REPORT_PATH.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            RIFE_SSIM_PLOT_PATH = RIFE_RESULTS_DIRECTORY / "ssim_motion_profile.png"
            figure, axes = plt.subplots(2, 1, figsize=(12, 5), constrained_layout=True)
            axes[0].plot(motion_weights, linewidth=0.7, color="#4b6a88")
            axes[0].scatter([0], [motion_weights[0]], color="#c43d32", s=24, label="wrap edge")
            axes[0].set_title("Dense RIFE motion profile (1 − circular SSIM)")
            axes[0].legend(loc="upper right")
            axes[1].plot(selected_motion, linewidth=0.8, color="#7a5535")
            axes[1].scatter([0], [selected_motion[0]], color="#c43d32", s=24, label="wrap edge")
            axes[1].set_title("After equal-motion resampling")
            axes[1].set_xlabel("Frame edge")
            axes[1].legend(loc="upper right")
            for axis in axes:
                axis.set_ylabel("1 − SSIM")
                axis.grid(alpha=0.2)
            figure.savefig(RIFE_SSIM_PLOT_PATH, dpi=160, facecolor="white")
            plt.close(figure)

            plot_preview = Image.open(RIFE_SSIM_PLOT_PATH).convert("RGB")
            plot_preview.thumbnail((CONTACT_SHEET_DISPLAY_MAX_WIDTH, 100000))
            display(Markdown("### Final RIFE + circular SSIM diagnostics"))
            display(plot_preview)
            del plot_preview
            display(Video(
                str(RIFE_FINAL_VIDEO_PATH),
                embed=False,
                width=RIFE_DISPLAY_WIDTH,
                html_attributes="controls loop muted playsinline",
            ))
            print({
                "final_video": str(RIFE_FINAL_VIDEO_PATH),
                "frames": target_frame_count,
                "fps": RIFE_FINAL_FPS,
                "seconds": round(target_frame_count / RIFE_FINAL_FPS, 3),
                "wraparound_ssim": round(float(selected_ssim[0]), 6),
                "motion_variation": round(report["resampled_ssim"]["motion_coefficient_of_variation"], 6),
            })

            if DOWNLOAD_FINAL_VIDEO:
                try:
                    from google.colab import files
                except ImportError:
                    print("Colab download helper unavailable; use the printed video path.")
                else:
                    files.download(str(RIFE_FINAL_VIDEO_PATH))

            if not RIFE_KEEP_WORK_FRAMES:
                expected_parent = (Path(LOCAL_ASSET_ROOT) / PROJECT_NAME / "rife_work").resolve()
                target = RIFE_WORK_DIRECTORY.resolve()
                if target.parent != expected_parent:
                    raise RuntimeError(f"Refusing to remove unexpected RIFE directory: {target}")
                shutil.rmtree(target)
                print("Removed temporary local RIFE PNGs after successful persistent export.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} clean cells")
