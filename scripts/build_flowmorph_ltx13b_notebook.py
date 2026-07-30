"""Build the one-round FlowMorph -> LTX-Video 13B production notebook.

The current prompt-only notebook is the source of truth for editable anchor
prompts and the proven FLUX/FlowMorph setup.  This builder copies source cells
only, changes the recursion to one twelve-interior round, and replaces the
RIFE finishing stack with resumable LTX 0.9.8 13B multi-frame conditioning.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Prompt_Only.ipynb"
OUTPUT = ROOT / "notebooks" / "StillLife_FlowMorph_LTX13B_Conditioned_Video.ipynb"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one replacement target: {old!r}")
    return text.replace(old, new, 1)


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


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source_notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
if len(source_notebook["cells"]) < 22:
    raise RuntimeError("Prompt-only source notebook is unexpectedly short")

# Keep the prompt-only workflow through the reusable sequence setup, then use
# its full-run cell with one round. All copied outputs are deliberately dropped.
notebook = copy.deepcopy(source_notebook)
notebook["cells"] = copy.deepcopy(source_notebook["cells"][:20])
full_round_cell = copy.deepcopy(source_notebook["cells"][21])

notebook["cells"][0]["source"] = lines(
    """
    # Prompt-only FlowMorph → LTX-Video 13B conditioned loop

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MNoichl/FluxFlowMorph/blob/main/notebooks/StillLife_FlowMorph_LTX13B_Conditioned_Video.ipynb)

    This notebook keeps the proven prompt-only FLUX.2/RIJKSOIL setup, but changes
    the finishing strategy:

    1. Generate the editable cyclic anchor paintings.
    2. Fit every unique anchor once and run **one** true FlowMorph round with
       twelve interior frames per circular gap.
    3. Reuse the fitted endpoint reconstructions at both sides of every gap.
       Each LTX job therefore receives 14 closely related still conditions:
       endpoint + 12 FlowMorph interiors + endpoint.
    4. Explicitly release the FLUX/FlowMorph 9B stack and verify CUDA memory.
    5. Load LTX-Video 0.9.8 13B distilled once with FP8 layerwise storage and
       group CPU offload, render every gap, and concatenate the resumable clips
       into one circular H.264 video.

    The conditions are placed eight video frames apart, at frame indices
    0, 8, …, 104. Thus every gap contains 105 frames (`13 × 8 + 1`), satisfying
    LTX's temporal contract while making the generated motion follow the actual
    FlowMorph trajectory rather than merely its endpoints.
    """
)
notebook["cells"][1]["source"] = lines(
    """
    ## 1. Editable prompt-only, one-round FlowMorph, and LTX-13B settings

    The anchor list remains the creative source of truth. FLUX settings control
    the paintings and fitted FlowMorph conditions; the separate LTX block
    controls video duration, conditioning strength, quality, memory behavior,
    and output size.
    """
)

settings = source(notebook["cells"][2])
settings = replace_once(
    settings,
    'PROJECT_NAME = "science_path_prompt_only_flowmorph"',
    'PROJECT_NAME = "science_path_flowmorph_ltx13b"',
)
settings = replace_once(
    settings,
    'FLOWMORPH_ROUND_SPECS = [\n'
    '    {"midpoint_count": 1, "prompt_mode": "explicit_midpoint"},\n'
    '    {"midpoint_count": 10, "prompt_mode": "shared_midpoint"},\n'
    "]",
    'FLOWMORPH_ROUND_SPECS = [\n'
    '    {"midpoint_count": 12, "prompt_mode": "shared_midpoint"},\n'
    "]",
)
tone_start = settings.index("# Optional post-FlowMorph tonal correction")
trial_start = settings.index("# Trial and notebook display.")
settings = settings[:tone_start] + settings[trial_start:]
finishing_start = settings.index("# Cyclic sequence and RIFE/SSIM finishing.")
settings = settings[:finishing_start] + dedent(
    """
    # The sequence session only uses this integer for its internal config.
    SOURCE_SEQUENCE_FPS = 12.0

    # LTX-Video 0.9.8 13B distilled multi-condition rendering.
    LTX_MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
    LTX_MODEL_REVISION = "7c64400e1861cc0d7b98d570a1926d5408ec60cd"
    LTX_UPSAMPLER_ID = "a-r-r-o-w/LTX-0.9.8-Latent-Upsampler"
    LTX_UPSAMPLER_REVISION = "e0c981533db26531c47dec16a124586cea53f11f"
    LTX_USE_TWO_STAGE_UPSCALING = True
    LTX_FINAL_WIDTH = 512
    LTX_FINAL_HEIGHT = 512
    LTX_FPS = 30
    LTX_FRAMES_PER_CONDITION_INTERVAL = 8
    LTX_CONDITIONING_STRENGTH = 1.0
    LTX_IMAGE_COND_NOISE_SCALE = 0.0
    LTX_DECODE_TIMESTEP = 0.05
    LTX_DECODE_NOISE_SCALE = 0.025
    LTX_GUIDANCE_SCALE = 1.0
    LTX_GUIDANCE_RESCALE = 0.7
    LTX_FIRST_PASS_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
    LTX_SECOND_PASS_TIMESTEPS = [1000, 909, 725, 421, 0]
    LTX_UPSCALE_DENOISE_STRENGTH = 0.999
    LTX_ADAIN_FACTOR = 1.0
    LTX_TONE_MAP_COMPRESSION_RATIO = 0.6
    LTX_MAX_PROMPT_WORDS = 200
    LTX_NEGATIVE_PROMPT = (
        "cuts, scene changes, camera shake, rapid camera movement, text, symbols, "
        "watermarks, worst quality, inconsistent motion, blurry, jittery, distorted"
    )
    LTX_MOTION_PROMPT_TEMPLATE = (
        "A locked camera observes a museum-quality Baroque oil still life. "
        "Across one continuous shot, the arrangement about {source_science} "
        "slowly and physically transforms into the arrangement about "
        "{target_science}. The interdisciplinary transition concerns "
        "{science_connection}. Nearby forms bend, unfold, exchange materials, "
        "and reorganize continuously; illumination, painted texture, scale, "
        "camera position, and the surrounding room remain coherent. "
        "Slow deliberate motion, no cuts, no new scene, no camera movement."
    )

    # Colab/L4-oriented memory controls. Group offload keeps only small model
    # groups on CUDA; stream prefetch is faster but consumes more pinned RAM.
    LTX_ENABLE_FP8_LAYERWISE_STORAGE = True
    LTX_ENABLE_GROUP_OFFLOAD = True
    LTX_GROUP_OFFLOAD_USE_STREAM = False
    LTX_GROUP_OFFLOAD_LOW_CPU_MEMORY = True
    LTX_MAX_RESERVED_GIB_BEFORE_LOAD = 1.0
    LTX_CACHE_DIR = HF_CACHE_DIR
    DELETE_LOCAL_FLUX_CACHE_BEFORE_LTX = True
    CLEAN_INTERRUPTED_LTX_DOWNLOADS_IF_NEEDED = True
    LTX_DISABLE_XET_FOR_DISK_SAFETY = True
    LTX_DOWNLOAD_HEADROOM_GIB = 5.0

    # Resumable output and notebook display.
    REUSE_EXISTING_LTX_CLIPS = True
    LTX_DISPLAY_EACH_CLIP = True
    LTX_DISPLAY_WIDTH = 640
    LTX_VIDEO_CRF = 16
    DOWNLOAD_LTX_FINAL_VIDEO = False
    """
).lstrip()
notebook["cells"][2]["source"] = settings.splitlines(keepends=True)

validation = source(notebook["cells"][10])
validation = replace_once(
    validation,
    'if len(FLOWMORPH_ROUND_SPECS) != 2:\n'
    '    raise ValueError("This notebook expects exactly two FlowMorph rounds")\n',
    'if len(FLOWMORPH_ROUND_SPECS) != 1:\n'
    '    raise ValueError("This notebook requires exactly one FlowMorph round")\n',
)
validation = replace_once(
    validation,
    'if FLOWMORPH_ROUND_SPECS[0] != {"midpoint_count": 1, "prompt_mode": "explicit_midpoint"}:\n'
    '    raise ValueError("Round 1 must use one explicitly prompted midpoint")\n'
    'if FLOWMORPH_ROUND_SPECS[1] != {"midpoint_count": 10, "prompt_mode": "shared_midpoint"}:\n'
    '    raise ValueError("Round 2 must use ten renders sharing one interpolation prompt")\n',
    'if FLOWMORPH_ROUND_SPECS[0] != {"midpoint_count": 12, "prompt_mode": "shared_midpoint"}:\n'
    '    raise ValueError("The sole round must render 12 interiors from one shared midpoint prompt")\n',
)
temporal_start = validation.index("if TEMPORAL_TONE_WINDOW_RADIUS < 1:")
openai_start = validation.index(
    'if OPENAI_IMAGE_DETAIL not in {"low", "high", "original", "auto"}:'
)
validation = validation[:temporal_start] + validation[openai_start:]
validation = replace_once(
    validation,
    'print({\n'
    '    "anchor_images": BASE_PROMPT_COUNT,',
    'if not 256 <= LTX_FINAL_WIDTH <= 1280 or not 256 <= LTX_FINAL_HEIGHT <= 720:\n'
    '    raise ValueError("LTX output must stay within the recommended sub-720p envelope")\n'
    'if LTX_FINAL_WIDTH % 8 or LTX_FINAL_HEIGHT % 8:\n'
    '    raise ValueError("LTX final width and height must be divisible by 8")\n'
    'if LTX_FRAMES_PER_CONDITION_INTERVAL % 8:\n'
    '    raise ValueError("LTX condition intervals must be multiples of 8 frames")\n'
    'if not 0 < LTX_CONDITIONING_STRENGTH <= 1:\n'
    '    raise ValueError("LTX conditioning strength must lie in (0, 1]")\n'
    'if LTX_GUIDANCE_SCALE != 1.0:\n'
    '    raise ValueError("The distilled LTX 13B model requires guidance 1.0")\n'
    'if not 0 <= LTX_TONE_MAP_COMPRESSION_RATIO <= 1:\n'
    '    raise ValueError("LTX tone-map compression must lie in [0, 1]")\n'
    'print({\n'
    '    "anchor_images": BASE_PROMPT_COUNT,',
)
validation = validation.replace(
    '    "rife_multiplier": RIFE_MULTIPLIER,\n',
    '    "ltx_clips": BASE_PROMPT_COUNT,\n'
    '    "ltx_conditions_per_clip": 14,\n'
    '    "ltx_frames_per_clip": 13 * LTX_FRAMES_PER_CONDITION_INTERVAL + 1,\n',
    1,
)
notebook["cells"][10]["source"] = validation.splitlines(keepends=True)

notebook["cells"][18]["source"] = lines(
    """
    ## 9. Sequence-native FlowMorph: one cached fit per anchor, one 12-frame gap pass

    The retained FLUX model is loaded once and the backward preflight runs once.
    Each unique anchor endpoint is fitted once, even though it belongs to two
    neighboring circular gaps. One image-aware shared midpoint prompt guides all
    twelve interior alphas in a gap, while the actual source and target prompt
    embeddings remain active on their respective halves.
    """
)

round_source = source(full_round_cell)
release_marker = (
    "# Free the retained 9B model before the RIFE subprocess claims the GPU."
)
release_index = round_source.index(release_marker)
round_source = round_source[:release_index].rstrip() + "\n\n"
round_source = round_source.replace(
    "final_recursive_flowmorph_sequence.json",
    "one_round_flowmorph_conditioning_sequence.json",
)
round_source += dedent(
    """
    # Freeze the exact per-gap conditioning series before releasing FlowMorph.
    # Every series has: canonical left endpoint, 12 interiors, canonical right.
    LTX_GAP_RECORDS = []
    for job in pair_jobs:
        condition_paths = [
            str(ENDPOINT_RECONSTRUCTION_PATHS[job["left"]["uid"]]),
            *[
                str(frame_record["output_path"])
                for frame_record in job["frame_records"]
            ],
            str(ENDPOINT_RECONSTRUCTION_PATHS[job["right"]["uid"]]),
        ]
        if len(condition_paths) != 14:
            raise RuntimeError(
                f"{job['pair_uid']} has {len(condition_paths)} conditions; expected 14"
            )
        missing = [path for path in condition_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{job['pair_uid']} has missing conditioning images: {missing}"
            )
        LTX_GAP_RECORDS.append({
            "pair_uid": job["pair_uid"],
            "left_uid": job["left"]["uid"],
            "right_uid": job["right"]["uid"],
            "source_science": job["left"]["science"],
            "target_science": job["right"]["science"],
            "science_connection": job["proposal"].science_connection,
            "shared_midpoint_prompt": job["proposal"].prompt,
            "condition_paths": condition_paths,
        })

    LTX_CONDITIONING_MANIFEST = (
        RUN_DIRECTORY / "metadata" / "ltx_conditioning_gaps.json"
    )
    LTX_CONDITIONING_MANIFEST.write_text(json.dumps({
        "cyclic": True,
        "gap_count": len(LTX_GAP_RECORDS),
        "flowmorph_interiors_per_gap": 12,
        "conditions_per_gap": 14,
        "records": LTX_GAP_RECORDS,
    }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
    print({
        "ltx_conditioning_manifest": str(LTX_CONDITIONING_MANIFEST),
        "cyclic_gaps": len(LTX_GAP_RECORDS),
        "conditions_per_gap": 14,
    })
    """
)
full_round_cell["source"] = round_source.splitlines(keepends=True)
notebook["cells"].extend(
    [
        markdown(
            """
            ## 10. Run the single circular FlowMorph round

            This is the expensive FLUX stage. It saves each gap as soon as it is
            decoded and finally writes an explicit 14-image conditioning list for
            every LTX clip. No second FlowMorph round and no RIFE pass are run.
            """
        ),
        full_round_cell,
        markdown(
            """
            ## 11. Explicitly release FLUX/FlowMorph before loading LTX

            LTX is a separate 13B video stack. This cell moves every known FLUX
            component to CPU, removes references and hooks, runs garbage
            collection, empties PyTorch's CUDA allocator, and reports remaining
            memory. A high residual is reported clearly before the 13B load.
            """
        ),
        code(
            """
            import gc
            import shutil
            import torch

            sequence_runner_for_release = globals().get("SEQUENCE_RUNNER")
            flowmorph_pipeline = getattr(
                sequence_runner_for_release,
                "pipeline",
                None,
            )
            if flowmorph_pipeline is not None:
                maybe_free = getattr(flowmorph_pipeline, "maybe_free_model_hooks", None)
                if callable(maybe_free):
                    maybe_free()
                for component_name in ("transformer", "vae", "text_encoder"):
                    component = getattr(flowmorph_pipeline, component_name, None)
                    if component is not None and callable(getattr(component, "to", None)):
                        component.to("cpu")

            for variable_name in (
                "ENDPOINT_CACHE",
                "IMAGE_ASSET_CACHE",
                "PROMPT_CONDITIONING_CACHE",
                "ENDPOINT_RECONSTRUCTION_PATHS",
                "SEQUENCE_SESSION",
                "SEQUENCE_RUNNER",
                "session_config",
                "session_template",
                "FLUX_PROMPT_TOKENIZER",
                "flowmorph_pipeline",
                "sequence_runner_for_release",
            ):
                value = globals().pop(variable_name, None)
                if value is not None:
                    del value

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass
                torch.cuda.reset_peak_memory_stats()
                LTX_PRELOAD_ALLOCATED_GIB = torch.cuda.memory_allocated() / 1024**3
                LTX_PRELOAD_RESERVED_GIB = torch.cuda.memory_reserved() / 1024**3
                LTX_GPU_TOTAL_GIB = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )
            else:
                raise RuntimeError("LTX-Video 13B requires a CUDA runtime")

            # The FLUX repository is about 32 GiB and the LTX 13B repository is
            # about 45 GiB. They need not coexist after all FlowMorph PNGs and
            # endpoint checkpoints have been saved. Remove only the exact,
            # reproducible local FLUX model cache—not Drive outputs or LoRA files.
            flux_cache_directory = (
                Path(HF_CACHE_DIR)
                / ("models--" + MODEL_ID.replace("/", "--"))
            )
            if DELETE_LOCAL_FLUX_CACHE_BEFORE_LTX and flux_cache_directory.is_dir():
                expected_parent = Path(HF_CACHE_DIR).resolve()
                if flux_cache_directory.resolve().parent != expected_parent:
                    raise RuntimeError(
                        "Refusing to delete an unexpected FLUX cache path: "
                        f"{flux_cache_directory}"
                    )
                shutil.rmtree(flux_cache_directory)
                print(
                    "Removed the released local FLUX model cache to make room "
                    f"for LTX: {flux_cache_directory}"
                )

            Path(LTX_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            LTX_DISK_FREE_GIB = (
                shutil.disk_usage(LTX_CACHE_DIR).free / 1024**3
            )
            print({
                "gpu": torch.cuda.get_device_name(0),
                "total_gib": round(LTX_GPU_TOTAL_GIB, 3),
                "allocated_gib_after_flux_release": round(
                    LTX_PRELOAD_ALLOCATED_GIB, 3
                ),
                "reserved_gib_after_flux_release": round(
                    LTX_PRELOAD_RESERVED_GIB, 3
                ),
                "ltx_cache_directory": str(Path(LTX_CACHE_DIR)),
                "local_disk_free_gib": round(LTX_DISK_FREE_GIB, 3),
            })
            if LTX_PRELOAD_RESERVED_GIB > LTX_MAX_RESERVED_GIB_BEFORE_LOAD:
                print(
                    "WARNING: CUDA still reserves more than the configured "
                    "preload threshold. If LTX OOMs, restart the runtime, set "
                    "RESUME_RUN_DIRECTORY to this run, and rerun through here."
                )
            """
        ),
        markdown(
            """
            ## 12. Validate and fingerprint the LTX conditioning jobs

            Fourteen stills are assigned to frames `0, 8, …, 104`, so each clip
            has 105 frames. The terminal frame is conditioned on the next anchor.
            When clips are assembled, that one terminal frame is omitted from each
            segment; the next clip begins on the identical canonical endpoint, and
            the final segment approaches the first anchor before playback loops.
            """
        ),
        code(
            """
            from pathlib import Path

            LTX_ROOT = RUN_DIRECTORY / "ltx13b"
            LTX_CLIP_DIRECTORY = LTX_ROOT / "clips"
            LTX_METADATA_DIRECTORY = LTX_ROOT / "metadata"
            LTX_CLIP_DIRECTORY.mkdir(parents=True, exist_ok=True)
            LTX_METADATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

            def make_ltx_motion_prompt(record):
                prompt = LTX_MOTION_PROMPT_TEMPLATE.format(
                    source_science=record["source_science"],
                    target_science=record["target_science"],
                    science_connection=record["science_connection"],
                ).strip()
                word_count = len(prompt.split())
                if word_count > LTX_MAX_PROMPT_WORDS:
                    raise ValueError(
                        f"{record['pair_uid']} LTX prompt has {word_count} words; "
                        f"maximum is {LTX_MAX_PROMPT_WORDS}"
                    )
                return prompt, word_count

            LTX_FRAME_INDICES = [
                index * LTX_FRAMES_PER_CONDITION_INTERVAL
                for index in range(14)
            ]
            LTX_NUM_FRAMES = LTX_FRAME_INDICES[-1] + 1
            if LTX_NUM_FRAMES % 8 != 1:
                raise RuntimeError("LTX frame count must be N*8+1")

            LTX_JOBS = []
            for gap_index, record in enumerate(LTX_GAP_RECORDS):
                prompt, prompt_word_count = make_ltx_motion_prompt(record)
                clip_path = (
                    LTX_CLIP_DIRECTORY
                    / f"{gap_index:04d}_{record['pair_uid']}.mp4"
                )
                metadata_path = (
                    LTX_METADATA_DIRECTORY
                    / f"{gap_index:04d}_{record['pair_uid']}.json"
                )
                contract = {
                    "model_id": LTX_MODEL_ID,
                    "model_revision": LTX_MODEL_REVISION,
                    "upsampler_id": (
                        LTX_UPSAMPLER_ID
                        if LTX_USE_TWO_STAGE_UPSCALING
                        else None
                    ),
                    "upsampler_revision": (
                        LTX_UPSAMPLER_REVISION
                        if LTX_USE_TWO_STAGE_UPSCALING
                        else None
                    ),
                    "pair_uid": record["pair_uid"],
                    "condition_sha256": [
                        file_sha256(path)
                        for path in record["condition_paths"]
                    ],
                    "condition_frame_indices": LTX_FRAME_INDICES,
                    "conditioning_strength": LTX_CONDITIONING_STRENGTH,
                    "num_frames": LTX_NUM_FRAMES,
                    "fps": LTX_FPS,
                    "final_size": [LTX_FINAL_WIDTH, LTX_FINAL_HEIGHT],
                    "prompt": prompt,
                    "negative_prompt": LTX_NEGATIVE_PROMPT,
                    "first_pass_timesteps": LTX_FIRST_PASS_TIMESTEPS,
                    "second_pass_timesteps": LTX_SECOND_PASS_TIMESTEPS,
                    "image_cond_noise_scale": LTX_IMAGE_COND_NOISE_SCALE,
                    "decode_timestep": LTX_DECODE_TIMESTEP,
                    "decode_noise_scale": LTX_DECODE_NOISE_SCALE,
                    "tone_map_compression_ratio": (
                        LTX_TONE_MAP_COMPRESSION_RATIO
                    ),
                }
                fingerprint = stable_fingerprint(contract)
                reusable = False
                if (
                    REUSE_EXISTING_LTX_CLIPS
                    and clip_path.is_file()
                    and metadata_path.is_file()
                ):
                    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                    reusable = (
                        saved.get("status") == "complete"
                        and saved.get("fingerprint") == fingerprint
                    )
                LTX_JOBS.append({
                    "gap_index": gap_index,
                    "record": record,
                    "prompt": prompt,
                    "prompt_word_count": prompt_word_count,
                    "clip_path": clip_path,
                    "metadata_path": metadata_path,
                    "contract": contract,
                    "fingerprint": fingerprint,
                    "reusable": reusable,
                })

            LTX_PENDING_JOBS = [job for job in LTX_JOBS if not job["reusable"]]
            print({
                "cyclic_gap_clips": len(LTX_JOBS),
                "pending_clips": len(LTX_PENDING_JOBS),
                "conditions_per_clip": 14,
                "condition_frame_indices": LTX_FRAME_INDICES,
                "frames_per_clip": LTX_NUM_FRAMES,
                "seconds_per_clip": round(LTX_NUM_FRAMES / LTX_FPS, 3),
                "expected_loop_seconds_without_duplicate_endpoints": round(
                    len(LTX_JOBS) * (LTX_NUM_FRAMES - 1) / LTX_FPS,
                    3,
                ),
            })
            """
        ),
        markdown(
            """
            ## 13. Load one memory-managed LTX 0.9.8 13B stack

            The 13B distilled transformer is stored layerwise in FP8 and computed
            in BF16. Group offload transfers only the layers currently needed to
            CUDA. The pipeline is retained across all pending gaps; if every clip
            fingerprint already matches, no 13B model is loaded.
            """
        ),
        code(
            """
            # Make this cell safe to rerun after an interrupted download or a
            # failed partial model construction.
            for stale_name in (
                "LTX_PIPE",
                "LTX_UPSCALE_PIPE",
                "LTX_UPSAMPLER",
                "ltx_transformer",
            ):
                stale = globals().pop(stale_name, None)
                if stale is not None:
                    try:
                        stale.to("cpu")
                    except (AttributeError, RuntimeError, ValueError):
                        pass
                    del stale
            gc.collect()
            torch.cuda.empty_cache()

            LTX_PIPE = None
            LTX_UPSCALE_PIPE = None
            LTX_UPSAMPLER = None

            if LTX_PENDING_JOBS:
                import os
                import huggingface_hub.constants as hf_hub_constants
                from huggingface_hub import snapshot_download
                from huggingface_hub.constants import HF_XET_CACHE
                from diffusers import AutoModel, LTXConditionPipeline
                from diffusers.hooks import apply_group_offloading
                from diffusers.pipelines.ltx.modeling_latent_upsampler import (
                    LTXLatentUpsamplerModel,
                )
                from diffusers.pipelines.ltx.pipeline_ltx_condition import (
                    LTXVideoCondition,
                )
                from diffusers import LTXLatentUpsamplePipeline

                # Xet reconstructs a target file while retaining a separate
                # chunk cache, which can temporarily duplicate a large shard.
                # Plain resumable HTTP is slower but materially safer on
                # Colab's constrained ephemeral disk.
                if LTX_DISABLE_XET_FOR_DISK_SAFETY:
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                    hf_hub_constants.HF_HUB_DISABLE_XET = True

                # Ask the Hub what is still missing before starting another
                # multi-gigabyte Xet transfer. This accounts for completed files
                # from a prior partial attempt and fails with an actionable disk
                # message rather than an opaque reconstruction error.
                ltx_download_plan = snapshot_download(
                    repo_id=LTX_MODEL_ID,
                    revision=LTX_MODEL_REVISION,
                    cache_dir=LTX_CACHE_DIR,
                    allow_patterns=[
                        "model_index.json",
                        "scheduler/*",
                        "tokenizer/*",
                        "text_encoder/*",
                        "transformer/*",
                        "vae/*",
                    ],
                    dry_run=True,
                )
                ltx_missing_bytes = sum(
                    item.file_size
                    for item in ltx_download_plan
                    if item.will_download
                )
                if LTX_USE_TWO_STAGE_UPSCALING:
                    upsampler_download_plan = snapshot_download(
                        repo_id=LTX_UPSAMPLER_ID,
                        revision=LTX_UPSAMPLER_REVISION,
                        cache_dir=LTX_CACHE_DIR,
                        dry_run=True,
                    )
                    ltx_missing_bytes += sum(
                        item.file_size
                        for item in upsampler_download_plan
                        if item.will_download
                    )
                ltx_free_bytes = shutil.disk_usage(LTX_CACHE_DIR).free
                ltx_required_bytes = (
                    ltx_missing_bytes
                    + int(LTX_DOWNLOAD_HEADROOM_GIB * 1024**3)
                )
                print({
                    "ltx_download_remaining_gib": round(
                        ltx_missing_bytes / 1024**3, 3
                    ),
                    "disk_free_gib": round(ltx_free_bytes / 1024**3, 3),
                    "required_including_headroom_gib": round(
                        ltx_required_bytes / 1024**3, 3
                    ),
                })
                if (
                    CLEAN_INTERRUPTED_LTX_DOWNLOADS_IF_NEEDED
                    and (
                        ltx_free_bytes < ltx_required_bytes
                        or LTX_DISABLE_XET_FOR_DISK_SAFETY
                    )
                ):
                    # Hugging Face checks for a shard's full target size even
                    # when a previous `.incomplete` reconstruction exists.
                    # Xet also retains a separate temporary chunk cache. Keep
                    # every completed blob, but remove these reproducible
                    # interrupted-transfer artifacts so the retry can proceed.
                    incomplete_paths = []
                    for repo_id in (
                        LTX_MODEL_ID,
                        LTX_UPSAMPLER_ID,
                    ):
                        repo_cache = (
                            Path(LTX_CACHE_DIR)
                            / ("models--" + repo_id.replace("/", "--"))
                        )
                        if repo_cache.is_dir():
                            incomplete_paths.extend(
                                path
                                for path in repo_cache.rglob("*.incomplete")
                                if path.is_file()
                            )
                    incomplete_bytes = sum(
                        path.stat().st_size
                        for path in incomplete_paths
                    )
                    for path in incomplete_paths:
                        path.unlink()

                    xet_cache_path = Path(HF_XET_CACHE)
                    xet_cache_bytes = 0
                    if xet_cache_path.is_dir():
                        xet_cache_bytes = sum(
                            path.stat().st_size
                            for path in xet_cache_path.rglob("*")
                            if path.is_file()
                        )
                        shutil.rmtree(xet_cache_path)

                    ltx_free_bytes = shutil.disk_usage(LTX_CACHE_DIR).free
                    print({
                        "interrupted_ltx_files_removed": len(
                            incomplete_paths
                        ),
                        "incomplete_ltx_gib_reclaimed": round(
                            incomplete_bytes / 1024**3, 3
                        ),
                        "xet_temporary_gib_reclaimed": round(
                            xet_cache_bytes / 1024**3, 3
                        ),
                        "disk_free_after_transfer_cleanup_gib": round(
                            ltx_free_bytes / 1024**3, 3
                        ),
                    })
                if ltx_free_bytes < ltx_required_bytes:
                    raise RuntimeError(
                        "Not enough local disk for the remaining LTX download. "
                        "Delete unused /content caches or set LTX_CACHE_DIR to "
                        "a Google Drive directory with at least "
                        f"{ltx_required_bytes / 1024**3:.1f} GiB free."
                    )

                ltx_transformer = AutoModel.from_pretrained(
                    LTX_MODEL_ID,
                    subfolder="transformer",
                    revision=LTX_MODEL_REVISION,
                    cache_dir=LTX_CACHE_DIR,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                if LTX_ENABLE_FP8_LAYERWISE_STORAGE:
                    ltx_transformer.enable_layerwise_casting(
                        storage_dtype=torch.float8_e4m3fn,
                        compute_dtype=torch.bfloat16,
                    )

                LTX_PIPE = LTXConditionPipeline.from_pretrained(
                    LTX_MODEL_ID,
                    revision=LTX_MODEL_REVISION,
                    cache_dir=LTX_CACHE_DIR,
                    transformer=ltx_transformer,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                LTX_PIPE.vae.enable_tiling()
                LTX_PIPE.vae.enable_slicing()

                cuda_device = torch.device("cuda")
                cpu_device = torch.device("cpu")
                if LTX_ENABLE_GROUP_OFFLOAD:
                    LTX_PIPE.transformer.enable_group_offload(
                        onload_device=cuda_device,
                        offload_device=cpu_device,
                        offload_type="leaf_level",
                        use_stream=LTX_GROUP_OFFLOAD_USE_STREAM,
                        low_cpu_mem_usage=LTX_GROUP_OFFLOAD_LOW_CPU_MEMORY,
                    )
                    apply_group_offloading(
                        LTX_PIPE.text_encoder,
                        onload_device=cuda_device,
                        offload_device=cpu_device,
                        offload_type="block_level",
                        num_blocks_per_group=2,
                        use_stream=LTX_GROUP_OFFLOAD_USE_STREAM,
                        low_cpu_mem_usage=LTX_GROUP_OFFLOAD_LOW_CPU_MEMORY,
                    )
                    apply_group_offloading(
                        LTX_PIPE.vae,
                        onload_device=cuda_device,
                        offload_device=cpu_device,
                        offload_type="leaf_level",
                        use_stream=False,
                    )
                else:
                    LTX_PIPE.enable_model_cpu_offload()

                if LTX_USE_TWO_STAGE_UPSCALING:
                    LTX_UPSAMPLER = LTXLatentUpsamplerModel.from_pretrained(
                        LTX_UPSAMPLER_ID,
                        revision=LTX_UPSAMPLER_REVISION,
                        cache_dir=LTX_CACHE_DIR,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )
                    LTX_UPSCALE_PIPE = LTXLatentUpsamplePipeline(
                        vae=LTX_PIPE.vae,
                        latent_upsampler=LTX_UPSAMPLER,
                    )

                print({
                    "ltx_model": LTX_MODEL_ID,
                    "model_loads": 1,
                    "fp8_layerwise_storage": (
                        LTX_ENABLE_FP8_LAYERWISE_STORAGE
                    ),
                    "group_offload": LTX_ENABLE_GROUP_OFFLOAD,
                    "two_stage_upscaling": LTX_USE_TWO_STAGE_UPSCALING,
                })
            else:
                print("Every LTX clip fingerprint matches; model load skipped.")
            """
        ),
        markdown(
            """
            ## 14. Render and save each conditioned LTX gap

            Results are written immediately. The low-resolution distilled pass,
            optional latent upscale, and short texture-refinement pass all reuse
            the same 14 conditions. Each saved segment omits its final exact
            endpoint to avoid a duplicated frame when clips are concatenated.
            """
        ),
        code(
            """
            import subprocess
            import imageio.v2 as imageio
            import imageio_ffmpeg
            import numpy as np
            from IPython.display import Video

            def round_down_for_ltx(value, divisor):
                rounded = value - value % divisor
                if rounded < divisor:
                    raise ValueError(f"Cannot round {value} to a positive LTX size")
                return rounded

            def render_ltx_job(job):
                record = job["record"]
                condition_images = []
                try:
                    for path in record["condition_paths"]:
                        with Image.open(path) as opened:
                            condition_images.append(opened.convert("RGB"))
                    conditions = [
                        LTXVideoCondition(
                            image=image,
                            frame_index=frame_index,
                            strength=LTX_CONDITIONING_STRENGTH,
                        )
                        for image, frame_index in zip(
                            condition_images,
                            LTX_FRAME_INDICES,
                            strict=True,
                        )
                    ]
                    generator = torch.Generator(device="cuda").manual_seed(
                        BASE_SEED + 100_000 + job["gap_index"]
                    )
                    compression = LTX_PIPE.vae_spatial_compression_ratio
                    downscale = 2 / 3 if LTX_USE_TWO_STAGE_UPSCALING else 1.0
                    first_height = round_down_for_ltx(
                        int(LTX_FINAL_HEIGHT * downscale),
                        compression,
                    )
                    first_width = round_down_for_ltx(
                        int(LTX_FINAL_WIDTH * downscale),
                        compression,
                    )
                    first_result = LTX_PIPE(
                        conditions=conditions,
                        prompt=job["prompt"],
                        negative_prompt=LTX_NEGATIVE_PROMPT,
                        width=first_width,
                        height=first_height,
                        num_frames=LTX_NUM_FRAMES,
                        frame_rate=LTX_FPS,
                        timesteps=LTX_FIRST_PASS_TIMESTEPS,
                        decode_timestep=LTX_DECODE_TIMESTEP,
                        decode_noise_scale=LTX_DECODE_NOISE_SCALE,
                        image_cond_noise_scale=LTX_IMAGE_COND_NOISE_SCALE,
                        guidance_scale=LTX_GUIDANCE_SCALE,
                        guidance_rescale=LTX_GUIDANCE_RESCALE,
                        generator=generator,
                        output_type=(
                            "latent"
                            if LTX_USE_TWO_STAGE_UPSCALING
                            else "pil"
                        ),
                    )

                    if LTX_USE_TWO_STAGE_UPSCALING:
                        first_latents = first_result.frames
                        del first_result
                        LTX_UPSAMPLER.to("cuda")
                        upscaled_latents = LTX_UPSCALE_PIPE(
                            latents=first_latents,
                            adain_factor=LTX_ADAIN_FACTOR,
                            tone_map_compression_ratio=(
                                LTX_TONE_MAP_COMPRESSION_RATIO
                            ),
                            output_type="latent",
                        ).frames
                        LTX_UPSAMPLER.to("cpu")
                        del first_latents
                        gc.collect()
                        torch.cuda.empty_cache()

                        render_height = first_height * 2
                        render_width = first_width * 2
                        frames = LTX_PIPE(
                            conditions=conditions,
                            prompt=job["prompt"],
                            negative_prompt=LTX_NEGATIVE_PROMPT,
                            width=render_width,
                            height=render_height,
                            num_frames=LTX_NUM_FRAMES,
                            denoise_strength=LTX_UPSCALE_DENOISE_STRENGTH,
                            timesteps=LTX_SECOND_PASS_TIMESTEPS,
                            latents=upscaled_latents,
                            decode_timestep=LTX_DECODE_TIMESTEP,
                            decode_noise_scale=LTX_DECODE_NOISE_SCALE,
                            image_cond_noise_scale=LTX_IMAGE_COND_NOISE_SCALE,
                            guidance_scale=LTX_GUIDANCE_SCALE,
                            guidance_rescale=LTX_GUIDANCE_RESCALE,
                            generator=generator,
                            output_type="pil",
                        ).frames[0]
                        del upscaled_latents
                    else:
                        frames = first_result.frames[0]
                        del first_result

                    frames = [
                        frame.resize(
                            (LTX_FINAL_WIDTH, LTX_FINAL_HEIGHT),
                            Image.Resampling.LANCZOS,
                        )
                        for frame in frames
                    ]
                    # Drop the terminal exact endpoint. The next clip starts on
                    # that same endpoint; the last clip approaches frame zero.
                    with imageio.get_writer(
                        str(job["clip_path"]),
                        fps=LTX_FPS,
                        codec="libx264",
                        quality=None,
                        macro_block_size=None,
                        output_params=[
                            "-crf",
                            str(LTX_VIDEO_CRF),
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                        ],
                    ) as writer:
                        for frame in frames[:-1]:
                            writer.append_data(np.asarray(frame))
                    job["metadata_path"].write_text(json.dumps({
                        "status": "complete",
                        "fingerprint": job["fingerprint"],
                        "contract": job["contract"],
                        "prompt_word_count": job["prompt_word_count"],
                        "saved_frames": len(frames) - 1,
                        "omitted_terminal_condition_frame": True,
                        "clip_path": str(job["clip_path"]),
                    }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
                    saved_frame_count = len(frames) - 1
                    for frame in frames:
                        frame.close()
                    return saved_frame_count
                finally:
                    for image in condition_images:
                        image.close()

            for pending_index, job in enumerate(LTX_PENDING_JOBS, start=1):
                print(
                    f"LTX gap {pending_index}/{len(LTX_PENDING_JOBS)}: "
                    f"{job['record']['left_uid']} → {job['record']['right_uid']}"
                )
                saved_frames = render_ltx_job(job)
                print({
                    "saved": str(job["clip_path"]),
                    "frames": saved_frames,
                    "prompt_words": job["prompt_word_count"],
                })
                if LTX_DISPLAY_EACH_CLIP:
                    display(Video(
                        filename=str(job["clip_path"]),
                        embed=False,
                        width=LTX_DISPLAY_WIDTH,
                    ))
                gc.collect()
                torch.cuda.empty_cache()

            missing_clips = [
                str(job["clip_path"])
                for job in LTX_JOBS
                if not job["clip_path"].is_file()
            ]
            if missing_clips:
                raise FileNotFoundError(
                    "LTX rendering ended with missing clips: "
                    + ", ".join(missing_clips)
                )
            """
        ),
        markdown(
            """
            ## 15. Concatenate the cyclic clips, audit, preview, and release LTX

            All per-gap MP4s share the same encoding settings, so FFmpeg can join
            them without another lossy video encode. The manifest records the
            FlowMorph conditioning contract, every clip fingerprint, frame count,
            model/memory settings, and the final circular duration.
            """
        ),
        code(
            """
            concat_path = LTX_ROOT / "concat.txt"

            def ffmpeg_concat_line(path):
                absolute = str(Path(path).resolve())
                return "file '" + absolute.replace("'", "'\\\\''") + "'"

            concat_path.write_text(
                "\\n".join(
                    ffmpeg_concat_line(job["clip_path"])
                    for job in LTX_JOBS
                ) + "\\n",
                encoding="utf-8",
            )
            LTX_FINAL_VIDEO_PATH = (
                LTX_ROOT / "flowmorph_conditioned_ltx13b_cyclic.mp4"
            )
            ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.check_call([
                ffmpeg_executable,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(LTX_FINAL_VIDEO_PATH),
            ])

            LTX_FINAL_MANIFEST = LTX_ROOT / "ltx13b_video_manifest.json"
            LTX_FINAL_MANIFEST.write_text(json.dumps({
                "project": PROJECT_NAME,
                "cyclic": True,
                "model_id": LTX_MODEL_ID,
                "model_revision": LTX_MODEL_REVISION,
                "upsampler_id": (
                    LTX_UPSAMPLER_ID
                    if LTX_USE_TWO_STAGE_UPSCALING
                    else None
                ),
                "upsampler_revision": (
                    LTX_UPSAMPLER_REVISION
                    if LTX_USE_TWO_STAGE_UPSCALING
                    else None
                ),
                "flowmorph_rounds": 1,
                "flowmorph_interiors_per_gap": 12,
                "conditions_per_gap": 14,
                "condition_frame_indices": LTX_FRAME_INDICES,
                "frames_generated_per_gap": LTX_NUM_FRAMES,
                "frames_saved_per_gap": LTX_NUM_FRAMES - 1,
                "gap_count": len(LTX_JOBS),
                "final_frame_count": len(LTX_JOBS) * (LTX_NUM_FRAMES - 1),
                "fps": LTX_FPS,
                "duration_seconds": (
                    len(LTX_JOBS) * (LTX_NUM_FRAMES - 1) / LTX_FPS
                ),
                "final_video": str(LTX_FINAL_VIDEO_PATH),
                "flowmorph_conditioning_manifest": str(
                    LTX_CONDITIONING_MANIFEST
                ),
                "fp8_layerwise_storage": LTX_ENABLE_FP8_LAYERWISE_STORAGE,
                "group_offload": LTX_ENABLE_GROUP_OFFLOAD,
                "clips": [
                    {
                        "pair_uid": job["record"]["pair_uid"],
                        "left_uid": job["record"]["left_uid"],
                        "right_uid": job["record"]["right_uid"],
                        "path": str(job["clip_path"]),
                        "fingerprint": job["fingerprint"],
                    }
                    for job in LTX_JOBS
                ],
            }, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")

            if LTX_UPSAMPLER is not None:
                LTX_UPSAMPLER.to("cpu")
            for component_name in ("transformer", "vae", "text_encoder"):
                component = (
                    getattr(LTX_PIPE, component_name, None)
                    if LTX_PIPE is not None
                    else None
                )
                if component is not None and callable(getattr(component, "to", None)):
                    try:
                        component.to("cpu")
                    except (RuntimeError, ValueError):
                        pass
            del LTX_PIPE, LTX_UPSCALE_PIPE, LTX_UPSAMPLER
            gc.collect()
            torch.cuda.empty_cache()

            print({
                "final_video": str(LTX_FINAL_VIDEO_PATH),
                "manifest": str(LTX_FINAL_MANIFEST),
                "duration_seconds": round(
                    len(LTX_JOBS) * (LTX_NUM_FRAMES - 1) / LTX_FPS,
                    3,
                ),
                "cyclic_closure": (
                    "last gap approaches the first canonical endpoint; "
                    "no duplicated boundary frame"
                ),
            })
            display(Video(
                filename=str(LTX_FINAL_VIDEO_PATH),
                embed=False,
                width=LTX_DISPLAY_WIDTH,
            ))
            if DOWNLOAD_LTX_FINAL_VIDEO:
                from google.colab import files
                files.download(str(LTX_FINAL_VIDEO_PATH))
            """
        ),
    ]
)

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"flowmorph-ltx13b-{index:02d}"
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} clean cells")
