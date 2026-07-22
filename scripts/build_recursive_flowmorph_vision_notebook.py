"""Build the corrected recursive vision + true FlowMorph midpoint notebook.

This builder deliberately reuses the broad, validated structure of the first
recursive notebook, then replaces standalone midpoint generation with actual
three-frame FlowMorph fits. The generated working notebook is tracked normally.
"""

from __future__ import annotations

import ast
import json
import re
import runpy
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_recursive_vision_notebook.py"
BASE_NOTEBOOK = ROOT / "notebooks" / "StillLife_Recursive_Vision_Interpolation.ipynb"
OUTPUT = ROOT / "notebooks" / "StillLife_Recursive_FlowMorph_Vision.ipynb"


def lines(source: str) -> list[str]:
    return (dedent(source).strip("\n") + "\n").splitlines(keepends=True)


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one builder replacement target: {old!r}")
    return source.replace(old, new, 1)


def literal_assignments(source: str, names: set[str]) -> dict[str, object]:
    """Read selected user-local settings without executing notebook code."""

    values: dict[str, object] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return values
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def restore_literal_assignment(source: str, name: str, value: object) -> str:
    pattern = rf"(?m)^{re.escape(name)}\s*=.*$"
    replacement = f"{name} = {value!r}"
    restored, count = re.subn(pattern, replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not restore local notebook setting {name}")
    return restored


# These values describe the user's local Drive layout and active resume target.
# Preserve them when regenerating the tracked working notebook from this builder.
preserved_local_settings: dict[str, object] = {}
if OUTPUT.is_file():
    try:
        existing_notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))
        existing_settings = "".join(existing_notebook["cells"][2]["source"])
        preserved_local_settings = literal_assignments(
            existing_settings,
            {"DRIVE_PROJECT_BASE", "OPENAI_KEY_FILENAME", "RESUME_RUN_DIRECTORY"},
        )
    except (KeyError, IndexError, json.JSONDecodeError):
        preserved_local_settings = {}


runpy.run_path(str(BASE_BUILDER), run_name="__main__")
notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
if len(notebook["cells"]) != 28:
    raise RuntimeError("The base recursive notebook structure changed unexpectedly")

notebook["cells"][0]["source"] = lines(
    r"""
    # Recursive science still-life loop — vision prompts + true FlowMorph midpoints

    This local working notebook keeps the useful structure of the recursive vision notebook while making FlowMorph the interpolation mechanism.

    1. Edit the anchor sciences and prompts directly in section 2.
    2. Generate only the anchor paintings with FLUX.2 Klein and weak previous-anchor continuity.
    3. For every cyclic neighbor pair, send both actual paintings, prompts, and science descriptions to the OpenAI vision model. It returns one literal midpoint prompt describing an interdisciplinary and optical middle ground.
    4. Prepare one sequence-level FlowMorph model, run one backward probe, and fit every unique image endpoint only once for reuse on both neighboring gaps.
    5. Round 1 inserts one explicitly prompted α=0.5 frame per gap. Round 2 inserts ten alpha positions per gap while reusing one image-aware interpolation prompt across those ten renders. Defaults: 15 anchors → 30 images → 330 images.
    6. Finish the duplicate-free cyclic sequence with Practical-RIFE, circular SSIM motion equalization, and H.264 export.

    Standalone FLUX generation is never used for recursive midpoint images. Only requested interior alphas are rendered; redundant reconstructed endpoints, per-pair model reloads, per-pair backward probes, LPIPS audits, and pair archives are omitted in this explicit art-production mode. Endpoint checkpoints and pair manifests are written directly into the auto-numbered timestamped Google Drive run directory, so interrupted work can be resumed.
    """
)

notebook["cells"][1]["source"] = lines(
    r"""
    ## 1. Editable run, model, API, FlowMorph, image, and video settings

    The two rounds intentionally use different schedules. Round 1 makes one explicitly prompted midpoint in each of 15 gaps. Round 2 sees the resulting 30 images, generates one shared interpolation prompt per gap, and renders ten interior alphas from each cached endpoint pair. This produces 330 unique images before RIFE while fitting only 30 unique endpoints.

    The research defaults use 100 optimization steps per endpoint. This art-production notebook starts at 30, keeps one model loaded, probes it once, and checkpoints every 10 steps. Increase the fit steps only if a one-pair comparison shows a visible benefit.
    """
)

settings = "".join(notebook["cells"][2]["source"])
settings = replace_once(
    settings,
    'PROJECT_NAME = "science_path_recursive_vision"\n',
    'PROJECT_NAME = "science_path_recursive_flowmorph"\n'
    'CONFIG_PATH = f"{PROJECT_ROOT}/configs/full_9b_lora.yaml"\n'
    'PROFILE = "auto"\n',
)
settings = replace_once(
    settings,
    "INTERPOLATION_ROUNDS = 2\nMIDPOINTS_PER_GAP = 1\n",
    "FLOWMORPH_ROUND_SPECS = [\n"
    '    {"midpoint_count": 1, "prompt_mode": "explicit_midpoint"},\n'
    '    {"midpoint_count": 10, "prompt_mode": "shared_midpoint"},\n'
    "]\n"
    "INTERPOLATION_ROUNDS = len(FLOWMORPH_ROUND_SPECS)\n",
)
settings = replace_once(
    settings,
    "REUSE_EXISTING_MIDPOINTS = True\n",
    "REUSE_EXISTING_MIDPOINTS = True\n"
    "RESUME_FLOWMORPH_SEQUENCE = True\n\n"
    "# Sequence-native true FlowMorph fitting/rendering.\n"
    "FLOWMORPH_FIT_LORA_SCALE = 1.2\n"
    "FLOWMORPH_RENDER_LORA_SCALE = 1.2\n"
    "FLOWMORPH_GUIDANCE_SCALE = 3.6\n"
    "FLOWMORPH_SCHEDULER_POINTS = 100\n"
    "FLOWMORPH_START_TIMESTEP_INDEX = 35\n"
    "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 30\n"
    "FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 30\n"
    "FLOWMORPH_PRED_LEARNING_RATE = 0.04\n"
    "FLOWMORPH_U_LEARNING_RATE = 0.01\n"
    "FLOWMORPH_RENDER_INDICES = [*range(35, 100, 5), 99]\n"
    "FLOWMORPH_CHECKPOINT_EVERY = 10\n",
)
settings = replace_once(settings, "RIFE_MULTIPLIER = 4\n", "RIFE_MULTIPLIER = 2\n")
settings = replace_once(
    settings,
    "# Weak continuity for anchors; pair conditioning for recursive midpoints.\n"
    "BASE_CONTINUITY_ENABLED = True\n"
    "BASE_REFERENCE_STRENGTH = 0.12\n"
    "BASE_REFERENCE_BLUR = 16.0\n"
    "BASE_REFERENCE_GRAIN_STRENGTH = 0.035  # Normalized monochrome noise sigma; 0 disables.\n"
    "MIDPOINT_CONDITIONING_ENABLED = True\n"
    "MIDPOINT_REFERENCE_STRENGTH = 0.08\n"
    "MIDPOINT_REFERENCE_BLUR = 18.0\n"
    "REFERENCE_BACKGROUND = (116, 105, 91)\n"
    "SAVE_SOFT_REFERENCES = False\n",
    "# Weak continuity applies only to standalone anchor generation.\n"
    "BASE_CONTINUITY_ENABLED = True\n"
    "BASE_REFERENCE_STRENGTH = 0.12\n"
    "BASE_REFERENCE_BLUR = 16.0\n"
    "BASE_REFERENCE_GRAIN_STRENGTH = 0.035  # Normalized monochrome noise sigma; 0 disables.\n"
    "REFERENCE_BACKGROUND = (116, 105, 91)\n"
    "SAVE_SOFT_REFERENCES = True  # Inspect in base_frames/soft_references and its preview sheet.\n",
)
for setting_name, setting_value in preserved_local_settings.items():
    settings = restore_literal_assignment(settings, setting_name, setting_value)
notebook["cells"][2]["source"] = settings.splitlines(keepends=True)

notebook["cells"][10]["source"] = lines(
    r"""
    if not 3 <= BASE_PROMPT_COUNT <= len(BASE_STAGES):
        raise ValueError(f"BASE_PROMPT_COUNT must be between 3 and {len(BASE_STAGES)}")
    if len(FLOWMORPH_ROUND_SPECS) != 2:
        raise ValueError("This notebook expects exactly two FlowMorph rounds")
    allowed_prompt_modes = {"explicit_midpoint", "shared_midpoint"}
    for index, spec in enumerate(FLOWMORPH_ROUND_SPECS, start=1):
        if spec.get("prompt_mode") not in allowed_prompt_modes:
            raise ValueError(f"Invalid prompt mode in round {index}: {spec}")
        if not 1 <= int(spec.get("midpoint_count", 0)) <= 20:
            raise ValueError(f"Round {index} midpoint_count must be between 1 and 20")
    if FLOWMORPH_ROUND_SPECS[0] != {"midpoint_count": 1, "prompt_mode": "explicit_midpoint"}:
        raise ValueError("Round 1 must use one explicitly prompted midpoint")
    if FLOWMORPH_ROUND_SPECS[1] != {"midpoint_count": 10, "prompt_mode": "shared_midpoint"}:
        raise ValueError("Round 2 must use ten renders sharing one interpolation prompt")
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
    if not 0 < BASE_REFERENCE_STRENGTH <= 0.35:
        raise ValueError("BASE_REFERENCE_STRENGTH must lie in (0, 0.35]")
    if not 0 <= BASE_REFERENCE_GRAIN_STRENGTH <= 0.25:
        raise ValueError("BASE_REFERENCE_GRAIN_STRENGTH must lie in [0, 0.25]")
    if OPENAI_IMAGE_DETAIL not in {"low", "high", "original", "auto"}:
        raise ValueError("OPENAI_IMAGE_DETAIL must be low, high, original, or auto")
    if FLOWMORPH_START_TIMESTEP_INDEX != FLOWMORPH_RENDER_INDICES[0]:
        raise ValueError("The first render index must equal the FlowMorph start index")
    if FLOWMORPH_RENDER_INDICES != sorted(set(FLOWMORPH_RENDER_INDICES)):
        raise ValueError("FLOWMORPH_RENDER_INDICES must be strictly increasing")
    if FLOWMORPH_RENDER_INDICES[-1] >= FLOWMORPH_SCHEDULER_POINTS:
        raise ValueError("FLOWMORPH_RENDER_INDICES must be smaller than scheduler points")
    if FLOWMORPH_SOURCE_OPTIMIZATION_STEPS != FLOWMORPH_TARGET_OPTIMIZATION_STEPS:
        raise ValueError("Sequence-cached endpoints require one shared optimization-step count")
    if FLOWMORPH_SOURCE_OPTIMIZATION_STEPS < 1:
        raise ValueError("FlowMorph optimization steps must be positive")

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
    for spec in FLOWMORPH_ROUND_SPECS:
        round_counts.append(round_counts[-1] * (int(spec["midpoint_count"]) + 1))
    pair_renders = sum(round_counts[:-1])
    openai_calls = pair_renders  # One image-aware prompt per gap in both modes.
    unique_endpoint_fits = round_counts[-2]
    print({
        "anchor_images": BASE_PROMPT_COUNT,
        "sequence_counts": round_counts,
        "openai_vision_calls": openai_calls,
        "sequence_pair_renders": pair_renders,
        "unique_endpoint_fits": unique_endpoint_fits,
        "final_generated_sequence_images": round_counts[-1],
        "round_specs": FLOWMORPH_ROUND_SPECS,
        "rife_multiplier": RIFE_MULTIPLIER,
    })
    print("Anchor order:", " → ".join(ids), "→", ids[0])
    """
)

contract = "".join(notebook["cells"][17]["source"])
pair_reference_marker = "\ndef pair_soft_reference(left_path, right_path, fraction):\n"
if pair_reference_marker not in contract:
    raise RuntimeError("Could not locate obsolete standalone midpoint reference helper")
contract = (
    contract.split(pair_reference_marker, 1)[0].rstrip()
    + '\n\nprint("Image-aware structured midpoint prompt contract ready for FlowMorph.")\n'
)
notebook["cells"][17]["source"] = contract.splitlines(keepends=True)

notebook["cells"][18]["source"] = lines(
    r"""
    ## 9. Recursively fit and insert true FlowMorph midpoint images

    Every cyclic gap receives one FlowMorph run. With the default one midpoint, the OpenAI model supplies the alpha-0.5 prompt and FlowMorph renders `[source, midpoint, target]` at alphas `[0.0, 0.5, 1.0]`. If `MIDPOINTS_PER_GAP` is larger, all requested fractional prompts are generated first and one `(M + 2)`-frame FlowMorph run renders the ordered schedule `[source, midpoint 1, …, midpoint M, target]`; the interior raw frames are inserted.

    Pair run directories live directly under this run's Drive folder and include staged inputs, source and target endpoint checkpoints, optimization histories, raw/display frames, metrics, provenance, and the validated archive. Proposal and FlowMorph fingerprints prevent stale reuse when images, prompts, or numerical settings change. No individual pair images are displayed; each completed round ends with one compact contact sheet.
    """
)

notebook["cells"][19]["source"] = lines(
    r"""
    import gc
    from flowmorph_klein.cli import select_hardware_profile
    from flowmorph_klein.config import ProjectTemplateConfig, load_config, resolve_config
    from flowmorph_klein.pipeline import FlowMorphRunner

    # The reusable package protects the exact 20-frame, 512px research contract.
    # This notebook is explicit experimental art mode: retain dimensional and
    # schedule safety while allowing midpoint-only rendering.
    def validate_recursive_flowmorph_contract(config):
        for name, value in (("width", config.input.width), ("height", config.input.height)):
            if not 256 <= value <= 2048 or value % 16 != 0:
                raise ValueError(f"input.{name} must be 256–2048 and divisible by 16")
        expected_frame_count = MIDPOINTS_PER_GAP + 2
        if config.flowmorph.frame_count != expected_frame_count:
            raise ValueError(
                f"Recursive midpoint FlowMorph runs require frame_count={expected_frame_count}"
            )
        if config.flowmorph.render_conditioning_mode.value != "prompt_schedule":
            raise ValueError("Recursive midpoint runs require prompt_schedule conditioning")
        if len(config.input.bridge_prompts or ()) != expected_frame_count:
            raise ValueError("Prompt schedule must contain source, every midpoint, and target")

    ProjectTemplateConfig._validate_full_shape_contract = validate_recursive_flowmorph_contract
    print("Experimental recursive FlowMorph midpoint contract enabled.")

    # The anchor pipeline is fused and CPU-offloaded for inference. FlowMorph
    # needs its own unfused differentiable transformer path, so release it first.
    release_flux_pipeline()

    def usage_payload(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return usage.model_dump(mode="json") if hasattr(usage, "model_dump") else str(usage)

    def flowmorph_contract(left, right, proposals, round_number, gap_index):
        seed = BASE_SEED + round_number * 100000 + gap_index * 100
        frame_count = len(proposals) + 2
        alphas = [index / (frame_count - 1) for index in range(frame_count)]
        contract = {
            "method": "FlowMorph prompt schedule; insert every decoded raw interior-alpha frame",
            "left_uid": left["uid"],
            "left_image_sha256": file_sha256(left["path"]),
            "left_prompt": left["prompt"],
            "right_uid": right["uid"],
            "right_image_sha256": file_sha256(right["path"]),
            "right_prompt": right["prompt"],
            "midpoint_prompts": [proposal.prompt for proposal in proposals],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "lora_source": LORA_SOURCE,
            "lora_revision": LORA_REVISION,
            "lora_weight_sha256": file_sha256(LOCAL_LORA_PATH),
            "fit_lora_scale": FLOWMORPH_FIT_LORA_SCALE,
            "render_lora_scale": FLOWMORPH_RENDER_LORA_SCALE,
            "guidance_scale": FLOWMORPH_GUIDANCE_SCALE,
            "scheduler_points": FLOWMORPH_SCHEDULER_POINTS,
            "start_timestep_index": FLOWMORPH_START_TIMESTEP_INDEX,
            "source_optimization_steps": FLOWMORPH_SOURCE_OPTIMIZATION_STEPS,
            "target_optimization_steps": FLOWMORPH_TARGET_OPTIMIZATION_STEPS,
            "pred_learning_rate": FLOWMORPH_PRED_LEARNING_RATE,
            "u_learning_rate": FLOWMORPH_U_LEARNING_RATE,
            "render_indices": list(FLOWMORPH_RENDER_INDICES),
            "checkpoint_every": FLOWMORPH_CHECKPOINT_EVERY,
            "frame_count": frame_count,
            "alphas": alphas,
            "midpoint_frame_indices": list(range(1, frame_count - 1)),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "seed": seed,
        }
        serialized = json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest(), contract, seed

    CURRENT_RECORDS = list(BASE_RECORDS)
    ROUND_MANIFESTS = []
    FLOWMORPH_PAIR_RUN_COUNT = 0
    FLOWMORPH_RUN_ROOT = RUN_DIRECTORY / "flowmorph_runs"
    FLOWMORPH_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    for round_number in range(1, INTERPOLATION_ROUNDS + 1):
        round_directory = RUN_DIRECTORY / "rounds" / f"round_{round_number:02d}"
        image_directory = round_directory / "images"
        proposal_directory = round_directory / "proposals"
        pair_root = FLOWMORPH_RUN_ROOT / f"round_{round_number:02d}"
        for directory in (image_directory, proposal_directory, pair_root):
            directory.mkdir(parents=True, exist_ok=True)

        incoming = list(CURRENT_RECORDS)
        outgoing = []
        gap_count = len(incoming)
        for gap_index, left in enumerate(incoming):
            FLOWMORPH_PAIR_RUN_COUNT += 1
            right = incoming[(gap_index + 1) % gap_count]
            outgoing.append(left)
            pair_uid = f"r{round_number:02d}_g{gap_index:04d}"
            proposal_records = []
            for midpoint_index in range(1, MIDPOINTS_PER_GAP + 1):
                fraction = midpoint_index / (MIDPOINTS_PER_GAP + 1)
                uid = f"{pair_uid}_m{midpoint_index:02d}"
                proposal_path = proposal_directory / f"{uid}.json"
                output_path = image_directory / f"{uid}.png"

                request_fingerprint, request_contract = midpoint_request_fingerprint(left, right, fraction)
                reused_proposal = False
                if REUSE_EXISTING_MIDPOINTS and proposal_path.is_file():
                    saved = json.loads(proposal_path.read_text(encoding="utf-8"))
                    if saved.get("request_fingerprint") == request_fingerprint:
                        proposal = MidpointProposal.model_validate(saved["proposal"])
                        response_id = saved.get("openai_response_id")
                        usage = saved.get("usage")
                        reused_proposal = True
                        print(f"Reusing endpoint-verified prompt {uid}")
                    else:
                        print(f"Endpoint or prompt contract changed; regenerating prompt {uid}")
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
                        "request_fingerprint": request_fingerprint,
                        "request_contract": request_contract,
                        "proposal": proposal.model_dump(mode="json"),
                        "openai_model": OPENAI_MODEL,
                        "openai_response_id": response_id,
                        "usage": usage,
                        "image_inputs_stored_in_manifest": False,
                    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    print(
                        f"OpenAI prompt for pair {gap_index + 1}/{gap_count}, "
                        f"position {midpoint_index}/{MIDPOINTS_PER_GAP}: {uid}"
                    )
                proposal_records.append({
                    "uid": uid,
                    "fraction": fraction,
                    "proposal": proposal,
                    "proposal_path": proposal_path,
                    "output_path": output_path,
                    "response_id": response_id,
                    "usage": usage,
                })

            proposals = [item["proposal"] for item in proposal_records]
            prompt_schedule = [left["prompt"], *[item.prompt for item in proposals], right["prompt"]]
            flow_fingerprint, flow_contract, seed = flowmorph_contract(
                left, right, proposals, round_number, gap_index
            )
            pair_run_directory = pair_root / f"{pair_uid}_{flow_fingerprint[:12]}"
            completion_path = image_directory / f"{pair_uid}.flowmorph.json"
            output_paths = [item["output_path"] for item in proposal_records]
            completed = False
            completion = None
            if completion_path.is_file() and all(path.is_file() for path in output_paths):
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                completed = completion.get("flowmorph_fingerprint") == flow_fingerprint

            if not completed:
                overrides = {
                    "run_mode": "experimental",
                    "project.name": f"{PROJECT_NAME}_{pair_uid}",
                    "model.id": MODEL_ID,
                    "model.revision": MODEL_REVISION,
                    "lora.source": str(LOCAL_LORA_PATH),
                    "lora.revision": None,
                    "lora.weight_name": LOCAL_LORA_PATH.name,
                    "lora.adapter_name": LORA_ADAPTER_NAME,
                    "lora.fit_scale": FLOWMORPH_FIT_LORA_SCALE,
                    "lora.render_scale": FLOWMORPH_RENDER_LORA_SCALE,
                    "lora.require_base_9b_compatibility": False,
                    "lora.allow_distilled_9b": True,
                    "input.source_image": str(left["path"]),
                    "input.target_image": str(right["path"]),
                    "input.source_prompt": left["prompt"],
                    "input.target_prompt": right["prompt"],
                    "input.bridge_prompt": None,
                    "input.bridge_prompts": prompt_schedule,
                    "input.width": IMAGE_WIDTH,
                    "input.height": IMAGE_HEIGHT,
                    "flowmorph.scheduler_points": FLOWMORPH_SCHEDULER_POINTS,
                    "flowmorph.start_timestep_index": FLOWMORPH_START_TIMESTEP_INDEX,
                    "flowmorph.optimization_steps_source": FLOWMORPH_SOURCE_OPTIMIZATION_STEPS,
                    "flowmorph.optimization_steps_target": FLOWMORPH_TARGET_OPTIMIZATION_STEPS,
                    "flowmorph.pred_learning_rate": FLOWMORPH_PRED_LEARNING_RATE,
                    "flowmorph.u_learning_rate": FLOWMORPH_U_LEARNING_RATE,
                    "flowmorph.frame_count": len(prompt_schedule),
                    "flowmorph.render_indices": FLOWMORPH_RENDER_INDICES,
                    "flowmorph.alpha_schedule": "linear",
                    "flowmorph.render_conditioning_mode": "prompt_schedule",
                    "flowmorph.checkpoint_every": FLOWMORPH_CHECKPOINT_EVERY,
                    "guidance.scale": FLOWMORPH_GUIDANCE_SCALE,
                    "reproducibility.seed": seed,
                    "paths.input_root": str(RUN_DIRECTORY),
                    "paths.work_root": str(Path(LOCAL_ASSET_ROOT) / PROJECT_NAME / "work" / pair_uid),
                    "paths.result_root": str(pair_root),
                    "paths.hf_cache": HF_CACHE_DIR,
                    "paths.drive_root": None,
                    "output.fps": int(SOURCE_SEQUENCE_FPS),
                    "output.save_contact_sheet": FLOWMORPH_SAVE_PAIR_CONTACT_SHEETS,
                    "output.save_webp": FLOWMORPH_SAVE_PAIR_ANIMATIONS,
                    "output.save_gif": FLOWMORPH_SAVE_PAIR_ANIMATIONS,
                    "output.save_mp4": FLOWMORPH_SAVE_PAIR_ANIMATIONS,
                }
                template = load_config(CONFIG_PATH, overrides=overrides)
                selected_profile = select_hardware_profile(
                    PROFILE if PROFILE != "auto" else template.model.profile
                )
                config = resolve_config(template, selected_profile=selected_profile, check_input_files=True)
                resume_pair = (
                    RESUME_FLOWMORPH_PAIR_RUNS
                    and (pair_run_directory / "run_manifest.json").is_file()
                )
                print(
                    f"FlowMorph round {round_number}, pair {gap_index + 1}/{gap_count}: "
                    f"{len(prompt_schedule)} frames, {MIDPOINTS_PER_GAP} inserted"
                )
                runner = FlowMorphRunner.from_config(config, run_directory=pair_run_directory)
                result = runner.run(resume=resume_pair)
                inserted = []
                for frame_index, proposal_record in enumerate(proposal_records, start=1):
                    midpoint_source = pair_run_directory / "raw_frames" / f"frame_{frame_index:03d}.png"
                    if not midpoint_source.is_file():
                        raise FileNotFoundError(f"FlowMorph interior frame is missing: {midpoint_source}")
                    shutil.copy2(midpoint_source, proposal_record["output_path"])
                    inserted.append({
                        "frame_index": frame_index,
                        "alpha": proposal_record["fraction"],
                        "source_frame": str(midpoint_source),
                        "inserted_image": str(proposal_record["output_path"]),
                    })
                completion = {
                    "pair_uid": pair_uid,
                    "status": "complete",
                    "method": "actual FlowMorph decoded raw interior latent frames",
                    "flowmorph_fingerprint": flow_fingerprint,
                    "flowmorph_contract": flow_contract,
                    "flowmorph_run_directory": str(pair_run_directory),
                    "inserted": inserted,
                    "archive": str(result.archive.path) if result.archive is not None else None,
                }
                completion_path.write_text(
                    json.dumps(completion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                print(f"Inserted {len(inserted)} true FlowMorph interior frame(s) for {pair_uid}")
                del result, runner, config, template
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Reusing fingerprint-verified FlowMorph pair {pair_uid}")

            for proposal_record in proposal_records:
                proposal = proposal_record["proposal"]
                outgoing.append({
                    "uid": proposal_record["uid"],
                    "kind": "flowmorph_midpoint",
                    "round": round_number,
                    "fraction": proposal_record["fraction"],
                    "alpha": proposal_record["fraction"],
                    "left_uid": left["uid"],
                    "right_uid": right["uid"],
                    "science": proposal.science_connection,
                    "visual_correspondence": proposal.visual_correspondence,
                    "prompt": proposal.prompt,
                    "seed": seed,
                    "path": str(proposal_record["output_path"]),
                    "proposal_path": str(proposal_record["proposal_path"]),
                    "flowmorph_completion_path": str(completion_path),
                    "flowmorph_run_directory": completion["flowmorph_run_directory"],
                    "flowmorph_fingerprint": flow_fingerprint,
                    "openai_response_id": proposal_record["response_id"],
                    "usage": proposal_record["usage"],
                })

        CURRENT_RECORDS = outgoing
        round_manifest_path = round_directory / "sequence_manifest.json"
        round_manifest_path.write_text(json.dumps({
            "round": round_number,
            "cyclic": True,
            "interpolation_method": "FlowMorph decoded raw interior-alpha frames",
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
        display(Markdown(f"### FlowMorph recursive round {round_number}: {len(outgoing)} cyclic images"))
        display(preview)
        del preview, round_images

    FINAL_RECORDS = CURRENT_RECORDS
    FINAL_SEQUENCE_MANIFEST = RUN_DIRECTORY / "metadata" / "final_recursive_flowmorph_sequence.json"
    FINAL_SEQUENCE_MANIFEST.write_text(json.dumps({
        "project": PROJECT_NAME,
        "cyclic": True,
        "interpolation_method": "vision-generated interior prompts + actual FlowMorph interior-alpha renders",
        "anchor_count": len(BASE_RECORDS),
        "interpolation_rounds": INTERPOLATION_ROUNDS,
        "midpoints_per_gap": MIDPOINTS_PER_GAP,
        "flowmorph_pair_runs": FLOWMORPH_PAIR_RUN_COUNT,
        "final_count": len(FINAL_RECORDS),
        "round_manifests": ROUND_MANIFESTS,
        "records": FINAL_RECORDS,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print({
        "final_images": len(FINAL_RECORDS),
        "true_flowmorph_interior_images": len(FINAL_RECORDS) - len(BASE_RECORDS),
        "manifest": str(FINAL_SEQUENCE_MANIFEST),
    })
    """
)

# Replace the earlier pair-run implementation with the sequence-native art path.
notebook["cells"][18]["source"] = lines(
    r"""
    ## 9. Sequence-native FlowMorph: cached endpoints, one model, one probe

    Round 1 generates one image-aware midpoint prompt and one true α=0.5 FlowMorph render per anchor gap. Round 2 analyzes each of the resulting 30 neighboring pairs once, then reuses that one interpolation prompt across ten true interior-alpha renders (`1/11` through `10/11`).

    The FLUX.2 model is loaded once. Every unique input image is encoded and fitted once, its endpoint checkpoint is reused for both neighboring gaps, and the production backward probe runs once for the live model. Only requested interior frames are rendered; this art path intentionally omits redundant endpoint reconstructions, pair-level LPIPS, pair archives, and repeated model setup. All endpoint checkpoints, prompts, rendered PNGs, and completion manifests remain resumable in the Drive run directory.
    """
)

notebook["cells"][19]["source"] = lines(
    r"""
    import gc
    from flowmorph_klein.cli import select_hardware_profile
    from flowmorph_klein.config import ProjectTemplateConfig, load_config, resolve_config
    from flowmorph_klein.pipeline import FlowMorphRunner
    from flowmorph_klein.sequence import FlowMorphSequenceSession

    # Explicit art-mode contract: preserve the numerical and geometry safety
    # checks while allowing the sequence engine to render interior alphas only.
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
    print("Sequence-native experimental FlowMorph contract enabled.")

    # The standalone anchor pipeline is fused and CPU-offloaded. Release it;
    # the sequence session loads one unfused differentiable model and retains it.
    release_flux_pipeline()

    def usage_payload(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return usage.model_dump(mode="json") if hasattr(usage, "model_dump") else str(usage)

    def stable_fingerprint(payload):
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def load_or_create_shared_prompt(left, right, round_number, gap_index, prompt_mode, path):
        request_fingerprint, request_contract = midpoint_request_fingerprint(left, right, 0.5)
        prompt_contract = {
            "request_fingerprint": request_fingerprint,
            "round": round_number,
            "gap_index": gap_index,
            "prompt_mode": prompt_mode,
            "one_prompt_reused_for_every_rendered_alpha": prompt_mode == "shared_midpoint",
        }
        combined_fingerprint = stable_fingerprint(prompt_contract)
        if REUSE_EXISTING_MIDPOINTS and path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("combined_fingerprint") == combined_fingerprint:
                return (
                    MidpointProposal.model_validate(saved["proposal"]),
                    saved.get("openai_response_id"),
                    saved.get("usage"),
                    combined_fingerprint,
                )
        proposal, response = propose_midpoint(left, right, 0.5)
        usage = usage_payload(response)
        path.write_text(json.dumps({
            "round": round_number,
            "gap_index": gap_index,
            "left_uid": left["uid"],
            "right_uid": right["uid"],
            "prompt_mode": prompt_mode,
            "combined_fingerprint": combined_fingerprint,
            "prompt_contract": prompt_contract,
            "request_contract": request_contract,
            "proposal": proposal.model_dump(mode="json"),
            "openai_model": OPENAI_MODEL,
            "openai_response_id": response.id,
            "usage": usage,
            "image_inputs_stored_in_manifest": False,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return proposal, response.id, usage, combined_fingerprint

    SEQUENCE_ROOT = RUN_DIRECTORY / "flowmorph_sequence"
    SEQUENCE_SESSION_DIRECTORY = SEQUENCE_ROOT / "session"
    SEQUENCE_ENDPOINT_ROOT = SEQUENCE_ROOT / "endpoints"
    SEQUENCE_ASSET_ROOT = SEQUENCE_ROOT / "encoded_inputs"
    for directory in (SEQUENCE_ROOT, SEQUENCE_ENDPOINT_ROOT, SEQUENCE_ASSET_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    bootstrap_left = BASE_RECORDS[0]
    bootstrap_right = BASE_RECORDS[1]
    bootstrap_schedule = [
        bootstrap_left["prompt"],
        bootstrap_left["prompt"],
        bootstrap_right["prompt"],
    ]
    session_overrides = {
        "run_mode": "experimental",
        "project.name": f"{PROJECT_NAME}_sequence_session",
        "model.id": MODEL_ID,
        "model.revision": MODEL_REVISION,
        "lora.source": str(LOCAL_LORA_PATH),
        "lora.revision": None,
        "lora.weight_name": LOCAL_LORA_PATH.name,
        "lora.adapter_name": LORA_ADAPTER_NAME,
        "lora.fit_scale": FLOWMORPH_FIT_LORA_SCALE,
        "lora.render_scale": FLOWMORPH_RENDER_LORA_SCALE,
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
        "flowmorph.scheduler_points": FLOWMORPH_SCHEDULER_POINTS,
        "flowmorph.start_timestep_index": FLOWMORPH_START_TIMESTEP_INDEX,
        "flowmorph.optimization_steps_source": FLOWMORPH_SOURCE_OPTIMIZATION_STEPS,
        "flowmorph.optimization_steps_target": FLOWMORPH_TARGET_OPTIMIZATION_STEPS,
        "flowmorph.pred_learning_rate": FLOWMORPH_PRED_LEARNING_RATE,
        "flowmorph.u_learning_rate": FLOWMORPH_U_LEARNING_RATE,
        "flowmorph.frame_count": len(bootstrap_schedule),
        "flowmorph.render_indices": FLOWMORPH_RENDER_INDICES,
        "flowmorph.alpha_schedule": "linear",
        "flowmorph.render_conditioning_mode": "prompt_schedule",
        "flowmorph.checkpoint_every": FLOWMORPH_CHECKPOINT_EVERY,
        "guidance.scale": FLOWMORPH_GUIDANCE_SCALE,
        "reproducibility.seed": BASE_SEED,
        "paths.input_root": str(RUN_DIRECTORY),
        "paths.work_root": str(Path(LOCAL_ASSET_ROOT) / PROJECT_NAME / "sequence_work"),
        "paths.result_root": str(SEQUENCE_ROOT),
        "paths.hf_cache": HF_CACHE_DIR,
        "paths.drive_root": None,
        "output.fps": int(SOURCE_SEQUENCE_FPS),
        "output.save_contact_sheet": False,
        "output.save_webp": False,
        "output.save_gif": False,
        "output.save_mp4": False,
        # Required by the reusable config validator. The sequence session does
        # not call FlowMorphRunner.run(), so no research archive is created.
        "output.create_zip": True,
    }
    session_template = load_config(CONFIG_PATH, overrides=session_overrides)
    session_profile = select_hardware_profile(
        PROFILE if PROFILE != "auto" else session_template.model.profile
    )
    session_config = resolve_config(
        session_template,
        selected_profile=session_profile,
        check_input_files=True,
    )
    session_resume = (
        RESUME_FLOWMORPH_SEQUENCE
        and (SEQUENCE_SESSION_DIRECTORY / "run_manifest.json").is_file()
    )
    SEQUENCE_RUNNER = FlowMorphRunner.from_config(
        session_config,
        run_directory=SEQUENCE_SESSION_DIRECTORY,
    )
    SEQUENCE_RUNNER.prepare(resume=session_resume)
    SEQUENCE_SESSION = FlowMorphSequenceSession(SEQUENCE_RUNNER)
    PROBE_REPORT = SEQUENCE_SESSION.run_backward_probe_once()
    print({
        "model_loads": 1,
        "backward_probes": 1,
        "fit_steps_per_unique_endpoint": FLOWMORPH_SOURCE_OPTIMIZATION_STEPS,
        "probe_peak_reserved_gib": round(PROBE_REPORT.peak_reserved_vram_bytes / (1024 ** 3), 3),
    })

    IMAGE_ASSET_CACHE, PROMPT_CONDITIONING_CACHE = SEQUENCE_SESSION.seed_prepared_assets(
        bootstrap_left["uid"],
        bootstrap_right["uid"],
    )
    ENDPOINT_CACHE = {}
    ENDPOINT_FINGERPRINTS = {}
    ROUND_MANIFESTS = []
    FLOWMORPH_PAIR_RENDER_COUNT = 0
    OPENAI_SHARED_PROMPT_COUNT = 0
    UNIQUE_ENDPOINT_FIT_COUNT = 0
    CURRENT_RECORDS = list(BASE_RECORDS)

    def endpoint_fingerprint(record):
        return stable_fingerprint({
            "uid": record["uid"],
            "image_sha256": file_sha256(record["path"]),
            "prompt": record["prompt"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "lora_sha256": file_sha256(LOCAL_LORA_PATH),
            "fit_lora_scale": FLOWMORPH_FIT_LORA_SCALE,
            "guidance_scale": FLOWMORPH_GUIDANCE_SCALE,
            "scheduler_points": FLOWMORPH_SCHEDULER_POINTS,
            "start_timestep_index": FLOWMORPH_START_TIMESTEP_INDEX,
            "optimization_steps": FLOWMORPH_SOURCE_OPTIMIZATION_STEPS,
            "pred_learning_rate": FLOWMORPH_PRED_LEARNING_RATE,
            "u_learning_rate": FLOWMORPH_U_LEARNING_RATE,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
        })

    for round_number, round_spec in enumerate(FLOWMORPH_ROUND_SPECS, start=1):
        midpoint_count = int(round_spec["midpoint_count"])
        prompt_mode = str(round_spec["prompt_mode"])
        fractions = [index / (midpoint_count + 1) for index in range(1, midpoint_count + 1)]
        round_directory = RUN_DIRECTORY / "rounds" / f"round_{round_number:02d}"
        image_directory = round_directory / "images"
        proposal_directory = round_directory / "proposals"
        for directory in (round_directory, image_directory, proposal_directory):
            directory.mkdir(parents=True, exist_ok=True)

        incoming = list(CURRENT_RECORDS)
        gap_count = len(incoming)
        pair_jobs = []

        # One image-aware LLM proposal per gap. In round 2 it is reused for all
        # ten alpha positions instead of making ten calls or inventing ten prompts.
        for gap_index, left in enumerate(incoming):
            right = incoming[(gap_index + 1) % gap_count]
            pair_uid = f"r{round_number:02d}_g{gap_index:04d}"
            proposal_path = proposal_directory / f"{pair_uid}_shared.json"
            proposal, response_id, usage, proposal_fingerprint = load_or_create_shared_prompt(
                left,
                right,
                round_number,
                gap_index,
                prompt_mode,
                proposal_path,
            )
            OPENAI_SHARED_PROMPT_COUNT += 1
            frame_records = []
            for midpoint_index, fraction in enumerate(fractions, start=1):
                uid = f"{pair_uid}_m{midpoint_index:02d}"
                frame_records.append({
                    "uid": uid,
                    "fraction": fraction,
                    "output_path": image_directory / f"{uid}.png",
                })
            pair_contract = {
                "method": "sequence-cached FlowMorph interior-alpha rendering",
                "round": round_number,
                "prompt_mode": prompt_mode,
                "left_uid": left["uid"],
                "left_image_sha256": file_sha256(left["path"]),
                "left_prompt": left["prompt"],
                "right_uid": right["uid"],
                "right_image_sha256": file_sha256(right["path"]),
                "right_prompt": right["prompt"],
                "proposal_fingerprint": proposal_fingerprint,
                "shared_midpoint_prompt": proposal.prompt,
                "alphas": fractions,
                "render_indices": list(FLOWMORPH_RENDER_INDICES),
                "render_lora_scale": FLOWMORPH_RENDER_LORA_SCALE,
                "guidance_scale": FLOWMORPH_GUIDANCE_SCALE,
            }
            pair_fingerprint = stable_fingerprint(pair_contract)
            completion_path = image_directory / f"{pair_uid}.flowmorph.json"
            completion = None
            completed = False
            if completion_path.is_file() and all(item["output_path"].is_file() for item in frame_records):
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                completed = completion.get("pair_fingerprint") == pair_fingerprint
            pair_jobs.append({
                "pair_uid": pair_uid,
                "left": left,
                "right": right,
                "proposal": proposal,
                "proposal_path": proposal_path,
                "response_id": response_id,
                "usage": usage,
                "frame_records": frame_records,
                "pair_contract": pair_contract,
                "pair_fingerprint": pair_fingerprint,
                "completion_path": completion_path,
                "completion": completion,
                "completed": completed,
            })

        missing_prompts = []
        for record in incoming:
            if record["prompt"] not in PROMPT_CONDITIONING_CACHE:
                missing_prompts.append(record["prompt"])
        for job in pair_jobs:
            prompt = job["proposal"].prompt
            if prompt not in PROMPT_CONDITIONING_CACHE:
                missing_prompts.append(prompt)
        missing_images = {
            record["uid"]: (
                record["path"],
                SEQUENCE_ASSET_ROOT / f"{record['uid']}.png",
            )
            for record in incoming
            if record["uid"] not in IMAGE_ASSET_CACHE
        }
        if missing_prompts or missing_images:
            new_prompts, new_images = SEQUENCE_SESSION.encode_missing_assets(
                prompts=missing_prompts,
                images=missing_images,
            )
            PROMPT_CONDITIONING_CACHE.update(new_prompts)
            IMAGE_ASSET_CACHE.update(new_images)
        print({
            "round": round_number,
            "encoded_unique_images_total": len(IMAGE_ASSET_CACHE),
            "encoded_unique_prompts_total": len(PROMPT_CONDITIONING_CACHE),
        })

        # Fit each unique incoming image once. An anchor fitted in round 1 is
        # reused in round 2 and on both its left and right neighboring gaps.
        for fit_index, record in enumerate(incoming, start=1):
            uid = record["uid"]
            if uid in ENDPOINT_CACHE:
                continue
            fingerprint = endpoint_fingerprint(record)
            checkpoint_directory = SEQUENCE_ENDPOINT_ROOT / f"{uid}_{fingerprint[:12]}"
            result = SEQUENCE_SESSION.fit_endpoint(
                endpoint_key=uid,
                asset=IMAGE_ASSET_CACHE[uid],
                conditioning=PROMPT_CONDITIONING_CACHE[record["prompt"]],
                checkpoint_directory=checkpoint_directory,
                resume=RESUME_FLOWMORPH_SEQUENCE and checkpoint_directory.exists(),
            )
            ENDPOINT_CACHE[uid] = result.endpoint
            ENDPOINT_FINGERPRINTS[uid] = fingerprint
            UNIQUE_ENDPOINT_FIT_COUNT += 1
            print(
                f"Round {round_number} endpoint {fit_index}/{len(incoming)}: {uid}; "
                f"steps={result.completed_steps}; checkpoint_reused={result.resumed}"
            )

        # Render every pending pair while the transformer remains resident.
        pending_frames = []
        pending_paths = []
        pending_jobs = []
        for pair_index, job in enumerate(pair_jobs, start=1):
            FLOWMORPH_PAIR_RENDER_COUNT += 1
            if job["completed"]:
                print(f"Reusing completed pair {job['pair_uid']}")
                continue
            left = job["left"]
            right = job["right"]
            shared = PROMPT_CONDITIONING_CACHE[job["proposal"].prompt]
            rendered = SEQUENCE_SESSION.render_midpoints(
                source=ENDPOINT_CACHE[left["uid"]],
                target=ENDPOINT_CACHE[right["uid"]],
                source_conditioning=PROMPT_CONDITIONING_CACHE[left["prompt"]],
                target_conditioning=PROMPT_CONDITIONING_CACHE[right["prompt"]],
                midpoint_conditionings=[shared] * midpoint_count,
                alphas=fractions,
            )
            start_index = len(pending_frames)
            pending_frames.extend(rendered)
            pending_paths.extend(item["output_path"] for item in job["frame_records"])
            job["pending_slice"] = (start_index, start_index + len(rendered))
            pending_jobs.append(job)
            print(
                f"Rendered pair {pair_index}/{len(pair_jobs)}: {job['pair_uid']} "
                f"({midpoint_count} interior frame(s))"
            )

        # One component swap and streaming VAE decode for the entire round.
        if pending_frames:
            SEQUENCE_SESSION.decode_frames_to_paths(
                pending_frames,
                pending_paths,
                restore_transformer=False,
            )
        for job in pending_jobs:
            start_index, end_index = job["pending_slice"]
            inserted = [
                {
                    "alpha": item["fraction"],
                    "image": str(item["output_path"]),
                    "shared_prompt": job["proposal"].prompt,
                }
                for item in job["frame_records"]
            ]
            completion = {
                "status": "complete",
                "pair_uid": job["pair_uid"],
                "pair_fingerprint": job["pair_fingerprint"],
                "pair_contract": job["pair_contract"],
                "sequence_session_directory": str(SEQUENCE_SESSION_DIRECTORY),
                "left_endpoint_checkpoint_fingerprint": ENDPOINT_FINGERPRINTS[job["left"]["uid"]],
                "right_endpoint_checkpoint_fingerprint": ENDPOINT_FINGERPRINTS[job["right"]["uid"]],
                "rendered_latent_count": end_index - start_index,
                "inserted": inserted,
            }
            job["completion_path"].write_text(
                json.dumps(completion, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            job["completion"] = completion
            job["completed"] = True
        del pending_frames
        gc.collect()

        outgoing = []
        for job in pair_jobs:
            left = job["left"]
            right = job["right"]
            proposal = job["proposal"]
            outgoing.append(left)
            for frame_record in job["frame_records"]:
                outgoing.append({
                    "uid": frame_record["uid"],
                    "kind": "flowmorph_midpoint",
                    "round": round_number,
                    "prompt_mode": prompt_mode,
                    "fraction": frame_record["fraction"],
                    "alpha": frame_record["fraction"],
                    "left_uid": left["uid"],
                    "right_uid": right["uid"],
                    "science": proposal.science_connection,
                    "visual_correspondence": proposal.visual_correspondence,
                    "prompt": proposal.prompt,
                    "path": str(frame_record["output_path"]),
                    "proposal_path": str(job["proposal_path"]),
                    "flowmorph_completion_path": str(job["completion_path"]),
                    "flowmorph_sequence_session": str(SEQUENCE_SESSION_DIRECTORY),
                    "flowmorph_fingerprint": job["pair_fingerprint"],
                    "openai_response_id": job["response_id"],
                    "usage": job["usage"],
                })

        CURRENT_RECORDS = outgoing
        round_manifest_path = round_directory / "sequence_manifest.json"
        round_manifest_path.write_text(json.dumps({
            "round": round_number,
            "cyclic": True,
            "interpolation_method": "sequence-cached true FlowMorph interior-alpha renders",
            "prompt_mode": prompt_mode,
            "one_shared_prompt_per_gap": True,
            "input_count": len(incoming),
            "midpoints_per_gap": midpoint_count,
            "alphas": fractions,
            "output_count": len(outgoing),
            "records": outgoing,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ROUND_MANIFESTS.append(str(round_manifest_path))

        # Contact sheets use small thumbnails; never allocate a 330 × 1024px mosaic.
        round_contact_sheet = round_directory / "contact_sheet.png"
        round_images = []
        for item in outgoing:
            preview_image = Image.open(item["path"]).convert("RGB")
            preview_image.thumbnail((160, 160))
            round_images.append(preview_image)
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
        display(Markdown(f"### Sequence FlowMorph round {round_number}: {len(outgoing)} cyclic images"))
        display(preview)
        preview.close()

    FINAL_RECORDS = CURRENT_RECORDS
    FINAL_SEQUENCE_MANIFEST = RUN_DIRECTORY / "metadata" / "final_recursive_flowmorph_sequence.json"
    FINAL_SEQUENCE_MANIFEST.write_text(json.dumps({
        "project": PROJECT_NAME,
        "cyclic": True,
        "interpolation_method": "one-model sequence-cached true FlowMorph",
        "anchor_count": len(BASE_RECORDS),
        "round_specs": FLOWMORPH_ROUND_SPECS,
        "model_loads": 1,
        "backward_probes": 1,
        "unique_endpoint_fits": UNIQUE_ENDPOINT_FIT_COUNT,
        "pair_renders": FLOWMORPH_PAIR_RENDER_COUNT,
        "one_openai_prompt_per_gap": True,
        "openai_prompt_count": OPENAI_SHARED_PROMPT_COUNT,
        "final_count": len(FINAL_RECORDS),
        "round_manifests": ROUND_MANIFESTS,
        "records": FINAL_RECORDS,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print({
        "final_images": len(FINAL_RECORDS),
        "unique_endpoint_fits": UNIQUE_ENDPOINT_FIT_COUNT,
        "pair_renders": FLOWMORPH_PAIR_RENDER_COUNT,
        "model_loads": 1,
        "backward_probes": 1,
        "manifest": str(FINAL_SEQUENCE_MANIFEST),
    })

    # Free the retained 9B model before the RIFE subprocess claims the GPU.
    for component_name in ("transformer", "vae", "text_encoder"):
        component = getattr(SEQUENCE_RUNNER.pipeline, component_name, None)
        if component is not None and callable(getattr(component, "to", None)):
            component.to("cpu")
    del ENDPOINT_CACHE, IMAGE_ASSET_CACHE, PROMPT_CONDITIONING_CACHE
    del SEQUENCE_SESSION, SEQUENCE_RUNNER, session_config, session_template
    gc.collect()
    torch.cuda.empty_cache()
    print("Released the sequence FlowMorph model; RIFE can now use the GPU.")
    """
)

notebook["cells"][20]["source"] = lines(
    r"""
    ## 10. Assemble, preview, and audit the generated cyclic FlowMorph sequence

    No endpoint is duplicated. Round 1 contributes one α=0.5 render per gap; round 2 contributes ten renders per gap at `1/11 … 10/11`, all driven by the gap's single shared image-aware prompt. The final-to-first gap is included in both rounds, and optional rotation places the playback boundary at the quietest neighboring pair without changing circular order.
    """
)

final_video_cell = "".join(notebook["cells"][27]["source"])
final_video_cell = replace_once(
    final_video_cell,
    '"recursive_vision_rife_ssim_loop.mp4"',
    '"recursive_flowmorph_vision_rife_ssim_loop.mp4"',
)
notebook["cells"][27]["source"] = final_video_cell.splitlines(keepends=True)

notebook["metadata"]["colab"]["name"] = OUTPUT.name
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} clean cells")
