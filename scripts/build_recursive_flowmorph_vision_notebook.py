"""Build the corrected recursive vision + true FlowMorph midpoint notebook.

This builder deliberately reuses the broad, validated structure of the first
recursive notebook, then replaces standalone midpoint generation with actual
three-frame FlowMorph fits. The generated working notebook remains ignored.
"""

from __future__ import annotations

import json
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


runpy.run_path(str(BASE_BUILDER), run_name="__main__")
notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
if len(notebook["cells"]) != 28:
    raise RuntimeError("The base recursive notebook structure changed unexpectedly")

notebook["cells"][0]["source"] = lines(
    r'''
    # Recursive science still-life loop — vision prompts + true FlowMorph midpoints

    This local working notebook keeps the useful structure of the recursive vision notebook while making FlowMorph the interpolation mechanism.

    1. Edit the anchor sciences and prompts directly in section 2.
    2. Generate only the anchor paintings with FLUX.2 Klein and weak previous-anchor continuity.
    3. For every cyclic neighbor pair, send both actual paintings, prompts, and science descriptions to the OpenAI vision model. It returns one literal midpoint prompt describing an interdisciplinary and optical middle ground.
    4. Fit both endpoint images with the project FlowMorph implementation and render exactly three alpha positions using `[source prompt, midpoint prompt, target prompt]`.
    5. Insert the actual decoded FlowMorph α=0.5 image, then repeat the process on the denser cyclic sequence. Defaults: 15 anchors → 30 images → 60 images.
    6. Finish the duplicate-free cyclic sequence with Practical-RIFE, circular SSIM motion equalization, and H.264 export.

    Standalone FLUX generation is never used for recursive midpoint images. At the default one midpoint per gap, every insertion is `raw_frames/frame_001.png` from a completed three-frame FlowMorph run. With multiple midpoints, all ordered interior raw frames come from one FlowMorph run for that gap. Pair runs and checkpoints are written directly into the auto-numbered timestamped Google Drive run directory, so interrupted work can be resumed.
    '''
)

notebook["cells"][1]["source"] = lines(
    r'''
    ## 1. Editable run, model, API, FlowMorph, image, and video settings

    The recursive count grows quickly. With `N` anchors, `M` inserted FlowMorph midpoints per gap, and `R` rounds, the final sequence contains `N × (M + 1)^R` images. The default is 15 standalone anchor generations plus 45 complete three-frame FlowMorph pair runs, producing a 60-image cyclic sequence before RIFE.

    FlowMorph renders `MIDPOINTS_PER_GAP + 2` linear alpha positions. With the default, the only inserted image is alpha `0.5`; alpha `0.0` and `1.0` remain audit reconstructions of the source and target endpoints. Fit and render parameters are exposed below.
    '''
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
    "REUSE_EXISTING_MIDPOINTS = True\n",
    "REUSE_EXISTING_MIDPOINTS = True\n"
    "RESUME_FLOWMORPH_PAIR_RUNS = True\n\n"
    "# True FlowMorph fitting/rendering. frame_count is MIDPOINTS_PER_GAP + 2 (default: 3).\n"
    "FLOWMORPH_FIT_LORA_SCALE = 1.2\n"
    "FLOWMORPH_RENDER_LORA_SCALE = 1.2\n"
    "FLOWMORPH_GUIDANCE_SCALE = 3.6\n"
    "FLOWMORPH_SCHEDULER_POINTS = 100\n"
    "FLOWMORPH_START_TIMESTEP_INDEX = 35\n"
    "FLOWMORPH_SOURCE_OPTIMIZATION_STEPS = 100\n"
    "FLOWMORPH_TARGET_OPTIMIZATION_STEPS = 100\n"
    "FLOWMORPH_PRED_LEARNING_RATE = 0.04\n"
    "FLOWMORPH_U_LEARNING_RATE = 0.01\n"
    "FLOWMORPH_RENDER_INDICES = [*range(35, 100, 5), 99]\n"
    "FLOWMORPH_CHECKPOINT_EVERY = 25\n"
    "FLOWMORPH_SAVE_PAIR_CONTACT_SHEETS = False\n"
    "FLOWMORPH_SAVE_PAIR_ANIMATIONS = False\n",
)
settings = replace_once(
    settings,
    "# Weak continuity for anchors; pair conditioning for recursive midpoints.\n"
    "BASE_CONTINUITY_ENABLED = True\n"
    "BASE_REFERENCE_STRENGTH = 0.12\n"
    "BASE_REFERENCE_BLUR = 16.0\n"
    "MIDPOINT_CONDITIONING_ENABLED = True\n"
    "MIDPOINT_REFERENCE_STRENGTH = 0.08\n"
    "MIDPOINT_REFERENCE_BLUR = 18.0\n"
    "REFERENCE_BACKGROUND = (116, 105, 91)\n"
    "SAVE_SOFT_REFERENCES = False\n",
    "# Weak continuity applies only to standalone anchor generation.\n"
    "BASE_CONTINUITY_ENABLED = True\n"
    "BASE_REFERENCE_STRENGTH = 0.12\n"
    "BASE_REFERENCE_BLUR = 16.0\n"
    "REFERENCE_BACKGROUND = (116, 105, 91)\n"
    "SAVE_SOFT_REFERENCES = False\n",
)
notebook["cells"][2]["source"] = settings.splitlines(keepends=True)

validation = "".join(notebook["cells"][10]["source"])
validation = replace_once(
    validation,
    "openai_calls = round_counts[-1] - round_counts[0]\n",
    "openai_calls = round_counts[-1] - round_counts[0]\n"
    "flowmorph_pair_runs = sum(round_counts[:-1])\n",
)
validation = replace_once(
    validation,
    "for name, strength in (\n"
    "    (\"BASE_REFERENCE_STRENGTH\", BASE_REFERENCE_STRENGTH),\n"
    "    (\"MIDPOINT_REFERENCE_STRENGTH\", MIDPOINT_REFERENCE_STRENGTH),\n"
    "):\n"
    "    if not 0 < strength <= 0.35:\n"
    "        raise ValueError(f\"{name} must lie in (0, 0.35]\")\n",
    "if not 0 < BASE_REFERENCE_STRENGTH <= 0.35:\n"
    "    raise ValueError(\"BASE_REFERENCE_STRENGTH must lie in (0, 0.35]\")\n",
)
validation = replace_once(
    validation,
    "    \"openai_vision_calls\": openai_calls,\n"
    "    \"total_flux_images\": round_counts[-1],\n"
    "    \"cyclic_gaps_per_round\": round_counts[:-1],\n",
    "    \"openai_vision_calls\": openai_calls,\n"
    "    \"standalone_flux_anchor_images\": round_counts[0],\n"
    "    \"flowmorph_pair_runs\": flowmorph_pair_runs,\n"
    "    \"final_generated_sequence_images\": round_counts[-1],\n"
    "    \"cyclic_gaps_per_round\": round_counts[:-1],\n",
)
validation += dedent(
    r'''

    if FLOWMORPH_START_TIMESTEP_INDEX != FLOWMORPH_RENDER_INDICES[0]:
        raise ValueError("The first FLOWMORPH_RENDER_INDICES value must equal FLOWMORPH_START_TIMESTEP_INDEX")
    if FLOWMORPH_RENDER_INDICES != sorted(set(FLOWMORPH_RENDER_INDICES)):
        raise ValueError("FLOWMORPH_RENDER_INDICES must be strictly increasing")
    if FLOWMORPH_RENDER_INDICES[-1] >= FLOWMORPH_SCHEDULER_POINTS:
        raise ValueError("FLOWMORPH_RENDER_INDICES must be smaller than FLOWMORPH_SCHEDULER_POINTS")
    if FLOWMORPH_SOURCE_OPTIMIZATION_STEPS < 1 or FLOWMORPH_TARGET_OPTIMIZATION_STEPS < 1:
        raise ValueError("FlowMorph optimization steps must be positive")
    flowmorph_frame_count = MIDPOINTS_PER_GAP + 2
    flowmorph_alphas = [index / (flowmorph_frame_count - 1) for index in range(flowmorph_frame_count)]
    print({
        "flowmorph_frames_per_pair": flowmorph_frame_count,
        "flowmorph_alphas": flowmorph_alphas,
        "inserted_frame_indices": list(range(1, flowmorph_frame_count - 1)),
    })
    '''
)
notebook["cells"][10]["source"] = validation.splitlines(keepends=True)

contract = "".join(notebook["cells"][17]["source"])
pair_reference_marker = "\ndef pair_soft_reference(left_path, right_path, fraction):\n"
if pair_reference_marker not in contract:
    raise RuntimeError("Could not locate obsolete standalone midpoint reference helper")
contract = contract.split(pair_reference_marker, 1)[0].rstrip() + "\n\nprint(\"Image-aware structured midpoint prompt contract ready for FlowMorph.\")\n"
notebook["cells"][17]["source"] = contract.splitlines(keepends=True)

notebook["cells"][18]["source"] = lines(
    r'''
    ## 9. Recursively fit and insert true FlowMorph midpoint images

    Every cyclic gap receives one FlowMorph run. With the default one midpoint, the OpenAI model supplies the alpha-0.5 prompt and FlowMorph renders `[source, midpoint, target]` at alphas `[0.0, 0.5, 1.0]`. If `MIDPOINTS_PER_GAP` is larger, all requested fractional prompts are generated first and one `(M + 2)`-frame FlowMorph run renders the ordered schedule `[source, midpoint 1, …, midpoint M, target]`; the interior raw frames are inserted.

    Pair run directories live directly under this run's Drive folder and include staged inputs, source and target endpoint checkpoints, optimization histories, raw/display frames, metrics, provenance, and the validated archive. Proposal and FlowMorph fingerprints prevent stale reuse when images, prompts, or numerical settings change. No individual pair images are displayed; each completed round ends with one compact contact sheet.
    '''
)

notebook["cells"][19]["source"] = lines(
    r'''
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
    '''
)

notebook["cells"][20]["source"] = lines(
    r'''
    ## 10. Assemble, preview, and audit the generated cyclic FlowMorph sequence

    No endpoint is duplicated. Every inserted image is a true FlowMorph interior-alpha render; the default is alpha 0.5. The final-to-first gap is included in every recursive round, and optional rotation places the playback boundary at the quietest neighboring pair without changing circular order.
    '''
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
