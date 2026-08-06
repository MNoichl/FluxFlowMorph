"""Pure helpers for local MiniMax H3 first/last-frame interpolation runs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_H3_MOTION_DIRECTIVE = (
    "The objects in #Image1 morphing into #Image2 . No camera movement, no panning, "
    "no exchange, no cuts. Only objects changing shape, form texture and color. "
    "No alpha blending. Objects moving as little as possible."
)


def _record_prompt(record: Mapping[str, Any]) -> str:
    for field in ("generation_prompt", "prompt"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    raise ValueError(f"anchor {record.get('uid', '<unknown>')!r} has no saved prompt")


def _resolve_record_path(record: Mapping[str, Any], source_run: Path) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"anchor {record.get('uid', '<unknown>')!r} has no image path")

    stated = Path(raw_path).expanduser()
    candidates = [stated]
    if not stated.is_absolute():
        candidates.append(source_run / stated)
    candidates.extend(
        (
            source_run / "base_frames" / stated.name,
            source_run / "frames" / stated.name,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = [path for path in source_run.rglob(stated.name) if path.is_file()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(
            f"anchor image {stated.name!r} is ambiguous inside {source_run}: "
            + ", ".join(str(path) for path in matches[:5])
        )
    raise FileNotFoundError(f"anchor image is missing: {stated}")


def load_h3_anchor_records(source_run: str | Path, *, require_complete: bool = True) -> list[dict[str, Any]]:
    """Load ordered FLUX anchors and their authored prompts from ``base_manifest.json``."""

    run_directory = Path(source_run).expanduser()
    manifest_path = run_directory / "metadata" / "base_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source run has no base manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("base manifest must contain a JSON object")
    if require_complete and payload.get("complete") is not True:
        raise ValueError(f"source base manifest is not marked complete: {manifest_path}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) < 2:
        raise ValueError("base manifest needs at least two ordered anchor records")

    records: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"base record {index} is not a JSON object")
        uid = str(raw_record.get("uid") or f"base_{index:03d}")
        if uid in seen_uids:
            raise ValueError(f"duplicate base anchor uid: {uid}")
        seen_uids.add(uid)
        record = dict(raw_record)
        record["uid"] = uid
        record["source_index"] = index
        record["authored_prompt"] = _record_prompt(raw_record)
        record["resolved_path"] = str(_resolve_record_path(raw_record, run_directory))
        records.append(record)
    return records


def cyclic_h3_pairs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return each adjacent pair plus the final-to-first loop-closing pair."""

    if len(records) < 2:
        raise ValueError("at least two records are required")
    pairs = []
    for index, left in enumerate(records):
        right = records[(index + 1) % len(records)]
        pairs.append(
            {
                "index": index,
                "pair_id": f"pair_{index:04d}_{left['uid']}_to_{right['uid']}",
                "left": dict(left),
                "right": dict(right),
            }
        )
    return pairs


def snap_h3_frame_count(duration_seconds: float, *, fps: int = 24) -> int:
    """Match the official H3 workflow's ``17k+5`` frame-grid expression."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    requested = max(5, round(float(duration_seconds) * fps))
    return int(requested + (5 - (requested % 17)) % 17)


def validate_h3_canvas(width: int, height: int) -> None:
    """Validate the open H3 model's native 768-short-edge canvas contract."""

    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("H3 width and height must be positive multiples of 32")
    if min(width, height) > 768 or max(width, height) > 1344:
        raise ValueError("H3 native output is capped at a 768 short edge and 1344 long edge")


def build_default_h3_prompt(
    *,
    duration_seconds: float,
    trigger: str = "RIJKSOIL",
    motion_directive: str = DEFAULT_H3_MOTION_DIRECTIVE,
) -> str:
    """Wrap the user's motion instruction in MiniMax's official FL2VA syntax."""

    clean_trigger = " ".join(trigger.split())
    if not clean_trigger:
        raise ValueError("trigger must not be empty")
    clean_directive = " ".join(motion_directive.split())
    clean_directive = re.sub(r"#Image1\b", "<Picture 1>", clean_directive, flags=re.IGNORECASE)
    clean_directive = re.sub(r"#Image2\b", "<Picture 2>", clean_directive, flags=re.IGNORECASE)
    if "<Picture 1>" not in clean_directive or "<Picture 2>" not in clean_directive:
        raise ValueError("motion directive must refer to both #Image1 and #Image2")
    if re.search(rf"\b{re.escape(clean_trigger)}\b", clean_directive, flags=re.IGNORECASE):
        raise ValueError("the motion directive must not duplicate the LoRA trigger")
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> aligns with the 0.00-second mark; <Picture 2> aligns with the "
        f"{float(duration_seconds):.2f}-second mark.\n\n"
        "integrated_multimodal_description: [Shot 1] "
        f"{clean_trigger}. {clean_directive} The scene is one continuous locked-off shot. "
        "The opening composition must exactly match <Picture 1>, and the final composition "
        "must settle exactly into <Picture 2>. Do not invent additional objects.\n\n"
        "overall_soundscape: Silence; no dialogue, music, or sound effects.\n"
        "non_diegetic_music: N/A"
    )


def wrap_openai_h3_motion(
    motion_description: str,
    *,
    duration_seconds: float,
    trigger: str = "RIJKSOIL",
) -> str:
    """Apply fixed endpoint/audio constraints around an image-aware OpenAI proposal."""

    clean = " ".join(motion_description.split())
    if len(clean) < 80:
        raise ValueError("OpenAI motion description is unexpectedly short")
    clean = re.sub(r"#Image1\b", "<Picture 1>", clean, flags=re.IGNORECASE)
    clean = re.sub(r"#Image2\b", "<Picture 2>", clean, flags=re.IGNORECASE)
    if "<Picture 1>" not in clean:
        clean = "Beginning exactly at <Picture 1>, " + clean
    if "<Picture 2>" not in clean:
        clean += " The forms settle exactly into <Picture 2>."
    clean = re.sub(rf"\b{re.escape(trigger)}\b[,.]?\s*", "", clean, flags=re.IGNORECASE)
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> aligns with the 0.00-second mark; <Picture 2> aligns with the "
        f"{float(duration_seconds):.2f}-second mark.\n\n"
        f"integrated_multimodal_description: [Shot 1] {trigger}. {clean} "
        "One continuous locked-off shot; no camera movement, cuts, dissolves, alpha blending, "
        "or newly invented objects. Objects move as little as possible.\n\n"
        "overall_soundscape: Silence; no dialogue, music, or sound effects.\n"
        "non_diegetic_music: N/A"
    )


def patch_h3_ui_workflow(
    template: Mapping[str, Any],
    *,
    first_image: str,
    last_image: str,
    prompt: str,
    width: int,
    height: int,
    duration_seconds: float,
    seed: int,
    output_prefix: str,
    diffusion_model: str,
    text_encoder: str,
    video_vae: str,
    audio_vae: str,
) -> dict[str, Any]:
    """Patch the pinned official ComfyUI H3 workflow for one local pair."""

    validate_h3_canvas(width, height)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    if not first_image or not last_image:
        raise ValueError("both ComfyUI input image names are required")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    workflow = copy.deepcopy(dict(template))
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    subgraphs = workflow.get("definitions", {}).get("subgraphs")
    if not isinstance(nodes, list) or not isinstance(links, list) or not isinstance(subgraphs, list):
        raise ValueError("H3 template is not a ComfyUI UI workflow with subgraphs")
    if len(subgraphs) != 1 or not isinstance(subgraphs[0], dict):
        raise ValueError("expected exactly one H3 subgraph definition")
    subgraph_id = subgraphs[0].get("id")
    main_nodes = [node for node in nodes if isinstance(node, dict) and node.get("type") == subgraph_id]
    if len(main_nodes) != 1:
        raise ValueError("could not identify the H3 subgraph instance")
    main = main_nodes[0]
    main_inputs = main.get("inputs")
    widgets = main.get("widgets_values")
    if not isinstance(main_inputs, list) or len(main_inputs) < 4 or not isinstance(widgets, list) or len(widgets) < 9:
        raise ValueError("official H3 subgraph interface changed")
    if [item.get("name") for item in main_inputs[:4]] != ["first_frame", "last_frame", "width", "height"]:
        raise ValueError("official H3 first/last-frame input order changed")

    first_link_id = main_inputs[0].get("link")
    if not isinstance(first_link_id, int):
        raise ValueError("official H3 template has no connected first frame")
    first_links = [link for link in links if isinstance(link, list) and link and link[0] == first_link_id]
    if len(first_links) != 1:
        raise ValueError("could not trace the H3 first-frame link")
    first_node_id = first_links[0][1]
    first_nodes = [node for node in nodes if isinstance(node, dict) and node.get("id") == first_node_id]
    if len(first_nodes) != 1 or first_nodes[0].get("type") != "LoadImage":
        raise ValueError("official H3 first frame is not a LoadImage node")
    first_node = first_nodes[0]
    first_node["widgets_values"] = [first_image, "image"]

    integer_node_ids = [node.get("id") for node in nodes if isinstance(node, dict) and isinstance(node.get("id"), int)]
    integer_link_ids = [link[0] for link in links if isinstance(link, list) and link and isinstance(link[0], int)]
    new_node_id = max(integer_node_ids, default=0) + 1
    new_link_id = max(integer_link_ids, default=0) + 1
    last_node = copy.deepcopy(first_node)
    last_node["id"] = new_node_id
    last_node["pos"] = [first_node.get("pos", [0, 0])[0], first_node.get("pos", [0, 0])[1] + 680]
    last_node["widgets_values"] = [last_image, "image"]
    last_node["outputs"][0]["links"] = [new_link_id]
    if len(last_node.get("outputs", [])) > 1:
        last_node["outputs"][1]["links"] = None
    nodes.append(last_node)
    links.append([new_link_id, new_node_id, 0, main["id"], 1, "IMAGE"])
    main_inputs[1]["link"] = new_link_id

    disconnected = {main_inputs[2].get("link"), main_inputs[3].get("link")}
    disconnected.discard(None)
    workflow["links"] = [
        link for link in links if not (isinstance(link, list) and link and link[0] in disconnected)
    ]
    main_inputs[2]["link"] = None
    main_inputs[3]["link"] = None

    widgets[:9] = [
        prompt,
        int(width),
        int(height),
        float(duration_seconds),
        int(seed),
        diffusion_model,
        text_encoder,
        video_vae,
        audio_vae,
    ]
    save_nodes = [node for node in nodes if isinstance(node, dict) and node.get("type") == "SaveVideo"]
    if len(save_nodes) != 1:
        raise ValueError("expected one SaveVideo node in the official H3 template")
    save_widgets = save_nodes[0].get("widgets_values")
    if not isinstance(save_widgets, list) or not save_widgets:
        raise ValueError("official H3 SaveVideo interface changed")
    save_widgets[0] = output_prefix
    return workflow


def stable_h3_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable cache fingerprint for a JSON-compatible job payload."""

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
