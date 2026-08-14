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
    "A static locked-off view begins exactly at #Image1 and ends exactly at #Image2. From the "
    "first moment, several spatially separated mapped forms across the left, center, and right "
    "change concurrently through minute, temporally coherent increments. Every contour, centroid, "
    "scale, material, texture, and color follows a continuous path whose speed and direction vary "
    "gently, so adjacent-frame increments are nearly imperceptible in isolation while the cumulative "
    "change remains clear across the full shot. Each mapping begins immediately at a gentle nonzero "
    "rate, flows steadily through "
    "the middle, and settles through progressively smaller adjustments into #Image2. Where the "
    "endpoints support it, backgrounds and broad color, shadow, or texture fields blend, flow, "
    "drift, spread, and reshape as stable opaque painterly fields with equally smooth temporal "
    "motion. Each mapped object remains solid, sharply resolved, and continuously identifiable. "
    "The whole frame remains one integrated, fluid scene rather than a sequence of discrete changes."
)

SOURCE_ONLY_PROMPT_TOKENS = ("RIJKSOIL",)

H3_ANCHOR_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)

H3_FINISHING_SOURCES = (
    (
        "border_stabilized",
        "minimax_h3_border_stabilized_cyclic_loop.mp4",
        "border_stabilization.json",
        "frame_count",
    ),
    (
        "rife_x2",
        "minimax_h3_rife_x2_cyclic_loop.mp4",
        "rife_report.json",
        "output_unique_frames",
    ),
    (
        "native_h3",
        "minimax_h3_native_cyclic_loop.mp4",
        "native_assembly.json",
        "native_unique_frames",
    ),
)


def recover_h3_finishing_source(run_directory: str | Path) -> dict[str, Any] | None:
    """Recover the best completed finishing video persisted inside one H3 run."""

    run_path = Path(run_directory).expanduser()
    for stage, video_name, report_name, count_key in H3_FINISHING_SOURCES:
        video_path = run_path / "video" / video_name
        metadata_path = run_path / "metadata" / report_name
        try:
            video_stat = video_path.stat()
            metadata_stat = metadata_path.stat()
            if video_stat.st_size <= 0 or metadata_stat.st_size <= 0:
                continue
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame_count = int(source_metadata[count_key])
            fps = float(source_metadata["fps"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if frame_count > 0 and fps > 0:
            return {
                "run_directory": run_path,
                "stage": stage,
                "video": video_path,
                "metadata": metadata_path,
                "frame_count": frame_count,
                "fps": fps,
                "freshness": max(video_stat.st_mtime, metadata_stat.st_mtime),
            }
    return None


def discover_h3_finishing_source(
    project_root: str | Path,
    *,
    preferred_run: str | Path | None = None,
) -> dict[str, Any] | None:
    """Use the current H3 run when complete, otherwise find the newest completed run."""

    if preferred_run is not None:
        recovered = recover_h3_finishing_source(preferred_run)
        if recovered is not None:
            return {**recovered, "selection": "current_run"}

    root = Path(project_root).expanduser()
    if not root.is_dir():
        return None
    run_directories: set[Path] = set()
    for _, _, report_name, _ in H3_FINISHING_SOURCES:
        run_directories.update(
            report_path.parent.parent for report_path in root.rglob(f"metadata/{report_name}")
        )
    recovered_sources = [
        recovered
        for run_directory in run_directories
        if (recovered := recover_h3_finishing_source(run_directory)) is not None
    ]
    if not recovered_sources:
        return None
    newest = max(
        recovered_sources,
        key=lambda recovered: (recovered["freshness"], str(recovered["run_directory"])),
    )
    return {**newest, "selection": "latest_completed_h3_run"}


def strip_h3_source_only_tokens(
    text: str,
    *,
    tokens: Sequence[str] = SOURCE_ONLY_PROMPT_TOKENS,
) -> str:
    """Remove prompt tokens that belong to the upstream FLUX/LoRA model only."""

    clean = str(text)
    for token in tokens:
        normalized = " ".join(str(token).split())
        if normalized:
            clean = re.sub(rf"\b{re.escape(normalized)}\b\s*[,;:.\-]?\s*", "", clean, flags=re.IGNORECASE)
    return " ".join(clean.split())


def _record_prompt(record: Mapping[str, Any]) -> str:
    for field in ("generation_prompt", "prompt"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _natural_relative_path_key(
    path: Path,
    root: Path,
) -> tuple[tuple[tuple[int, int | str], ...], str]:
    relative = path.relative_to(root).as_posix()
    return (
        tuple(
            (0, int(token)) if token.isdigit() else (1, token.casefold())
            for token in re.split(r"(\d+)", relative)
            if token
        ),
        relative,
    )


def _directory_anchor_records(source_directory: Path) -> list[dict[str, Any]]:
    image_paths = sorted(
        (
            path
            for path in source_directory.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in H3_ANCHOR_IMAGE_SUFFIXES
            and not any(part.startswith(".") for part in path.relative_to(source_directory).parts)
        ),
        key=lambda path: _natural_relative_path_key(path, source_directory),
    )
    if len(image_paths) < 2:
        suffixes = ", ".join(sorted(H3_ANCHOR_IMAGE_SUFFIXES))
        raise ValueError(
            f"source directory needs at least two supported images; found {len(image_paths)} "
            f"under {source_directory} (supported extensions: {suffixes})"
        )

    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        stem_slug = re.sub(r"[^a-z0-9]+", "_", image_path.stem.casefold()).strip("_")
        uid = f"image_{index:04d}_{stem_slug[:48] or 'anchor'}"
        records.append(
            {
                "uid": uid,
                "source_index": index,
                "path": image_path.relative_to(source_directory).as_posix(),
                "resolved_path": str(image_path.resolve()),
                "authored_prompt": "",
                "source_kind": "directory_scan",
            }
        )
    return records


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
    """Load ordered H3 anchors from a manifest or a plain image directory.

    A ``metadata/base_manifest.json`` remains authoritative when present. If it is
    absent, supported images anywhere below ``source_run`` are natural-sorted by
    relative path and receive stable synthetic IDs plus blank optional prompts.
    """

    run_directory = Path(source_run).expanduser()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {run_directory}")
    manifest_path = run_directory / "metadata" / "base_manifest.json"
    if not manifest_path.is_file():
        return _directory_anchor_records(run_directory)
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
        record["source_kind"] = "base_manifest"
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
    motion_directive: str = DEFAULT_H3_MOTION_DIRECTIVE,
    trigger: str | None = None,
) -> str:
    """Wrap the motion instruction in FL2VA syntax; ignore legacy FLUX trigger arguments."""

    _ = trigger  # Backward compatibility for an already-open pre-fix notebook.
    clean_directive = strip_h3_source_only_tokens(motion_directive)
    clean_directive = re.sub(r"#Image1\b", "<Picture 1>", clean_directive, flags=re.IGNORECASE)
    clean_directive = re.sub(r"#Image2\b", "<Picture 2>", clean_directive, flags=re.IGNORECASE)
    if "<Picture 1>" not in clean_directive or "<Picture 2>" not in clean_directive:
        raise ValueError("motion directive must refer to both #Image1 and #Image2")
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the target video; "
        f"<Picture 2> (from [Shot 1]) aligns with the {float(duration_seconds):.2f}-second "
        "mark of the target video.\n\n"
        "integrated_multimodal_description: [Shot 1] "
        f"{clean_directive} The camera holds a Static Shot with unchanged framing, lens, and "
        "viewpoint. Spatially separated forms on the left, center, and right transform concurrently "
        "throughout the shot, with overlapping local changes at every intermediate time. The frame "
        "remains one integrated scene: no frame-spanning dividing line, moving sheet or band, "
        "single propagation front, or side-by-side display separates earlier and later states. "
        "Where supported by the endpoints, background planes and broad color, shadow, or texture "
        "fields blend and flow locally through overlapping opaque intermediate states. The tabletop, "
        "illumination, object density, and negative space evolve together until the exact <Picture 2> "
        "composition is reached."
        "\n\noverall_soundscape: N/A\n"
        "non_diegetic_music: N/A"
    )


def wrap_openai_h3_motion(
    motion_description: str,
    *,
    duration_seconds: float,
    trigger: str | None = None,
) -> str:
    """Apply fixed constraints; ignore legacy upstream-FLUX trigger arguments."""

    _ = trigger  # Backward compatibility for an already-open pre-fix notebook.
    clean = strip_h3_source_only_tokens(motion_description)
    if len(clean) < 80:
        raise ValueError("OpenAI motion description is unexpectedly short")
    clean = re.sub(
        r"#Image1\b|<Picture\s+1>|\bPicture\s+1\b",
        "<Picture 1>",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"#Image2\b|<Picture\s+2>|\bPicture\s+2\b",
        "<Picture 2>",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"^integrated_multimodal_description:\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^\[Shot 1\]\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not re.search(r'[.!?][\"\u201d\u2019\']?$', clean):
        raise ValueError("OpenAI motion description must end with a complete sentence")
    if "<Picture 1>" not in clean:
        clean = "Beginning exactly from <Picture 1>, " + clean
    if "<Picture 2>" not in clean:
        clean += " The forms settle exactly into <Picture 2>."
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the target video; "
        f"<Picture 2> (from [Shot 1]) aligns with the {float(duration_seconds):.2f}-second "
        "mark of the target video.\n\n"
        f"integrated_multimodal_description: [Shot 1] {clean} "
        "The camera holds a Static Shot with unchanged framing, lens, and viewpoint. From early in "
        "the shot onward, multiple spatially separated forms on the left, center, and right transform "
        "concurrently at overlapping but slightly offset rates. Every contour, centroid, scale, "
        "material, texture, color, highlight, and broad field follows a continuous path whose speed "
        "and direction vary gently. Adjacent frames differ through minute coherent increments that "
        "are nearly imperceptible in isolation, while the cumulative transformation remains clear "
        "across the full shot. Motion begins immediately "
        "at a gentle nonzero rate, flows fluidly through the middle, and settles through progressively "
        "smaller adjustments into the exact final state, so no individual frame change calls attention "
        "to itself. Every intermediate time contains local changes distributed throughout the frame. "
        "The composition remains one integrated scene, with no frame-spanning dividing line, moving "
        "sheet or band, single propagation front, or side-by-side display of the endpoint states. "
        "Mapped objects remain opaque, solid, sharply resolved, and continuously identifiable. "
        "Where supported by the endpoints, background planes and broad color, shadow, or texture "
        "fields blend, flow, drift, spread, and reshape through overlapping opaque intermediate "
        "states. The support surface, illumination, object density, and negative space evolve "
        "together until the exact <Picture 2> composition is reached.\n\n"
        "overall_soundscape: N/A\n"
        "non_diegetic_music: N/A"
    )


def validate_h3_prompt_byte_budget(
    prompt: str,
    *,
    max_utf8_bytes: int,
    model_context_tokens: int,
    reserved_condition_tokens: int = 0,
) -> int:
    """Enforce a dependency-free upper bound on H3 prompt tokens.

    H3's Qwen tokenizer uses byte-level BPE, so the number of encoded text
    tokens cannot exceed the number of UTF-8 bytes in the prompt. Limiting
    bytes therefore gives a conservative token bound without importing the
    heavyweight ComfyUI/Transformers stack. The reserved budget accounts for
    the two visual-conditioning blocks and their labels.
    """

    prompt_text = str(prompt)
    if not prompt_text.strip():
        raise ValueError("H3 prompt cannot be blank")
    if max_utf8_bytes <= 0 or model_context_tokens <= 0:
        raise ValueError("H3 prompt budgets must be positive")
    if reserved_condition_tokens < 0:
        raise ValueError("reserved H3 conditioning tokens cannot be negative")
    if max_utf8_bytes + reserved_condition_tokens > model_context_tokens:
        raise ValueError("H3 worst-case prompt budget exceeds the model context")

    byte_count = len(prompt_text.encode("utf-8"))
    if byte_count > max_utf8_bytes:
        raise ValueError(
            f"H3 prompt has {byte_count} UTF-8 bytes; operational maximum is "
            f"{max_utf8_bytes}"
        )
    if byte_count + reserved_condition_tokens > model_context_tokens:
        raise ValueError(
            f"H3 prompt plus reserved conditioning could need up to "
            f"{byte_count + reserved_condition_tokens} tokens, above the "
            f"{model_context_tokens}-token model context"
        )
    return byte_count


def _unique_h3_node(nodes: Sequence[Any], node_type: str) -> dict[str, Any]:
    matches = [node for node in nodes if isinstance(node, dict) and node.get("type") == node_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one internal {node_type} node, found {len(matches)}")
    return matches[0]


def _h3_node_widgets(node: Mapping[str, Any], minimum: int) -> list[Any]:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) < minimum:
        raise ValueError(f"official H3 {node.get('type', '<unknown>')} widget interface changed")
    return widgets


def h3_ui_workflow_controls(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """Read the executable controls from inside the official H3 subgraph."""

    subgraphs = workflow.get("definitions", {}).get("subgraphs")
    if not isinstance(subgraphs, list) or len(subgraphs) != 1 or not isinstance(subgraphs[0], dict):
        raise ValueError("expected exactly one H3 subgraph definition")
    internal_nodes = subgraphs[0].get("nodes")
    if not isinstance(internal_nodes, list):
        raise ValueError("official H3 subgraph has no internal nodes")

    h3_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "MiniMaxH3ImageToVideo"), 4)
    duration_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "PrimitiveFloat"), 1)
    noise_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "RandomNoise"), 2)
    diffusion_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "UNETLoader"), 1)
    text_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "CLIPLoader"), 1)
    vae_nodes = [
        node for node in internal_nodes if isinstance(node, dict) and node.get("type") == "VAELoader"
    ]
    if len(vae_nodes) != 2:
        raise ValueError(f"expected two internal VAELoader nodes, found {len(vae_nodes)}")
    vae_names = [_h3_node_widgets(node, 1)[0] for node in vae_nodes]
    audio_names = [name for name in vae_names if "audio" in str(name).lower()]
    video_names = [name for name in vae_names if "audio" not in str(name).lower()]
    if len(audio_names) != 1 or len(video_names) != 1:
        raise ValueError("could not distinguish the internal H3 video and audio VAEs")
    return {
        "prompt": h3_widgets[0],
        "width": h3_widgets[1],
        "height": h3_widgets[2],
        "frame_count_fallback": h3_widgets[3],
        "duration_seconds": duration_widgets[0],
        "seed": noise_widgets[0],
        "seed_control": noise_widgets[1],
        "diffusion_model": diffusion_widgets[0],
        "text_encoder": text_widgets[0],
        "video_vae": video_names[0],
        "audio_vae": audio_names[0],
    }


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
    prompt = strip_h3_source_only_tokens(prompt)
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
    subgraph = subgraphs[0]
    subgraph_id = subgraph.get("id")
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

    # comfy-cli expands the subgraph and executes these internal widget values. Updating only
    # the outer instance leaves the official vaporwave demo prompt, 1344x768 canvas, and 2 s
    # duration active. Patch both layers, then audit the executable layer before submission.
    internal_nodes = subgraph.get("nodes")
    if not isinstance(internal_nodes, list):
        raise ValueError("official H3 subgraph has no internal nodes")
    h3_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "MiniMaxH3ImageToVideo"), 4)
    h3_widgets[:4] = [prompt, int(width), int(height), snap_h3_frame_count(duration_seconds)]
    duration_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "PrimitiveFloat"), 1)
    duration_widgets[0] = float(duration_seconds)
    noise_widgets = _h3_node_widgets(_unique_h3_node(internal_nodes, "RandomNoise"), 2)
    noise_widgets[:2] = [int(seed), "fixed"]
    _h3_node_widgets(_unique_h3_node(internal_nodes, "UNETLoader"), 1)[0] = diffusion_model
    _h3_node_widgets(_unique_h3_node(internal_nodes, "CLIPLoader"), 1)[0] = text_encoder
    vae_nodes = [
        node for node in internal_nodes if isinstance(node, dict) and node.get("type") == "VAELoader"
    ]
    if len(vae_nodes) != 2:
        raise ValueError(f"expected two internal VAELoader nodes, found {len(vae_nodes)}")
    audio_nodes = [
        node for node in vae_nodes if "audio" in str(_h3_node_widgets(node, 1)[0]).lower()
    ]
    video_nodes = [node for node in vae_nodes if node not in audio_nodes]
    if len(audio_nodes) != 1 or len(video_nodes) != 1:
        raise ValueError("could not distinguish the internal H3 video and audio VAEs")
    _h3_node_widgets(video_nodes[0], 1)[0] = video_vae
    _h3_node_widgets(audio_nodes[0], 1)[0] = audio_vae

    save_nodes = [node for node in nodes if isinstance(node, dict) and node.get("type") == "SaveVideo"]
    if len(save_nodes) != 1:
        raise ValueError("expected one SaveVideo node in the official H3 template")
    save_widgets = save_nodes[0].get("widgets_values")
    if not isinstance(save_widgets, list) or not save_widgets:
        raise ValueError("official H3 SaveVideo interface changed")
    save_widgets[0] = output_prefix

    expected_controls = {
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "frame_count_fallback": snap_h3_frame_count(duration_seconds),
        "duration_seconds": float(duration_seconds),
        "seed": int(seed),
        "seed_control": "fixed",
        "diffusion_model": diffusion_model,
        "text_encoder": text_encoder,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
    }
    actual_controls = h3_ui_workflow_controls(workflow)
    if actual_controls != expected_controls:
        raise RuntimeError(
            "H3 executable subgraph controls do not match the requested job: "
            f"expected={expected_controls!r}, actual={actual_controls!r}"
        )
    return workflow


def stable_h3_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable cache fingerprint for a JSON-compatible job payload."""

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
