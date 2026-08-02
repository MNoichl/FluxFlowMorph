"""Prepare one cyclic round of SAGE structural guides for FLUX rendering.

This is a small adapter around the authors' pinned SAGE runtime.  The paper
expects short input clips and obtains endpoint motion from SEA-RAFT.  Our art
workflow starts with still paintings, so there is no honest optical flow to
extract.  We retain SAGE's GlueStick foreground-line matching, canonical
normalization, Hungarian assignment, global cubic trajectory, local line
interpolation, and rasterized frame-wise conditions, but expose a deterministic
"synthetic flow" bend for the otherwise underdetermined still-image case. The
notebook consumes these guides with FLUX.2 Klein and its project LoRA; this
helper deliberately contains no generative model backend.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sage-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gluestick-checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--generated-frames", type=int, default=13)
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--max-lines", type=int, default=200)
    parser.add_argument("--max-matched-lines", type=int, default=160)
    parser.add_argument("--minimum-matched-lines", type=int, default=8)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--trajectory-bend", type=float, default=0.04)
    parser.add_argument("--synthetic-flow-scale", type=float, default=0.16)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (args.sage_repo / "models" / "gluestick").is_dir():
        raise FileNotFoundError(f"Not a SAGE checkout: {args.sage_repo}")
    superpoint_checkpoint = (
        args.sage_repo / "models" / "resources" / "weights" / "superpoint_v1.pth"
    )
    if not superpoint_checkpoint.is_file():
        raise FileNotFoundError(
            "SAGE's vendored GlueStick requires the separate SuperPoint weights at "
            f"{superpoint_checkpoint}. Rerun notebook section 9 to download and verify them."
        )
    for path in (args.manifest, args.gluestick_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.width % 64 or args.height % 64:
        raise ValueError("SAGE width and height must be divisible by 64")
    if args.generated_frames < 5:
        raise ValueError("At least five generated frames are required")
    if args.minimum_matched_lines < 1:
        raise ValueError("minimum-matched-lines must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def fit_rgb(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.fit(
            opened.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def fit_mask(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as opened:
        resized = ImageOps.fit(
            opened.convert("L"),
            (width, height),
            method=Image.Resampling.NEAREST,
            centering=(0.5, 0.5),
        )
    mask = np.asarray(resized, dtype=np.uint8) >= 128
    coverage = float(mask.mean())
    if not 0.001 < coverage <= 1.0:
        raise ValueError(f"Mask {path} has unusable coverage {coverage:.4f}")
    return mask


def mask_box(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Foreground mask is empty")
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    width = max(2.0, max_x - min_x + 1.0)
    height = max(2.0, max_y - min_y + 1.0)
    return Box(
        cx=(min_x + max_x) / 2.0,
        cy=(min_y + max_y) / 2.0,
        width=width,
        height=height,
    )


def points_in_mask(lines: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paper-faithful foreground selection: retain lines intersecting mask."""
    if not len(lines):
        return np.zeros((0,), dtype=bool)
    fractions = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    start = lines[:, 0][:, None, :]
    end = lines[:, 1][:, None, :]
    sampled = start * (1.0 - fractions[None, :, None]) + end * fractions[None, :, None]
    xs = np.clip(np.rint(sampled[..., 0]).astype(np.int64), 0, mask.shape[1] - 1)
    ys = np.clip(np.rint(sampled[..., 1]).astype(np.int64), 0, mask.shape[0] - 1)
    return mask[ys, xs].any(axis=1)


def normalize_lines(lines: np.ndarray, box: Box) -> np.ndarray:
    normalized = lines.astype(np.float64, copy=True)
    normalized[..., 0] = (normalized[..., 0] - box.cx) / (box.width / 2.0)
    normalized[..., 1] = (normalized[..., 1] - box.cy) / (box.height / 2.0)
    return normalized


def choose_matches(
    lines_a: np.ndarray,
    lines_b: np.ndarray,
    box_a: Box,
    box_b: Box,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm_a = normalize_lines(lines_a, box_a)
    norm_b = normalize_lines(lines_b, box_b)
    centers_a = norm_a.mean(axis=1)
    centers_b = norm_b.mean(axis=1)
    costs = ((centers_a[:, None, :] - centers_b[None, :, :]) ** 2).sum(axis=2)
    index_a, index_b = linear_sum_assignment(costs)
    if len(index_a) > maximum:
        order = np.argsort(costs[index_a, index_b], kind="stable")[:maximum]
        index_a, index_b = index_a[order], index_b[order]
    matched_a = lines_a[index_a].astype(np.float64)
    matched_b = lines_b[index_b].astype(np.float64)
    matched_norm_a = norm_a[index_a]
    matched_norm_b = norm_b[index_b]

    # Segment endpoint order is arbitrary. Pick the orientation that rotates
    # and translates the least in canonical space.
    direct = ((matched_norm_a - matched_norm_b) ** 2).sum(axis=(1, 2))
    reversed_cost = ((matched_norm_a - matched_norm_b[:, ::-1]) ** 2).sum(axis=(1, 2))
    reverse = reversed_cost < direct
    matched_b[reverse] = matched_b[reverse, ::-1]
    matched_norm_b[reverse] = matched_norm_b[reverse, ::-1]
    return matched_a, matched_b, matched_norm_a, matched_norm_b


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return u**3 * p0 + 3.0 * u**2 * t * p1 + 3.0 * u * t**2 * p2 + t**3 * p3


def interpolate_structures(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    box_a: Box,
    box_b: Box,
    frame_count: int,
    synthetic_flow_scale: float,
    bend: float,
    width: int,
    height: int,
) -> list[np.ndarray]:
    center_a = np.array([box_a.cx, box_a.cy], dtype=np.float64)
    center_b = np.array([box_b.cx, box_b.cy], dtype=np.float64)
    delta = center_b - center_a
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        direction = np.array([1.0, 0.0])
    else:
        direction = delta / distance
    perpendicular = np.array([-direction[1], direction[0]])
    diagonal = math.hypot(width, height)
    synthetic_flow = delta * synthetic_flow_scale + perpendicular * bend * diagonal
    control_0 = center_a
    control_1 = center_a + synthetic_flow
    control_2 = center_b - synthetic_flow
    control_3 = center_b

    output: list[np.ndarray] = []
    # The paper defines T *inbetween* structures. Exact endpoints are retained
    # separately by the notebook, so neither endpoint is duplicated here.
    for t in np.linspace(0.0, 1.0, frame_count + 2)[1:-1]:
        center = cubic_bezier(control_0, control_1, control_2, control_3, float(t))
        box_width = (1.0 - t) * box_a.width + t * box_b.width
        box_height = (1.0 - t) * box_a.height + t * box_b.height
        local = (1.0 - t) * norm_a + t * norm_b
        lines = local.copy()
        lines[..., 0] = center[0] + local[..., 0] * box_width / 2.0
        lines[..., 1] = center[1] + local[..., 1] * box_height / 2.0
        output.append(lines)
    return output


def palette(index: int, count: int) -> tuple[int, int, int]:
    hue = (index / max(1, count)) % 1.0
    hsv = np.uint8([[[round(hue * 179), 190, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def rasterize_conditions(
    structures: list[np.ndarray],
    width: int,
    height: int,
    line_width: int,
    output_directory: Path,
) -> list[Image.Image]:
    output_directory.mkdir(parents=True, exist_ok=True)
    count = len(structures[0])
    colors = [palette(index, count) for index in range(count)]
    images: list[Image.Image] = []
    for frame_index, lines in enumerate(structures):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for line_index, line in enumerate(lines):
            coordinates = tuple(float(value) for value in line.reshape(-1))
            draw.line(coordinates, fill=colors[line_index], width=line_width)
        image.save(output_directory / f"condition_{frame_index:04d}.png")
        images.append(image)
    return images


def diagnostic_overlay(
    image: Image.Image,
    lines: np.ndarray,
    output_path: Path,
    line_width: int,
) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for index, line in enumerate(lines):
        draw.line(
            tuple(float(value) for value in line.reshape(-1)),
            fill=palette(index, len(lines)),
            width=max(1, line_width),
        )
    overlay.save(output_path)
    overlay.close()


def load_gluestick(args: argparse.Namespace):
    sys.path.insert(0, str(args.sage_repo))
    from models.gluestick.models.two_view_pipeline import TwoViewPipeline

    configuration = {
        "name": "two_view_pipeline",
        "use_lines": True,
        "extractor": {
            "name": "wireframe",
            "sp_params": {
                "force_num_keypoints": False,
                "max_num_keypoints": args.max_points,
            },
            "wireframe_params": {
                "merge_points": True,
                "merge_line_endpoints": True,
            },
            "max_n_lines": args.max_lines,
        },
        "matcher": {
            "name": "gluestick",
            "weights": str(args.gluestick_checkpoint),
            "trainable": False,
        },
        "ground_truth": {"from_pose_depth": False},
    }
    return TwoViewPipeline(configuration).to("cuda").eval()


@torch.inference_mode()
def detect_lines(model, source: Image.Image, target: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    from models.gluestick import batch_to_np, numpy_image_to_torch

    gray_a = np.asarray(source.convert("L"))
    gray_b = np.asarray(target.convert("L"))
    inputs = {
        "image0": numpy_image_to_torch(gray_a).to("cuda")[None],
        "image1": numpy_image_to_torch(gray_b).to("cuda")[None],
    }
    prediction = batch_to_np(model(inputs))
    return prediction["lines0"], prediction["lines1"]


def load_fcvg(args: argparse.Namespace):
    sys.path.insert(0, str(args.sage_repo))
    from diffusers import AutoencoderKLTemporalDecoder
    from models.controlnext_vid_svd import ControlNeXtSDVModel
    from models.unet_spatio_temporal_condition_controlnext import (
        UNetSpatioTemporalConditionControlNeXtModel,
    )
    from pipeline.pipeline_FCVG import StableVideoDiffusionPipelineControlNeXtReverse
    from transformers import CLIPVisionModelWithProjection

    common = {
        "revision": args.base_model_revision,
        "cache_dir": str(args.hf_cache),
        "torch_dtype": torch.float16,
    }
    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.base_model_id,
        subfolder="unet",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        variant="fp16",
        **common,
    )
    controlnext = ControlNeXtSDVModel()
    controlnext.load_state_dict(load_file(str(args.controlnext_checkpoint)), strict=True)
    unet.load_state_dict(load_file(str(args.unet_checkpoint)), strict=False)
    unet.to(dtype=torch.float16)
    controlnext.to(dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.base_model_id,
        subfolder="image_encoder",
        **common,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.base_model_id,
        subfolder="vae",
        variant="fp16",
        **common,
    )
    pipeline = StableVideoDiffusionPipelineControlNeXtReverse.from_pretrained(
        args.base_model_id,
        controlnext=controlnext,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        variant="fp16",
        **common,
    )
    pipeline.enable_model_cpu_offload()
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    return pipeline


def ffmpeg_sequence(input_directory: Path, output_path: Path, fps: float, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(input_directory / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    anchors = manifest["anchors"]
    if len(anchors) < 3:
        raise ValueError("A cyclic SAGE sequence needs at least three anchors")

    contract = {
        "adapter": "still_synthetic_flow_v1",
        "sage_repository_commit": subprocess.check_output(
            ["git", "-C", str(args.sage_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "base_model_id": args.base_model_id,
        "base_model_revision": args.base_model_revision,
        "gluestick_sha256": sha256_file(args.gluestick_checkpoint),
        "controlnext_sha256": sha256_file(args.controlnext_checkpoint),
        "unet_sha256": sha256_file(args.unet_checkpoint),
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "manifest",
                "output_root",
                "sage_repo",
                "gluestick_checkpoint",
                "controlnext_checkpoint",
                "unet_checkpoint",
                "hf_cache",
                "reuse",
                "phase",
            }
        },
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()

    gluestick = None
    if args.phase != "render":
        print("Loading GlueStick once for every cyclic gap...", flush=True)
        gluestick = load_gluestick(args)
    prepared: list[dict[str, Any]] = []
    for gap_index, left in enumerate(anchors):
        right = anchors[(gap_index + 1) % len(anchors)]
        gap_uid = f"gap_{gap_index:04d}_{left['uid']}_to_{right['uid']}"
        gap_directory = args.output_root / gap_uid
        metadata_path = gap_directory / "metadata.json"
        endpoint_hashes = {
            "left_image": sha256_file(Path(left["path"])),
            "right_image": sha256_file(Path(right["path"])),
            "left_mask": sha256_file(Path(left["mask_path"])),
            "right_mask": sha256_file(Path(right["mask_path"])),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "contract_hash": contract_hash,
                    "gap_uid": gap_uid,
                    "endpoint_hashes": endpoint_hashes,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if (args.reuse or args.phase == "render") and metadata_path.is_file():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            complete_paths = [Path(path) for path in saved.get("complete_frame_paths", [])]
            if args.reuse and saved.get("fingerprint") == fingerprint and complete_paths and all(
                path.is_file() for path in complete_paths
            ):
                prepared.append(saved)
                print(f"Reusing complete {gap_uid}", flush=True)
                continue
            condition_paths = [Path(path) for path in saved.get("condition_paths", [])]
            if (
                args.phase == "render"
                and saved.get("fingerprint") == fingerprint
                and len(condition_paths) == args.generated_frames
                and all(path.is_file() for path in condition_paths)
            ):
                source = fit_rgb(Path(left["path"]), args.width, args.height)
                target = fit_rgb(Path(right["path"]), args.width, args.height)
                conditions = [Image.open(path).convert("RGB") for path in condition_paths]
                saved.update({
                    "_source": source,
                    "_target": target,
                    "_conditions": conditions,
                })
                prepared.append(saved)
                print(f"Reusing prepared structural guides for {gap_uid}", flush=True)
                continue

        if args.phase == "render":
            raise RuntimeError(
                f"{gap_uid} has no matching prepared structural guide. "
                "Run the same command with --phase prepare first."
            )

        gap_directory.mkdir(parents=True, exist_ok=True)
        conditions_directory = gap_directory / "conditions"
        rendered_directory = gap_directory / "rendered"
        complete_directory = gap_directory / "complete"
        for directory in (conditions_directory, rendered_directory, complete_directory):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=False)

        source = fit_rgb(Path(left["path"]), args.width, args.height)
        target = fit_rgb(Path(right["path"]), args.width, args.height)
        mask_a = fit_mask(Path(left["mask_path"]), args.width, args.height)
        mask_b = fit_mask(Path(right["mask_path"]), args.width, args.height)
        lines_a, lines_b = detect_lines(gluestick, source, target)
        box_a, box_b = mask_box(mask_a), mask_box(mask_b)
        selected_a = lines_a[points_in_mask(lines_a, mask_a)]
        selected_b = lines_b[points_in_mask(lines_b, mask_b)]
        if min(len(selected_a), len(selected_b)) < args.minimum_matched_lines:
            raise RuntimeError(
                f"{gap_uid}: only {len(selected_a)} and {len(selected_b)} foreground "
                f"lines survived. Improve the masks, lower --minimum-matched-lines, "
                "or use full-frame masks."
            )
        matched_a, matched_b, norm_a, norm_b = choose_matches(
            selected_a, selected_b, box_a, box_b, args.max_matched_lines
        )
        structures = interpolate_structures(
            norm_a,
            norm_b,
            box_a,
            box_b,
            args.generated_frames,
            args.synthetic_flow_scale,
            args.trajectory_bend,
            args.width,
            args.height,
        )
        conditions = rasterize_conditions(
            structures,
            args.width,
            args.height,
            args.line_width,
            conditions_directory,
        )
        source.save(gap_directory / "source.png")
        target.save(gap_directory / "target.png")
        Image.fromarray(mask_a.astype(np.uint8) * 255).save(gap_directory / "source_mask.png")
        Image.fromarray(mask_b.astype(np.uint8) * 255).save(gap_directory / "target_mask.png")
        diagnostic_overlay(source, matched_a, gap_directory / "source_matched_lines.png", args.line_width)
        diagnostic_overlay(target, matched_b, gap_directory / "target_matched_lines.png", args.line_width)
        prepared_item = {
                "gap_index": gap_index,
                "gap_uid": gap_uid,
                "left": left,
                "right": right,
                "gap_directory": str(gap_directory),
                "conditions_directory": str(conditions_directory),
                "rendered_directory": str(rendered_directory),
                "complete_directory": str(complete_directory),
                "fingerprint": fingerprint,
                "endpoint_hashes": endpoint_hashes,
                "detected_lines": [int(len(lines_a)), int(len(lines_b))],
                "foreground_lines": [int(len(selected_a)), int(len(selected_b))],
                "matched_lines": int(len(matched_a)),
                "boxes": [asdict(box_a), asdict(box_b)],
                "condition_paths": [
                    str(conditions_directory / f"condition_{index:04d}.png")
                    for index in range(args.generated_frames)
                ],
                "complete": False,
                "_source": source,
                "_target": target,
                "_conditions": conditions,
        }
        prepared.append(prepared_item)
        write_json(
            metadata_path,
            {key: value for key, value in prepared_item.items() if not key.startswith("_")},
        )
        print(
            f"Prepared {gap_uid}: {len(matched_a)} matched foreground lines",
            flush=True,
        )

    del gluestick
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    if args.phase == "prepare":
        preparation_manifest = {
            "method": "SAGE",
            "phase": "prepared",
            "contract": contract,
            "contract_hash": contract_hash,
            "gaps": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in prepared
            ],
        }
        write_json(args.output_root / "sage_preparation_manifest.json", preparation_manifest)
        for item in prepared:
            for key in ("_source", "_target"):
                image = item.get(key)
                if image is not None:
                    image.close()
            for condition in item.get("_conditions", []):
                condition.close()
        print(json.dumps({
            "prepared": True,
            "gaps": len(prepared),
            "preparation_manifest": str(args.output_root / "sage_preparation_manifest.json"),
        }, indent=2), flush=True)
        return

    print("Loading FCVG once for every cyclic gap...", flush=True)
    fcvg = load_fcvg(args)
    completed: list[dict[str, Any]] = []
    for item in prepared:
        if item.get("complete"):
            completed.append(item)
            continue
        source = item.pop("_source")
        target = item.pop("_target")
        conditions = item.pop("_conditions")
        gap_directory = Path(item["gap_directory"])
        rendered_directory = Path(item["rendered_directory"])
        complete_directory = Path(item["complete_directory"])
        print(
            f"Rendering SAGE gap {item['gap_index'] + 1}/{len(anchors)}: "
            f"{item['left']['uid']} -> {item['right']['uid']}",
            flush=True,
        )
        generator = torch.Generator(device="cuda").manual_seed(
            args.seed + int(item["gap_index"])
        )
        output = fcvg(
            source,
            target,
            conditions,
            decode_chunk_size=args.decode_chunk_size,
            num_frames=args.generated_frames,
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=args.noise_aug_strength,
            fps=7,
            control_weight=args.control_weight,
            width=args.width,
            height=args.height,
            min_guidance_scale=args.min_guidance,
            max_guidance_scale=args.max_guidance,
            frames_per_batch=args.frames_per_batch,
            num_inference_steps=args.inference_steps,
            overlap=args.overlap,
            generator=generator,
        ).frames
        generated = [frame for group in output for frame in group]
        if len(generated) != args.generated_frames:
            raise RuntimeError(
                f"FCVG returned {len(generated)} frames, expected {args.generated_frames}"
            )
        rendered_paths = []
        for frame_index, frame in enumerate(generated):
            path = rendered_directory / f"sage_{frame_index:04d}.png"
            frame.convert("RGB").save(path)
            rendered_paths.append(str(path))

        complete_images = [source, *[frame.convert("RGB") for frame in generated], target]
        complete_paths = []
        for frame_index, frame in enumerate(complete_images):
            path = complete_directory / f"frame_{frame_index:04d}.png"
            frame.save(path)
            complete_paths.append(str(path))
        clip_path = gap_directory / "transition.mp4"
        ffmpeg_sequence(complete_directory, clip_path, args.fps, args.crf)
        item.update(
            {
                "rendered_frame_paths": rendered_paths,
                "complete_frame_paths": complete_paths,
                "clip_path": str(clip_path),
                "generated_frames": args.generated_frames,
                "complete": True,
            }
        )
        write_json(gap_directory / "metadata.json", item)
        completed.append(item)
        for frame in complete_images[1:-1]:
            frame.close()
        source.close()
        target.close()
        for condition in conditions:
            condition.close()
        print(f"Saved {clip_path}", flush=True)

    del fcvg
    gc.collect()
    torch.cuda.empty_cache()

    sequence_directory = args.output_root / "cyclic_frames"
    if sequence_directory.exists():
        shutil.rmtree(sequence_directory)
    sequence_directory.mkdir(parents=True, exist_ok=False)
    sequence_paths = []
    output_index = 0
    for item in completed:
        # Exact source anchor + every generated SAGE inbetween. The exact target
        # is the next gap's source, so omitting it avoids duplicate anchors.
        paths = [Path(path) for path in item["complete_frame_paths"]]
        for source_path in paths[:-1]:
            output_path = sequence_directory / f"frame_{output_index:04d}.png"
            shutil.copy2(source_path, output_path)
            sequence_paths.append(str(output_path))
            output_index += 1
    final_video_path = args.output_root / "sage_cyclic_one_round.mp4"
    ffmpeg_sequence(sequence_directory, final_video_path, args.fps, args.crf)
    final_manifest = {
        "method": "SAGE",
        "paper": "https://arxiv.org/abs/2510.24667v2",
        "cyclic": True,
        "still_image_adaptation": True,
        "still_motion_fallback": "deterministic synthetic cubic control vector",
        "contract": contract,
        "contract_hash": contract_hash,
        "gaps": completed,
        "sequence_frame_paths": sequence_paths,
        "final_video_path": str(final_video_path),
        "frame_count": len(sequence_paths),
        "fps": args.fps,
    }
    write_json(args.output_root / "sage_sequence_manifest.json", final_manifest)
    print(json.dumps({
        "complete": True,
        "gaps": len(completed),
        "sequence_frames": len(sequence_paths),
        "final_video": str(final_video_path),
    }, indent=2), flush=True)


def prepare_structure_main() -> None:
    """Generate SAGE conditions only; FLUX rendering stays in the notebook."""

    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    anchors = manifest["anchors"]
    if len(anchors) < 3:
        raise ValueError("A cyclic SAGE sequence needs at least three anchors")

    contract = {
        "adapter": "sage_structure_for_flux2_klein_v2_dense_warp",
        "renderer": "external_flux2_klein_with_project_lora",
        "sage_repository_commit": subprocess.check_output(
            ["git", "-C", str(args.sage_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "gluestick_sha256": sha256_file(args.gluestick_checkpoint),
        "settings": {
            "width": args.width,
            "height": args.height,
            "generated_frames": args.generated_frames,
            "max_points": args.max_points,
            "max_lines": args.max_lines,
            "max_matched_lines": args.max_matched_lines,
            "minimum_matched_lines": args.minimum_matched_lines,
            "line_width": args.line_width,
            "trajectory_bend": args.trajectory_bend,
            "synthetic_flow_scale": args.synthetic_flow_scale,
        },
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()

    print("Loading GlueStick once for every cyclic gap...", flush=True)
    gluestick = load_gluestick(args)
    prepared: list[dict[str, Any]] = []
    for gap_index, left in enumerate(anchors):
        right = anchors[(gap_index + 1) % len(anchors)]
        gap_uid = f"gap_{gap_index:04d}_{left['uid']}_to_{right['uid']}"
        gap_directory = args.output_root / gap_uid
        metadata_path = gap_directory / "structure_metadata.json"
        endpoint_hashes = {
            "left_image": sha256_file(Path(left["path"])),
            "right_image": sha256_file(Path(right["path"])),
            "left_mask": sha256_file(Path(left["mask_path"])),
            "right_mask": sha256_file(Path(right["mask_path"])),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "contract_hash": contract_hash,
                    "gap_uid": gap_uid,
                    "endpoint_hashes": endpoint_hashes,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if args.reuse and metadata_path.is_file():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            condition_paths = [Path(path) for path in saved.get("condition_paths", [])]
            structure_data_path = Path(saved.get("structure_data_path", ""))
            if (
                saved.get("fingerprint") == fingerprint
                and len(condition_paths) == args.generated_frames
                and all(path.is_file() for path in condition_paths)
                and structure_data_path.is_file()
            ):
                prepared.append(saved)
                print(f"Reusing SAGE structures for {gap_uid}", flush=True)
                continue

        gap_directory.mkdir(parents=True, exist_ok=True)
        conditions_directory = gap_directory / "conditions"
        if conditions_directory.exists():
            shutil.rmtree(conditions_directory)
        conditions_directory.mkdir(parents=True, exist_ok=False)

        source = fit_rgb(Path(left["path"]), args.width, args.height)
        target = fit_rgb(Path(right["path"]), args.width, args.height)
        mask_a = fit_mask(Path(left["mask_path"]), args.width, args.height)
        mask_b = fit_mask(Path(right["mask_path"]), args.width, args.height)
        lines_a, lines_b = detect_lines(gluestick, source, target)
        box_a, box_b = mask_box(mask_a), mask_box(mask_b)
        selected_a = lines_a[points_in_mask(lines_a, mask_a)]
        selected_b = lines_b[points_in_mask(lines_b, mask_b)]
        if min(len(selected_a), len(selected_b)) < args.minimum_matched_lines:
            raise RuntimeError(
                f"{gap_uid}: only {len(selected_a)} and {len(selected_b)} foreground "
                f"lines survived. Improve the masks, lower --minimum-matched-lines, "
                "or use full-frame masks."
            )
        matched_a, matched_b, norm_a, norm_b = choose_matches(
            selected_a, selected_b, box_a, box_b, args.max_matched_lines
        )
        structures = interpolate_structures(
            norm_a,
            norm_b,
            box_a,
            box_b,
            args.generated_frames,
            args.synthetic_flow_scale,
            args.trajectory_bend,
            args.width,
            args.height,
        )
        conditions = rasterize_conditions(
            structures,
            args.width,
            args.height,
            args.line_width,
            conditions_directory,
        )
        structure_data_path = gap_directory / "sage_structure_data.npz"
        np.savez_compressed(
            structure_data_path,
            matched_source_lines=matched_a.astype(np.float32),
            matched_target_lines=matched_b.astype(np.float32),
            matched_source_normalized=norm_a.astype(np.float32),
            matched_target_normalized=norm_b.astype(np.float32),
            intermediate_lines=np.stack(structures).astype(np.float32),
            condition_alphas=np.linspace(
                0.0, 1.0, args.generated_frames + 2, dtype=np.float32
            )[1:-1],
        )
        source.save(gap_directory / "source.png")
        target.save(gap_directory / "target.png")
        Image.fromarray(mask_a.astype(np.uint8) * 255).save(gap_directory / "source_mask.png")
        Image.fromarray(mask_b.astype(np.uint8) * 255).save(gap_directory / "target_mask.png")
        diagnostic_overlay(source, matched_a, gap_directory / "source_matched_lines.png", args.line_width)
        diagnostic_overlay(target, matched_b, gap_directory / "target_matched_lines.png", args.line_width)
        item = {
            "gap_index": gap_index,
            "gap_uid": gap_uid,
            "left": left,
            "right": right,
            "gap_directory": str(gap_directory),
            "conditions_directory": str(conditions_directory),
            "fingerprint": fingerprint,
            "endpoint_hashes": endpoint_hashes,
            "detected_lines": [int(len(lines_a)), int(len(lines_b))],
            "foreground_lines": [int(len(selected_a)), int(len(selected_b))],
            "matched_lines": int(len(matched_a)),
            "boxes": [asdict(box_a), asdict(box_b)],
            "structure_data_path": str(structure_data_path),
            "structure_data_sha256": sha256_file(structure_data_path),
            "condition_alphas": [
                float(value)
                for value in np.linspace(0.0, 1.0, args.generated_frames + 2)[1:-1]
            ],
            "condition_paths": [
                str(conditions_directory / f"condition_{index:04d}.png")
                for index in range(args.generated_frames)
            ],
        }
        write_json(metadata_path, item)
        prepared.append(item)
        source.close()
        target.close()
        for condition in conditions:
            condition.close()
        print(f"Prepared {gap_uid}: {len(matched_a)} matched foreground lines", flush=True)

    del gluestick
    gc.collect()
    torch.cuda.empty_cache()
    preparation_manifest = {
        "method": "SAGE structural guidance",
        "phase": "prepared_for_flux2_klein",
        "contract": contract,
        "contract_hash": contract_hash,
        "gaps": prepared,
    }
    output_path = args.output_root / "sage_preparation_manifest.json"
    write_json(output_path, preparation_manifest)
    print(json.dumps({
        "prepared": True,
        "gaps": len(prepared),
        "preparation_manifest": str(output_path),
        "generative_backend_loaded": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    prepare_structure_main()
