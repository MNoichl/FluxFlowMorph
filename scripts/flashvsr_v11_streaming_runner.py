#!/usr/bin/env python3
"""Memory-bounded FlashVSR Stable/Sparse-Sage inference for frame sequences.

This adapter uses ComfyUI-FlashVSR-Stable's bundled Triton Sparse Sage backend,
while preserving the repository's v1.1 tiny-long inference structure. It lazily
loads 4x conditioning frames, writes decoded frames to FFmpeg immediately, warms
the causal stream with frames from the end of the cycle, and trims that warm-up
from the encoded result. It never imports or compiles Block-Sparse Attention or
Flash Attention.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def compute_target_dimensions(
    width: int,
    height: int,
    scale: float,
    multiple: int = 128,
) -> tuple[int, int]:
    """Match FlashVSR's bicubic-upscale then center-crop dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("Input dimensions must be positive")
    if scale <= 0:
        raise ValueError("Scale must be positive")
    target_width = int(round(width * scale)) // multiple * multiple
    target_height = int(round(height * scale)) // multiple * multiple
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Scaled dimensions are smaller than FlashVSR's 128-pixel grid")
    return target_width, target_height


@dataclass(frozen=True)
class CyclicStreamPlan:
    source_indices: tuple[int, ...]
    pipeline_frames: int
    pipeline_output_frames: int
    trim_start: int
    trim_count: int


def build_cyclic_stream_plan(
    frame_count: int,
    warmup_frames: int = 16,
    lookahead_frames: int = 4,
) -> CyclicStreamPlan:
    """Build an 8n+1 FlashVSR stream that emits exactly one duplicate-free cycle."""
    if frame_count < 1:
        raise ValueError("At least one input frame is required")
    if warmup_frames < 0 or lookahead_frames < 4:
        raise ValueError("Warm-up must be non-negative and lookahead at least four frames")
    required = warmup_frames + frame_count + lookahead_frames
    pipeline_frames = math.ceil((required - 1) / 8) * 8 + 1
    pipeline_output_frames = pipeline_frames - 4
    if warmup_frames + frame_count > pipeline_output_frames:
        raise AssertionError("Cyclic stream plan cannot contain the requested output")
    indices = tuple((position - warmup_frames) % frame_count for position in range(pipeline_frames))
    return CyclicStreamPlan(
        source_indices=indices,
        pipeline_frames=pipeline_frames,
        pipeline_output_frames=pipeline_output_frames,
        trim_start=warmup_frames,
        trim_count=frame_count,
    )


def load_manifest(path: Path) -> tuple[list[Path], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = [Path(value).expanduser() for value in payload.get("frames", [])]
    fps = float(payload.get("fps", 0))
    if not frames or fps <= 0:
        raise ValueError("Input manifest must contain non-empty 'frames' and positive 'fps'")
    missing = [frame for frame in frames if not frame.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} input frames; first: {missing[0]}")
    return frames, fps


class LazyCyclicFrames:
    """Load only the temporal slices currently requested by the streaming model."""

    def __init__(self, paths: Sequence[Path], plan: CyclicStreamPlan, scale: float):
        from PIL import Image

        self.paths = tuple(paths)
        self.plan = plan
        self.scale = float(scale)
        with Image.open(self.paths[0]) as image:
            self.input_width, self.input_height = image.size
        self.width, self.height = compute_target_dimensions(
            self.input_width, self.input_height, self.scale
        )

    def _load(self, source_index: int):
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(self.paths[source_index]) as opened:
            image = opened.convert("RGB")
            scaled_width = int(round(image.width * self.scale))
            scaled_height = int(round(image.height * self.scale))
            image = image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)
            left = (scaled_width - self.width) // 2
            top = (scaled_height - self.height) // 2
            image = image.crop((left, top, left + self.width, top + self.height))
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32)
        return (tensor.div_(127.5).sub_(1.0)).to(torch.bfloat16)

    def temporal_slice(self, start: int, stop: int):
        import torch

        start = max(0, int(start))
        stop = min(int(stop), self.plan.pipeline_frames)
        if stop <= start:
            return torch.empty(
                (1, 3, 0, self.height, self.width), dtype=torch.bfloat16, device="cpu"
            )
        frames = [self._load(self.plan.source_indices[index]) for index in range(start, stop)]
        return torch.stack(frames, dim=1).unsqueeze(0)


class RawVideoWriter:
    def __init__(
        self,
        ffmpeg: str,
        output: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.partial = output.with_name(output.stem + ".partial" + output.suffix)
        self.partial.unlink(missing_ok=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.partial),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin was not created")
        self.count = 0

    def append_tensor(self, frame) -> None:
        import torch

        array = (
            frame.float()
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        assert self.process.stdin is not None
        self.process.stdin.write(array.tobytes())
        self.count += 1

    def close(self, expected_frames: int) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg exited with code {return_code}")
        if self.count != expected_frames:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(f"Encoded {self.count} frames; expected {expected_frames}")
        self.partial.replace(self.output)

    def abort(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.partial.unlink(missing_ok=True)


def initialize_pipeline(repository: Path, weights: Path):
    import torch

    sys.path.insert(0, str(repository))
    from src.models import ModelManager, wan_video_dit
    from src.models.TCDecoder import build_tcdecoder
    from src.models.sparse_sage.core import sparse_sageattn  # noqa: F401
    from src.models.utils import Causal_LQ4x_Proj
    from src.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline

    # The fork exposes an ATTENTION_MODE label but selects its actual sparse
    # implementation through USE_BLOCK_ATTN. Set both explicitly so an installed
    # Block-Sparse package can never change this run's backend.
    wan_video_dit.ATTENTION_MODE = "sparse_sage_attention"
    wan_video_dit.USE_BLOCK_ATTN = False
    print(
        json.dumps(
            {
                "flashvsr_implementation": "ComfyUI-FlashVSR-Stable",
                "attention_backend": wan_video_dit.ATTENTION_MODE,
                "block_sparse_enabled": wan_video_dit.USE_BLOCK_ATTN,
                "custom_cuda_extension_compiled": False,
            },
            indent=2,
        ),
        flush=True,
    )

    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([str(weights / "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(manager, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1
    ).to("cuda", dtype=torch.bfloat16)
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(weights / "LQ_proj_in.ckpt", map_location="cpu"), strict=True
    )
    pipe.denoising_model().LQ_proj_in.to("cuda")
    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        device="cuda",
        dtype=torch.bfloat16,
        new_latent_channels=784,
    )
    missing = pipe.TCDecoder.load_state_dict(
        torch.load(weights / "TCDecoder.ckpt", map_location="cpu"), strict=False
    )
    print("TCDecoder state:", missing, flush=True)
    pipe.to("cuda", dtype=torch.bfloat16)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    context = torch.load(
        repository / "posi_prompt.pth",
        map_location="cpu",
    )
    pipe.init_cross_kv(context_tensor=context)
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def run_streaming(
    pipe,
    source: LazyCyclicFrames,
    plan: CyclicStreamPlan,
    writer: RawVideoWriter,
    *,
    seed: int,
    sparse_ratio: float,
    local_range: int,
    color_fix: bool,
) -> None:
    """Stable-fork v1.1 tiny-long loop with lazy input and immediate encoding."""
    import torch
    from src.pipelines.flashvsr_tiny_long import model_fn_wan_video
    from tqdm.auto import tqdm

    frame_count = plan.pipeline_frames
    process_count = (frame_count - 1) // 8 - 2
    if process_count < 1:
        raise ValueError("FlashVSR tiny-long needs a longer input sequence")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe.denoising_model().LQ_proj_in.clear_cache()
    pipe.TCDecoder.clean_mem()
    previous_input_index = 0
    emitted = 0
    color_warning_printed = False
    pre_cache_k = None
    pre_cache_v = None

    for process_index in tqdm(range(process_count), desc="FlashVSR stream blocks"):
        if process_index == 0:
            pre_cache_k = [None] * len(pipe.dit.blocks)
            pre_cache_v = [None] * len(pipe.dit.blocks)
            lq_latents = None
            for inner_index in range(7):
                start = max(0, inner_index * 4 - 3)
                stop = (inner_index + 1) * 4 - 3
                conditioning = source.temporal_slice(start, stop).to("cuda")
                current = pipe.denoising_model().LQ_proj_in.stream_forward(conditioning)
                del conditioning
                if current is None:
                    continue
                if lq_latents is None:
                    lq_latents = current
                else:
                    for layer_index in range(len(lq_latents)):
                        lq_latents[layer_index] = torch.cat(
                            [lq_latents[layer_index], current[layer_index]], dim=1
                        )
            current_input_index = 21
            latent_frames = 6
        else:
            lq_latents = None
            for inner_index in range(2):
                start = process_index * 8 + 17 + inner_index * 4
                stop = process_index * 8 + 21 + inner_index * 4
                conditioning = source.temporal_slice(start, stop).to("cuda")
                current = pipe.denoising_model().LQ_proj_in.stream_forward(conditioning)
                del conditioning
                if current is None:
                    continue
                if lq_latents is None:
                    lq_latents = current
                else:
                    for layer_index in range(len(lq_latents)):
                        lq_latents[layer_index] = torch.cat(
                            [lq_latents[layer_index], current[layer_index]], dim=1
                        )
            current_input_index = process_index * 8 + 21
            latent_frames = 2

        del current
        current_latents = torch.randn(
            (1, 16, latent_frames, source.height // 8, source.width // 8),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        noise_prediction, pre_cache_k, pre_cache_v = model_fn_wan_video(
            pipe.dit,
            x=current_latents,
            timestep=pipe.timestep,
            context=None,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=lq_latents,
            is_full_block=False,
            is_stream=True,
            pre_cache_k=pre_cache_k,
            pre_cache_v=pre_cache_v,
            topk_ratio=sparse_ratio * 768 * 1280 / (source.height * source.width),
            kv_ratio=3.0,
            cur_process_idx=process_index,
            t_mod=pipe.t_mod,
            t=pipe.t,
            local_range=local_range,
        )
        current_latents = current_latents - noise_prediction
        lq_frames = source.temporal_slice(previous_input_index, current_input_index).to("cuda")
        decoded = pipe.TCDecoder.decode_video(
            current_latents.transpose(1, 2),
            parallel=False,
            show_progress_bar=False,
            cond=lq_frames,
        ).transpose(1, 2).mul_(2).sub_(1)
        if color_fix:
            try:
                decoded = pipe.ColorCorrector(
                    decoded,
                    lq_frames,
                    clip_range=(-1, 1),
                    chunk_size=None,
                    method="adain",
                )
            except Exception as error:  # Match official fallback, but make it visible.
                if not color_warning_printed:
                    print(f"FlashVSR color-fix warning (continuing): {error}", flush=True)
                    color_warning_printed = True

        for temporal_index in range(decoded.shape[2]):
            output_index = previous_input_index + temporal_index
            if plan.trim_start <= output_index < plan.trim_start + plan.trim_count:
                writer.append_tensor(decoded[0, :, temporal_index])
                emitted += 1
        previous_input_index = current_input_index
        del lq_latents, current_latents, noise_prediction, lq_frames, decoded
        if emitted >= plan.trim_count:
            break

    if emitted != plan.trim_count:
        raise RuntimeError(f"FlashVSR emitted {emitted} selected frames; expected {plan.trim_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("sparse_sage_attention",),
        default="sparse_sage_attention",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-ratio", type=float, default=2.0)
    parser.add_argument("--local-range", type=int, choices=(9, 11), default=11)
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--no-color-fix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attention_backend != "sparse_sage_attention":
        raise ValueError("This runner intentionally supports only bundled Sparse Sage")
    started = time.time()
    for filename in (
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
        "diffusion_pytorch_model_streaming_dmd.safetensors",
    ):
        if not (args.weights / filename).is_file():
            raise FileNotFoundError(args.weights / filename)
    paths, fps = load_manifest(args.manifest)
    plan = build_cyclic_stream_plan(len(paths), warmup_frames=args.warmup_frames)
    source = LazyCyclicFrames(paths, plan, args.scale)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FlashVSR requires CUDA")
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "input_frames": len(paths),
                "pipeline_frames": plan.pipeline_frames,
                "cyclic_warmup_frames": plan.trim_start,
                "input_resolution": [source.input_width, source.input_height],
                "output_resolution": [source.width, source.height],
                "fps": fps,
            },
            indent=2,
        ),
        flush=True,
    )
    pipe = initialize_pipeline(args.repo.resolve(), args.weights.resolve())
    writer = RawVideoWriter(
        args.ffmpeg,
        args.output,
        source.width,
        source.height,
        fps,
        args.crf,
        args.preset,
    )
    try:
        run_streaming(
            pipe,
            source,
            plan,
            writer,
            seed=args.seed,
            sparse_ratio=args.sparse_ratio,
            local_range=args.local_range,
            color_fix=not args.no_color_fix,
        )
        writer.close(plan.trim_count)
    except BaseException:
        writer.abort()
        raise

    report = {
        "complete": True,
        "method": (
            "ComfyUI-FlashVSR-Stable v1.1 tiny-long with bundled Triton Sparse Sage, "
            "lazy input, and streamed H.264 output"
        ),
        "attention_backend": args.attention_backend,
        "custom_cuda_extension_compiled": False,
        "cyclic": True,
        "input_unique_frames": len(paths),
        "output_unique_frames": writer.count,
        "pipeline_frames_with_context": plan.pipeline_frames,
        "cyclic_warmup_frames": plan.trim_start,
        "fps": fps,
        "duration_seconds": writer.count / fps,
        "scale_requested": args.scale,
        "input_resolution": [source.input_width, source.input_height],
        "output_resolution": [source.width, source.height],
        "sparse_ratio": args.sparse_ratio,
        "local_range": args.local_range,
        "color_fix": not args.no_color_fix,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "video": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
