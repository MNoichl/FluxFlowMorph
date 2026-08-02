"""Interpolate an explicitly closed PNG sequence with pinned Practical-RIFE."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multi", type=int, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RIFE")
    if args.multi < 2 or args.batch_size < 1:
        raise ValueError("RIFE multiplier must be >=2 and batch size must be positive")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    sys.path.insert(0, str(args.model.resolve().parent))
    sys.path.insert(0, str(args.repo.resolve()))
    import train_log.IFNet_HDv3 as rife_ifnet_module
    from train_log.IFNet_HDv3 import IFNet

    grid_cache: dict[tuple[str, str, tuple[int, ...]], torch.Tensor] = {}

    def dtype_safe_warp(tensor_input: torch.Tensor, tensor_flow: torch.Tensor) -> torch.Tensor:
        key = (str(tensor_flow.device), str(tensor_flow.dtype), tuple(tensor_flow.shape))
        if key not in grid_cache:
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
            grid_cache[key] = torch.cat((horizontal, vertical), dim=1)
        normalized_flow = torch.cat((
            tensor_flow[:, 0:1] / ((tensor_input.shape[3] - 1.0) / 2.0),
            tensor_flow[:, 1:2] / ((tensor_input.shape[2] - 1.0) / 2.0),
        ), dim=1)
        grid = (grid_cache[key] + normalized_flow).permute(0, 2, 3, 1)
        return F.grid_sample(
            tensor_input, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    rife_ifnet_module.warp = dtype_safe_warp
    input_paths = sorted(args.input.glob("*.png"), key=lambda path: int(path.stem))
    if len(input_paths) < 2:
        raise ValueError("RIFE input needs at least two numbered PNG files")
    args.output.mkdir(parents=True, exist_ok=False)

    model = IFNet()
    state = torch.load(args.model / "flownet.pkl", map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"RIFE checkpoint missing keys: {result.missing_keys}")
    model.to(device).eval()
    if args.fp16:
        model.half()
    model_dtype = next(model.parameters()).dtype

    first = Image.open(input_paths[0]).convert("RGB")
    height, width = first.height, first.width
    first.close()
    block = max(128, int(128 / args.scale))
    padded_height = ((height - 1) // block + 1) * block
    padded_width = ((width - 1) // block + 1) * block
    padding = (0, padded_width - width, 0, padded_height - height)

    def load_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if image.size != (width, height):
                raise ValueError(f"Mismatched input dimensions at {path}: {image.size}")
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        return F.pad(tensor.to(device=device, dtype=model_dtype) / 255.0, padding)

    def save_tensor(tensor: torch.Tensor, path: Path) -> None:
        array = (tensor[0, :, :height, :width].float().clamp(0, 1) * 255.0).round().byte()
        Image.fromarray(array.permute(1, 2, 0).cpu().numpy(), mode="RGB").save(
            path, compress_level=4
        )

    pair_count = len(input_paths) - 1
    shutil.copy2(input_paths[0], args.output / "0000000.png")
    pair_start = 0
    active_batch_size = min(args.batch_size, pair_count)
    with torch.inference_mode():
        while pair_start < pair_count:
            current_size = min(active_batch_size, pair_count - pair_start)
            pair_indices = list(range(pair_start, pair_start + current_size))
            left = right = inputs = None
            try:
                left = torch.cat([load_tensor(input_paths[index]) for index in pair_indices])
                right = torch.cat([load_tensor(input_paths[index + 1]) for index in pair_indices])
                inputs = torch.cat((left, right), dim=1)
                scale_list = [16 / args.scale, 8 / args.scale, 4 / args.scale, 2 / args.scale, 1 / args.scale]
                for step in range(1, args.multi):
                    _, _, merged = model(inputs, step / args.multi, scale_list)
                    middle = merged[-1]
                    for offset, pair_index in enumerate(pair_indices):
                        save_tensor(
                            middle[offset : offset + 1],
                            args.output / f"{pair_index * args.multi + step:07d}.png",
                        )
                for pair_index in pair_indices:
                    shutil.copy2(
                        input_paths[pair_index + 1],
                        args.output / f"{(pair_index + 1) * args.multi:07d}.png",
                    )
            except torch.cuda.OutOfMemoryError:
                if current_size == 1:
                    raise
                active_batch_size = max(1, current_size // 2)
                del left, right, inputs
                torch.cuda.empty_cache()
                print(f"RIFE OOM; retrying with batch_size={active_batch_size}", flush=True)
                continue
            pair_start += current_size
            print(f"RIFE pairs: {pair_start}/{pair_count}", flush=True)
    print(f"RIFE complete: {pair_count * args.multi + 1} PNG frames", flush=True)


if __name__ == "__main__":
    main()
