"""Fixtures for explicitly opt-in FLUX.2 Klein Base-9B integration tests.

The tests in this directory deliberately do not infer that a CUDA device,
gated-model access, or a compatible LoRA is available.  A real-model test
runs only after both the global gate and its test-specific gate are set to a
truthy value.  This makes a skipped test an honest unexecuted test, rather
than a synthetic substitute reported as a production pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from flowmorph_klein import MODEL_REVISION
from flowmorph_klein.config import ResolvedRunConfig


GLOBAL_GATE = "FLOWMORPH_RUN_FLUX2_9B_INTEGRATION"
TRUTHY = frozenset({"1", "true", "yes", "on"})


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY


@dataclass(frozen=True)
class IntegrationHarness:
    root: Path

    @property
    def hf_cache(self) -> Path:
        configured = os.environ.get("FLOWMORPH_TEST_HF_CACHE") or os.environ.get("HF_HOME")
        return Path(configured).expanduser() if configured else self.root / "hf_cache"

    @property
    def lora_source(self) -> str | None:
        value = os.environ.get("FLOWMORPH_TEST_LORA_SOURCE", "").strip()
        return value or None

    def require(self, *specific_gates: str, require_lora: bool = False) -> None:
        missing = [name for name in (GLOBAL_GATE, *specific_gates) if not _enabled(name)]
        if missing:
            pytest.skip("opt-in production integration disabled; set " + ", ".join(missing))
        if not torch.cuda.is_available():
            pytest.skip("opt-in production integration requires a CUDA GPU")
        if require_lora and self.lora_source is None:
            pytest.skip("LoRA integration requires FLOWMORPH_TEST_LORA_SOURCE")

    def _write_endpoint_images(self) -> tuple[Path, Path]:
        input_root = self.root / "images"
        input_root.mkdir(parents=True, exist_ok=True)
        source_path = input_root / "source.png"
        target_path = input_root / "target.png"

        source = Image.new("RGB", (512, 512), (24, 58, 132))
        source_draw = ImageDraw.Draw(source)
        source_draw.ellipse((112, 112, 400, 400), fill=(230, 184, 62))
        source.save(source_path)

        target = Image.new("RGB", (512, 512), (112, 32, 78))
        target_draw = ImageDraw.Draw(target)
        target_draw.rectangle((112, 112, 400, 400), fill=(74, 205, 164))
        target.save(target_path)
        return source_path, target_path

    def config(
        self,
        *,
        run_mode: str,
        frame_count: int,
        source_steps: int,
        target_steps: int,
        with_lora: bool = False,
    ) -> ResolvedRunConfig:
        source_path, target_path = self._write_endpoint_images()
        profile = os.environ.get("FLOWMORPH_TEST_PROFILE", "a100_80gb_full")
        lora: dict[str, object] = {}
        if with_lora:
            if self.lora_source is None:
                raise RuntimeError("with_lora=True requires FLOWMORPH_TEST_LORA_SOURCE")
            lora = {
                "source": self.lora_source,
                "revision": os.environ.get("FLOWMORPH_TEST_LORA_REVISION") or None,
                "weight_name": os.environ.get("FLOWMORPH_TEST_LORA_WEIGHT_NAME") or None,
                "fit_scale": float(os.environ.get("FLOWMORPH_TEST_LORA_SCALE", "1.0")),
                "render_scale": float(os.environ.get("FLOWMORPH_TEST_LORA_SCALE", "1.0")),
            }

        return ResolvedRunConfig.model_validate(
            {
                "run_mode": run_mode,
                "project": {
                    "name": f"integration_{run_mode}",
                    "repository_root": Path(__file__).resolve().parents[2],
                },
                "paths": {
                    "input_root": self.root / "images",
                    "work_root": self.root / "work",
                    "result_root": self.root / "results",
                    "hf_cache": self.hf_cache,
                    "drive_root": None,
                },
                "model": {
                    "revision": MODEL_REVISION,
                    "profile": profile,
                },
                "lora": lora,
                "input": {
                    "source_image": source_path,
                    "target_image": target_path,
                    "source_prompt": "a yellow circle on a blue field",
                    "target_prompt": "a green square on a burgundy field",
                    "bridge_prompt": "a clean geometric transformation",
                    "width": 512,
                    "height": 512,
                },
                "flowmorph": {
                    "scheduler_points": 100,
                    "start_timestep_index": 35,
                    "optimization_steps_source": source_steps,
                    "optimization_steps_target": target_steps,
                    "frame_count": frame_count,
                    "render_indices": [35, 55, 75, 95],
                },
                "memory": {"run_production_backward_probe": True},
            }
        )


@pytest.fixture
def integration_harness(tmp_path: Path) -> IntegrationHarness:
    return IntegrationHarness(tmp_path)
