"""Runtime discovery, safe Hugging Face authentication, and access checks."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import DIFFUSERS_COMMIT, MODEL_ID, MODEL_REVISION


_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)((?:hf[_-]?token|huggingface[_-]?token|authorization)\s*[:=]\s*)[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:token|access_token|auth|authorization)=)[^&\s]+"
)


class EnvironmentValidationError(RuntimeError):
    """Raised when the selected production runtime cannot satisfy the contract."""


class ModelAccessError(EnvironmentValidationError):
    """Raised when gated model access has not been granted."""


@dataclass(frozen=True)
class AuthenticationResult:
    token: str
    source: str

    def __repr__(self) -> str:  # prevent accidental notebook display
        return f"AuthenticationResult(token=<redacted>, source={self.source!r})"


def redact_secrets(value: str) -> str:
    redacted = _TOKEN_RE.sub("<redacted-hf-token>", value)
    redacted = _SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", redacted)
    redacted = _BEARER_RE.sub(r"\1<redacted>", redacted)
    return _QUERY_SECRET_RE.sub(r"\1<redacted>", redacted)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        rows = []
        for line in completed.stdout.strip().splitlines():
            values = [item.strip() for item in line.split(",")]
            if len(values) == 5:
                rows.append(
                    {
                        "name": values[0],
                        "memory_total_mib": int(values[1]),
                        "memory_free_mib": int(values[2]),
                        "driver_version": values[3],
                        "compute_capability": values[4],
                    }
                )
        return {"available": bool(rows), "gpus": rows}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {"available": False, "gpus": []}


def collect_environment() -> dict[str, Any]:
    """Collect reproducibility information without serializing environment secrets."""

    cuda_available = torch.cuda.is_available()
    selected_device = "cuda:0" if cuda_available else "cpu"
    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except (RuntimeError, TypeError):
                free_bytes, total_bytes = None, properties.total_memory
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_vram_bytes": int(total_bytes),
                    "free_vram_bytes": int(free_bytes) if free_bytes is not None else None,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_available": cuda_available,
        "selected_device": selected_device,
        "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "cuda_devices": cuda_devices,
        "nvidia_smi": _nvidia_smi(),
        "packages": {
            name: _package_version(name)
            for name in (
                "diffusers",
                "transformers",
                "peft",
                "accelerate",
                "huggingface-hub",
                "safetensors",
                "numpy",
                "pillow",
            )
        },
        "diffusers_commit_expected": DIFFUSERS_COMMIT,
        "model_id_expected": MODEL_ID,
        "model_revision_expected": MODEL_REVISION,
    }


def write_environment(path: str | Path, environment: dict[str, Any] | None = None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = environment or collect_environment()
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)
    output.write_text(redact_secrets(serialized) + "\n", encoding="utf-8")
    return output


def resolve_hf_token(*, allow_interactive: bool = True) -> AuthenticationResult:
    """Resolve HF credentials without printing or persisting the token here."""

    environment_token = os.environ.get("HF_TOKEN")
    if environment_token:
        return AuthenticationResult(environment_token, "environment")

    try:
        from google.colab import userdata

        colab_token = userdata.get("HF_TOKEN")
        if colab_token:
            return AuthenticationResult(colab_token, "colab_secret")
    # ``google.colab`` can be importable from a Colab-backed kernel opened in
    # VS Code even though its UI-only Secrets bridge is unavailable.  That
    # bridge raises a custom TimeoutException (not a RuntimeError), so treat
    # any ordinary lookup failure as "no Colab secret" and continue to the
    # explicit interactive Hub login below.  BaseException subclasses such as
    # KeyboardInterrupt and SystemExit still propagate.
    except Exception:
        pass

    if not allow_interactive:
        raise ModelAccessError("No Hugging Face token is available in HF_TOKEN or Colab secrets")

    try:
        from huggingface_hub import get_token, login

        login(skip_if_logged_in=True)
        token = get_token()
    except Exception as exc:  # hub changes its interactive exception types across versions
        raise ModelAccessError(redact_secrets(f"Hugging Face login failed: {exc}")) from exc
    if not token:
        raise ModelAccessError("Hugging Face login completed without an available token")
    return AuthenticationResult(token, "interactive_login")


def verify_model_access(
    authentication: AuthenticationResult,
    *,
    model_id: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
) -> dict[str, Any]:
    """Check the gated repository once and return non-secret model metadata."""

    from huggingface_hub import HfApi

    try:
        info = HfApi(token=authentication.token).model_info(model_id, revision=revision)
    except Exception as exc:
        message = redact_secrets(str(exc))
        raise ModelAccessError(
            f"Cannot access gated model {model_id!r}. Accept its license terms and verify your HF account. {message}"
        ) from exc
    return {
        "model_id": info.id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "gated": info.gated,
        "private": info.private,
        "authentication_source": authentication.source,
    }


def require_cuda_for_production() -> torch.device:
    if not torch.cuda.is_available():
        raise EnvironmentValidationError("The full FLUX.2 Klein Base 9B reproduction requires a CUDA GPU")
    return torch.device("cuda:0")
