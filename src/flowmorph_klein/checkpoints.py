"""Atomic, compatibility-checked endpoint checkpoints.

Endpoint tensors are kept in safetensors and metadata in JSON so a checkpoint
can be inspected without importing this package.  Optimizer moments can be
included for exact mid-fit resumption; final compact endpoint states may omit
them.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file


REQUIRED_TENSORS = frozenset({"z", "delta", "u"})
TRANSACTION_TENSOR = "checkpoint.transaction_id"
GENERATIONS_DIRECTORY = ".generations"
LATEST_POINTER = "LATEST"
RETAINED_GENERATIONS = 2
COMPATIBILITY_FIELDS = (
    "endpoint",
    "model_id",
    "model_revision",
    "lora_source",
    "lora_revision",
    "lora_file_sha256",
    "lora_scale",
    "prompt_checksum",
    "source_image_checksum",
    "processed_image_checksum",
    "preprocessing_hash",
    "resize_mode",
    "scheduler_configuration",
    "start_timestep_index",
    "latent_shape",
    "optimizer_configuration",
    "loss_mode",
    "guidance_configuration",
    "precision_configuration",
    "diffusers_commit",
    "flowmorph_commit",
    "flux2_commit",
)


class CheckpointError(RuntimeError):
    """Base error for malformed or incompatible endpoint checkpoints."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when resuming would mix incompatible model or input state."""

    def __init__(self, mismatches: Mapping[str, tuple[Any, Any]]):
        self.mismatches = dict(mismatches)
        fields = ", ".join(sorted(mismatches))
        super().__init__(f"Checkpoint is incompatible in fields: {fields}")


@dataclass(frozen=True)
class LoadedCheckpoint:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    directory: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 for checkpoint compatibility metadata."""

    return hashlib.sha256(_canonical_json(dict(metadata)).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CheckpointError(f"Checkpoint entry {name!r} is not a tensor")
    if value.layout != torch.strided:
        raise CheckpointError(f"Checkpoint entry {name!r} must use strided tensor layout")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise CheckpointError(
            f"Checkpoint entry {name!r} contains non-finite values"
        )
    # Clone unconditionally: callers may legitimately pass the same zero
    # tensor for multiple logical fields, while safetensors rejects shared
    # storage because it cannot preserve aliases.
    return value.detach().to("cpu").contiguous().clone()


def flatten_optimizer_state(optimizer: torch.optim.Optimizer) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Convert an optimizer state dict into safetensor entries plus JSON data."""

    state_dict = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    scalar_state: dict[str, dict[str, Any]] = {}
    for parameter_id, values in state_dict["state"].items():
        pid = str(parameter_id)
        scalar_state[pid] = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                tensors[f"optimizer.state.{pid}.{name}"] = _checkpoint_tensor(value, name)
            else:
                scalar_state[pid][name] = value
    descriptor = {
        "param_groups": state_dict["param_groups"],
        "scalar_state": scalar_state,
    }
    return tensors, descriptor


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    tensors: Mapping[str, torch.Tensor],
    descriptor: Mapping[str, Any],
) -> None:
    """Restore optimizer moments previously returned by ``flatten_optimizer_state``."""

    state: dict[int, dict[str, Any]] = {}
    scalar_state = descriptor.get("scalar_state", {})
    for parameter_id, values in scalar_state.items():
        state[int(parameter_id)] = dict(values)
    prefix = "optimizer.state."
    for key, value in tensors.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        parameter_id, name = remainder.split(".", maxsplit=1)
        state.setdefault(int(parameter_id), {})[name] = value
    optimizer.load_state_dict({"state": state, "param_groups": list(descriptor["param_groups"])})


def unflatten_optimizer_state(
    tensors: Mapping[str, torch.Tensor],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild an optimizer state dict for an optimizer created after loading.

    ``torch.optim.Optimizer.load_state_dict`` moves tensor state to the
    current parameter devices, so checkpoint tensors intentionally remain on
    CPU here.
    """

    state: dict[int, dict[str, Any]] = {
        int(parameter_id): dict(values)
        for parameter_id, values in descriptor.get("scalar_state", {}).items()
    }
    prefix = "optimizer.state."
    for key, value in tensors.items():
        if key.startswith(prefix):
            parameter_id, name = key[len(prefix) :].split(".", maxsplit=1)
            state.setdefault(int(parameter_id), {})[name] = value
    return {"state": state, "param_groups": list(descriptor["param_groups"])}


def save_endpoint_checkpoint(
    directory: str | Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Atomically write one endpoint checkpoint and return its directory."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    missing = REQUIRED_TENSORS.difference(tensors)
    if missing:
        raise CheckpointError(f"Endpoint checkpoint is missing tensors: {sorted(missing)}")

    saved_tensors = {name: _checkpoint_tensor(value, name) for name, value in tensors.items()}
    output_metadata = dict(metadata)
    output_metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    output_metadata.setdefault("latent_shape", list(saved_tensors["z"].shape))

    if tuple(saved_tensors["z"].shape) != tuple(saved_tensors["delta"].shape):
        raise CheckpointError("z and delta tensor shapes differ")
    if tuple(saved_tensors["z"].shape) != tuple(saved_tensors["u"].shape):
        raise CheckpointError("z and u tensor shapes differ")

    transaction_id = secrets.randbits(62)
    saved_tensors[TRANSACTION_TENSOR] = torch.tensor(transaction_id, dtype=torch.int64)
    output_metadata["transaction_id"] = transaction_id

    if optimizer is not None:
        optimizer_tensors, descriptor = flatten_optimizer_state(optimizer)
        saved_tensors.update(optimizer_tensors)
        output_metadata["optimizer_state"] = descriptor
        output_metadata["optimizer_state_saved"] = True
    else:
        output_metadata["optimizer_state_saved"] = False

    output_metadata["metadata_fingerprint"] = metadata_fingerprint(
        {key: output_metadata.get(key) for key in COMPATIBILITY_FIELDS}
    )

    generations = destination / GENERATIONS_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)
    generation_name = f"{transaction_id:016x}"
    generation_path = generations / generation_name
    pending_path = Path(
        tempfile.mkdtemp(prefix=".pending-", dir=generations)
    )
    try:
        pending_tensor_path = pending_path / "tensors.safetensors"
        save_file(saved_tensors, pending_tensor_path)
        # Re-open before promotion; a truncated write must never become the checkpoint.
        loaded = load_file(pending_tensor_path, device="cpu")
        if set(loaded) != set(saved_tensors):
            raise CheckpointError("Safetensors verification returned a different key set")
        _atomic_json(pending_path / "metadata.json", output_metadata)
        os.replace(pending_path, generation_path)
    except BaseException:
        shutil.rmtree(pending_path, ignore_errors=True)
        raise

    # The pointer is the commit record. Canonical files remain for the
    # requested transparent layout, while resume authorization follows only a
    # complete generation pair.
    _atomic_text(destination / LATEST_POINTER, generation_name + "\n")
    _atomic_copy(
        generation_path / "tensors.safetensors",
        destination / "tensors.safetensors",
    )
    _atomic_json(destination / "metadata.json", output_metadata)

    complete_generations = sorted(
        (
            path
            for path in generations.iterdir()
            if path.is_dir()
            and not path.name.startswith(".pending-")
            and (path / "tensors.safetensors").is_file()
            and (path / "metadata.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for obsolete in complete_generations[RETAINED_GENERATIONS:]:
        shutil.rmtree(obsolete)
    return destination


def validate_checkpoint_compatibility(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: tuple[str, ...] = COMPATIBILITY_FIELDS,
) -> None:
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in fields:
        if field not in expected:
            continue
        if _canonical_json(actual.get(field)) != _canonical_json(expected.get(field)):
            mismatches[field] = (actual.get(field), expected.get(field))
    if mismatches:
        raise CheckpointCompatibilityError(mismatches)


def load_endpoint_checkpoint(
    directory: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    source = Path(directory)
    generations = source / GENERATIONS_DIRECTORY
    candidates: list[tuple[Path, Path, str]] = []
    pointer_path = source / LATEST_POINTER
    pointed_generation: Path | None = None
    if pointer_path.is_file():
        try:
            generation_name = pointer_path.read_text(encoding="utf-8").strip()
        except OSError:
            generation_name = ""
        if generation_name and Path(generation_name).name == generation_name:
            pointed_generation = generations / generation_name
            candidates.append(
                (
                    pointed_generation / "tensors.safetensors",
                    pointed_generation / "metadata.json",
                    f"generation {generation_name}",
                )
            )

    # Canonical files support the specified layout and legacy checkpoints.
    candidates.append(
        (
            source / "tensors.safetensors",
            source / "metadata.json",
            "canonical pair",
        )
    )
    if generations.is_dir():
        remaining = sorted(
            (
                path
                for path in generations.iterdir()
                if path.is_dir()
                and path != pointed_generation
                and not path.name.startswith(".pending-")
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        candidates.extend(
            (
                path / "tensors.safetensors",
                path / "metadata.json",
                f"generation {path.name}",
            )
            for path in remaining
        )

    failures: list[str] = []
    compatibility_error: CheckpointCompatibilityError | None = None
    for tensor_path, metadata_path, label in candidates:
        if not tensor_path.is_file() or not metadata_path.is_file():
            failures.append(f"{label}: incomplete")
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not isinstance(metadata, dict):
                raise CheckpointError("checkpoint metadata must be a JSON object")
            tensors = load_file(tensor_path, device=str(device))
            transaction_tensor = tensors.pop(TRANSACTION_TENSOR, None)
            if transaction_tensor is None or transaction_tensor.numel() != 1:
                raise CheckpointError(
                    "checkpoint tensor transaction marker is missing or malformed"
                )
            if int(transaction_tensor.item()) != metadata.get("transaction_id"):
                raise CheckpointError(
                    "checkpoint tensor/metadata transaction mismatch; the prior atomic "
                    "write was interrupted"
                )
            missing = REQUIRED_TENSORS.difference(tensors)
            if missing:
                raise CheckpointError(
                    f"checkpoint is missing tensors: {sorted(missing)}"
                )
            shapes = {tuple(tensors[name].shape) for name in REQUIRED_TENSORS}
            if len(shapes) != 1:
                raise CheckpointError("checkpoint endpoint tensor shapes differ")
            non_finite = sorted(
                name
                for name, tensor in tensors.items()
                if tensor.is_floating_point()
                and not bool(torch.isfinite(tensor).all().item())
            )
            if non_finite:
                raise CheckpointError(
                    "checkpoint contains non-finite tensors: "
                    + ", ".join(non_finite)
                )
            if expected_metadata is not None:
                validate_checkpoint_compatibility(metadata, expected_metadata)
            return LoadedCheckpoint(
                tensors=dict(tensors),
                metadata=metadata,
                directory=source,
            )
        except CheckpointCompatibilityError as error:
            if label == "canonical pair" or (
                pointed_generation is not None
                and label == f"generation {pointed_generation.name}"
            ):
                raise
            compatibility_error = compatibility_error or error
            failures.append(f"{label}: {error}")
        except (
            CheckpointError,
            OSError,
            json.JSONDecodeError,
            SafetensorError,
            ValueError,
        ) as error:
            failures.append(f"{label}: {error}")

    if compatibility_error is not None and all(
        "incompatible" in failure for failure in failures if "incomplete" not in failure
    ):
        raise compatibility_error
    if not failures:
        raise CheckpointError(f"Incomplete checkpoint at {source}")
    raise CheckpointError(
        f"No complete checkpoint generation is recoverable at {source}: "
        + "; ".join(failures)
    )
