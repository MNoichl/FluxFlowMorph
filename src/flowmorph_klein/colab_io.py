"""Colab-aware path, staging, checksum, and run-ID utilities.

All active work is staged into local storage.  Colab-specific imports are
guarded so these helpers remain usable in normal Jupyter and CPU tests.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ChecksumMismatchError, InputStagingError
from .types import ColabPaths, FileChecksum, StagedFile


DEFAULT_PROJECT_ROOT = Path("/content/FlowMorphKlein9B")
DEFAULT_INPUT_ROOT = Path("/content/flowmorph_klein_images/max_v1")
DEFAULT_WORK_ROOT = Path("/content/flowmorph_klein_work/max_v1")
DEFAULT_RESULT_ROOT = Path(
    "/content/flowmorph_klein_results/max_v1/full_lora_reproduction_v1"
)
DEFAULT_HF_CACHE = Path("/content/hf_cache")
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/FlowMorphKlein9B")


def detect_colab() -> bool:
    """Return whether the current interpreter can import Google Colab APIs."""

    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def configure_colab_paths(
    *,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    work_root: str | Path = DEFAULT_WORK_ROOT,
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    hf_cache: str | Path = DEFAULT_HF_CACHE,
    drive_root: str | Path | None = DEFAULT_DRIVE_ROOT,
    create: bool = True,
    create_drive_root: bool = False,
) -> ColabPaths:
    """Resolve the standard directory layout and create local roots on demand."""

    paths = ColabPaths(
        project_root=Path(project_root).expanduser().resolve(strict=False),
        input_root=Path(input_root).expanduser().resolve(strict=False),
        work_root=Path(work_root).expanduser().resolve(strict=False),
        result_root=Path(result_root).expanduser().resolve(strict=False),
        hf_cache=Path(hf_cache).expanduser().resolve(strict=False),
        drive_root=(
            Path(drive_root).expanduser().resolve(strict=False)
            if drive_root is not None
            else None
        ),
    )
    if create:
        for path in (
            paths.project_root,
            paths.input_root,
            paths.work_root,
            paths.result_root,
            paths.hf_cache,
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise InputStagingError(f"cannot create local path {path}: {error}") from error
        if create_drive_root and paths.drive_root is not None:
            try:
                paths.drive_root.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise InputStagingError(
                    f"cannot create persistent Drive path {paths.drive_root}: {error}"
                ) from error
    return paths


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 without loading the full file into memory."""

    candidate = Path(path)
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as error:
        raise InputStagingError(f"cannot checksum {candidate}: {error}") from error
    return digest.hexdigest()


def _safe_relative_path(name: str) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise InputStagingError(f"unsafe staged filename {name!r}")
    if ":" in pure.parts[0]:
        raise InputStagingError(f"unsafe staged filename {name!r}")
    return Path(*pure.parts)


def _read_upload_bytes(value: Any, *, max_bytes: int) -> bytes:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, (bytearray, memoryview)):
        payload = bytes(value)
    elif hasattr(value, "read"):
        payload = value.read(max_bytes + 1)
        if not isinstance(payload, bytes):
            raise InputStagingError("uploaded file objects must return bytes")
    else:
        raise InputStagingError(
            f"unsupported uploaded value type {type(value).__name__}; expected bytes or a binary file"
        )
    if len(payload) > max_bytes:
        raise InputStagingError(
            f"uploaded file exceeds the configured {max_bytes}-byte safety limit"
        )
    return payload


def _atomic_write_bytes(destination: Path, payload: bytes, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise InputStagingError(f"refusing to overwrite staged file {destination}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except OSError as error:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise InputStagingError(f"cannot stage {destination}: {error}") from error


def stage_uploaded_files(
    destination: str | Path,
    uploaded: Mapping[str, Any] | None = None,
    *,
    overwrite: bool = False,
    max_file_bytes: int = 2 * 1024 * 1024 * 1024,
) -> tuple[StagedFile, ...]:
    """Stage a Colab upload mapping or an explicitly supplied byte mapping."""

    if uploaded is None:
        if not detect_colab():
            raise InputStagingError(
                "interactive upload is available only in Colab; pass an uploaded mapping instead"
            )
        from google.colab import files  # type: ignore[import-not-found]

        uploaded = files.upload()
    root = Path(destination).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    staged: list[StagedFile] = []
    destinations: set[Path] = set()
    for name, value in uploaded.items():
        relative = _safe_relative_path(str(name))
        target = (root / relative).resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as error:
            raise InputStagingError(f"upload path escapes destination: {name!r}") from error
        if target in destinations:
            raise InputStagingError(f"duplicate uploaded destination {relative}")
        destinations.add(target)
        payload = _read_upload_bytes(value, max_bytes=max_file_bytes)
        _atomic_write_bytes(target, payload, overwrite=overwrite)
        staged.append(
            StagedFile(
                source_name=str(name),
                destination=target,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    return tuple(staged)


def _atomic_copy(source: Path, destination: Path, *, overwrite: bool) -> StagedFile:
    if source.is_symlink():
        raise InputStagingError(f"refusing to stage symbolic link {source}")
    if not source.is_file():
        raise InputStagingError(f"staging source is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise InputStagingError(f"refusing to overwrite staged file {destination}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        shutil.copystat(source, destination)
    except OSError as error:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise InputStagingError(f"cannot copy {source} to {destination}: {error}") from error
    digest = sha256_file(destination)
    return StagedFile(source.name, destination, digest, destination.stat().st_size)


def stage_drive_inputs(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[StagedFile, ...]:
    """Copy a Drive file or folder into local working storage with checksums."""

    source_path = Path(source).expanduser().resolve(strict=False)
    destination_root = Path(destination).expanduser().resolve(strict=False)
    if not source_path.exists():
        raise InputStagingError(f"Drive input does not exist: {source_path}")
    if source_path == destination_root or destination_root.is_relative_to(source_path):
        raise InputStagingError("destination cannot be the source directory or one of its children")

    if source_path.is_file():
        return (_atomic_copy(source_path, destination_root / source_path.name, overwrite=overwrite),)
    if source_path.is_symlink() or not source_path.is_dir():
        raise InputStagingError(f"unsupported Drive input type: {source_path}")

    staged: list[StagedFile] = []
    for candidate in sorted(source_path.rglob("*")):
        if candidate.is_symlink():
            raise InputStagingError(f"refusing to stage symbolic link {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(source_path)
            staged.append(
                _atomic_copy(candidate, destination_root / relative, overwrite=overwrite)
            )
    return tuple(staged)


def extract_input_bundle(
    archive: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
    max_files: int = 10_000,
    max_total_bytes: int = 20 * 1024 * 1024 * 1024,
) -> tuple[StagedFile, ...]:
    """Safely extract a ZIP input bundle, rejecting traversal and symlinks."""

    archive_path = Path(archive).expanduser().resolve(strict=False)
    root = Path(destination).expanduser().resolve(strict=False)
    if not archive_path.is_file():
        raise InputStagingError(f"input bundle does not exist: {archive_path}")
    try:
        bundle = zipfile.ZipFile(archive_path, mode="r")
    except (OSError, zipfile.BadZipFile) as error:
        raise InputStagingError(f"invalid ZIP input bundle {archive_path}: {error}") from error

    staged: list[StagedFile] = []
    try:
        members = bundle.infolist()
        files = [member for member in members if not member.is_dir()]
        if len(files) > max_files:
            raise InputStagingError(
                f"input bundle contains {len(files)} files, above limit {max_files}"
            )
        total_size = sum(member.file_size for member in files)
        if total_size > max_total_bytes:
            raise InputStagingError(
                f"input bundle expands to {total_size} bytes, above limit {max_total_bytes}"
            )

        destinations: set[Path] = set()
        root.mkdir(parents=True, exist_ok=True)
        for member in files:
            relative = _safe_relative_path(member.filename)
            mode = member.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise InputStagingError(f"ZIP symbolic links are not supported: {member.filename}")
            if member.flag_bits & 0x1:
                raise InputStagingError(f"encrypted ZIP entries are not supported: {member.filename}")
            target = (root / relative).resolve(strict=False)
            try:
                target.relative_to(root)
            except ValueError as error:
                raise InputStagingError(
                    f"ZIP entry escapes destination: {member.filename}"
                ) from error
            if target in destinations:
                raise InputStagingError(f"duplicate ZIP destination {member.filename}")
            destinations.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                raise InputStagingError(f"refusing to overwrite extracted file {target}")

            temporary_name: str | None = None
            digest = hashlib.sha256()
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
                ) as output:
                    temporary_name = output.name
                    with bundle.open(member, mode="r") as source_handle:
                        while True:
                            chunk = source_handle.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, target)
            except OSError as error:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
                raise InputStagingError(
                    f"cannot extract {member.filename} from {archive_path}: {error}"
                ) from error
            staged.append(
                StagedFile(
                    source_name=member.filename,
                    destination=target,
                    sha256=digest.hexdigest(),
                    size_bytes=member.file_size,
                )
            )
    finally:
        bundle.close()
    return tuple(staged)


def _iter_files(paths: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for item in paths:
        candidate = Path(item).expanduser().resolve(strict=False)
        if candidate.is_symlink():
            raise InputStagingError(f"refusing to checksum symbolic link {candidate}")
        if candidate.is_file():
            files.add(candidate)
        elif candidate.is_dir():
            for child in candidate.rglob("*"):
                if child.is_symlink():
                    raise InputStagingError(f"refusing to checksum symbolic link {child}")
                if child.is_file():
                    files.add(child.resolve(strict=False))
        else:
            raise InputStagingError(f"checksum path does not exist: {candidate}")
    return sorted(files)


def compute_file_checksums(
    paths: Iterable[str | Path] | str | Path,
) -> tuple[FileChecksum, ...]:
    """Compute deterministic SHA-256 records for files or directory trees."""

    items = (paths,) if isinstance(paths, (str, Path)) else paths
    return tuple(
        FileChecksum(path=file, sha256=sha256_file(file), size_bytes=file.stat().st_size)
        for file in _iter_files(items)
    )


def verify_file_checksum(path: str | Path, expected_sha256: str) -> str:
    """Verify a SHA-256 and return its normalized lowercase form."""

    normalized = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ChecksumMismatchError("expected SHA-256 must contain exactly 64 hex characters")
    actual = sha256_file(path)
    if not secrets.compare_digest(actual, normalized):
        raise ChecksumMismatchError(
            f"checksum mismatch for {Path(path)}: expected {normalized}, got {actual}"
        )
    return actual


def create_run_id(
    project_name: str,
    *,
    now: datetime | None = None,
    nonce: str | None = None,
) -> str:
    """Create a filesystem-safe, sortable, collision-resistant run ID."""

    slug = re.sub(r"[^a-z0-9]+", "-", project_name.strip().lower()).strip("-")
    if not slug:
        raise InputStagingError("project name must contain at least one letter or digit")
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    suffix = nonce or secrets.token_hex(4)
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,32}", suffix):
        raise InputStagingError("run-ID nonce must be 4-32 safe characters")
    return f"{slug}-{timestamp:%Y%m%dT%H%M%SZ}-{suffix}"
