"""Compact, atomic FlowMorph run archives with explicit exclusions."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


ARCHIVE_SUFFIX = ".flowmorph-klein.zip"
INCLUDED_TOP_LEVEL_DIRECTORIES = frozenset(
    {
        "inputs",
        "conditioning",
        "conditioning_comparison",
        "checkpoints",
        "optimization",
        "endpoint_reconstruction",
        "raw_frames",
        "display_frames",
        "previews",
    }
)
INCLUDED_TOP_LEVEL_FILES = frozenset(
    {
        "config.resolved.yaml",
        "environment.json",
        "run_manifest.json",
        "execution.log",
        "checksums.sha256",
        "model_report.json",
        "lora_report.json",
        "memory_report.json",
        "schedule.json",
        "attention_and_schedule.json",
        "metrics.json",
    }
)
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        "__pycache__",
        "hf_cache",
        "hub",
        "models--black-forest-labs",
        ".cache",
        "pip-cache",
    }
)
TEXT_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".txt", ".log", ".csv", ".md", ".toml"})
TOKEN_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"(?i)(?:hf[_-]?token|huggingface[_-]?token)\s*[:=]\s*['\"]?[^\s'\",]{8,}"),
)


class PackagingError(RuntimeError):
    """Raised when an archive is incomplete, unsafe, or corrupt."""


@dataclass(frozen=True)
class ArchiveReport:
    path: Path
    sha256: str
    size_bytes: int
    member_count: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_forbidden(relative_path: Path) -> bool:
    lowered = {part.lower() for part in relative_path.parts}
    if lowered.intersection(FORBIDDEN_PARTS):
        return True
    if any(part.lower().startswith(".pending-") for part in relative_path.parts):
        return True
    name = relative_path.name.lower()
    if name.endswith((".pyc", ".part", ".tmp")):
        return True
    if name.endswith(ARCHIVE_SUFFIX) or name.endswith(f"{ARCHIVE_SUFFIX}.sha256"):
        return True
    # Adapter/model weights are excluded, while explicit endpoint state files are allowed.
    if name.endswith((".bin", ".ckpt", ".pt", ".pth")):
        return True
    if name.endswith(".safetensors") and (not relative_path.parts or relative_path.parts[0] != "checkpoints"):
        return True
    return False


def iter_archive_files(run_directory: str | Path) -> Iterator[tuple[Path, Path]]:
    """Yield ``(absolute, relative)`` members from the output allowlist."""

    root = Path(run_directory).resolve()
    for name in sorted(INCLUDED_TOP_LEVEL_FILES):
        path = root / name
        if path.is_file() and not _is_forbidden(Path(name)):
            yield path, Path(name)
    for top_level in sorted(INCLUDED_TOP_LEVEL_DIRECTORIES):
        directory = root / top_level
        if not directory.is_dir():
            continue
        for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
            relative = path.relative_to(root)
            if not _is_forbidden(relative):
                yield path, relative


def _scan_for_secrets(path: Path, relative_path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    data = path.read_bytes()
    for pattern in TOKEN_PATTERNS:
        if pattern.search(data):
            raise PackagingError(f"Potential Hugging Face token found in {relative_path}")


def write_checksums(
    run_directory: str | Path,
    members: Iterable[tuple[Path, Path]] | None = None,
) -> Path:
    root = Path(run_directory).resolve()
    selected = list(iter_archive_files(root) if members is None else members)
    checksum_path = root / "checksums.sha256"
    lines = [f"{sha256_file(path)}  {relative.as_posix()}" for path, relative in selected if path != checksum_path]
    fd, temporary_name = tempfile.mkstemp(prefix=".checksums.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if lines:
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, checksum_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return checksum_path


def validate_archive(path: str | Path, expected_members: Iterable[str] | None = None) -> list[str]:
    archive = Path(path)
    if not archive.is_file():
        raise PackagingError(f"Archive does not exist: {archive}")
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as handle:
            corrupt = handle.testzip()
            if corrupt is not None:
                raise PackagingError(f"Archive member failed CRC validation: {corrupt}")
            names = handle.namelist()
    except zipfile.BadZipFile as exc:
        raise PackagingError(f"Invalid ZIP archive: {archive}") from exc
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise PackagingError("Archive contains an unsafe member path")
    if expected_members is not None:
        missing = sorted(set(expected_members).difference(names))
        if missing:
            raise PackagingError(f"Archive is missing expected members: {missing}")
    return names


def create_run_archive(
    run_directory: str | Path,
    run_id: str,
    *,
    required_members: Iterable[str] = ("config.resolved.yaml", "run_manifest.json"),
) -> ArchiveReport:
    """Build, validate, hash, and atomically publish a ZIP64 run archive."""

    root = Path(run_directory).resolve()
    if not root.is_dir():
        raise PackagingError(f"Run directory does not exist: {root}")
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise PackagingError("run_id must be a non-empty filename component")

    write_checksums(root)
    members = list(iter_archive_files(root))
    member_names = [relative.as_posix() for _, relative in members]
    missing = sorted(set(required_members).difference(member_names))
    if missing:
        raise PackagingError(f"Run is missing required archive members: {missing}")
    for path, relative in members:
        _scan_for_secrets(path, relative)

    artifact_directory = root / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    final_path = artifact_directory / f"{run_id}{ARCHIVE_SUFFIX}"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{run_id}.", suffix=".zip.tmp", dir=artifact_directory)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as handle:
            for path, relative in members:
                handle.write(path, arcname=relative.as_posix())
        names = validate_archive(temporary_path, expected_members=member_names)
        os.replace(temporary_path, final_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    digest = sha256_file(final_path)
    checksum_path = final_path.with_suffix(final_path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {final_path.name}\n", encoding="utf-8")
    return ArchiveReport(
        path=final_path,
        sha256=digest,
        size_bytes=final_path.stat().st_size,
        member_count=len(names),
    )
