"""Deterministic Hugging Face source parsing and asset resolution.

Only Hugging Face Hub APIs perform remote access.  Tokens are passed directly
to those APIs and are deliberately absent from every returned object and error
message.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse


_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"})


@dataclass(frozen=True, slots=True)
class ParsedHuggingFaceSource:
    original: str
    repo_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    weight_name: str | None = None
    local_path: Path | None = None

    @property
    def is_local(self) -> bool:
        return self.local_path is not None

    @property
    def filename(self) -> str | None:
        if self.weight_name is None:
            return None
        return f"{self.subfolder}/{self.weight_name}" if self.subfolder else self.weight_name


# A shorter public alias is convenient for callers and backwards compatible
# with early notebook drafts that used this name.
HuggingFaceSource = ParsedHuggingFaceSource


@dataclass(frozen=True, slots=True)
class HuggingFaceRepositoryInspection:
    repo_id: str
    requested_revision: str | None
    resolved_revision: str
    files: tuple[str, ...]
    safetensors_files: tuple[str, ...]
    card_data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedHuggingFaceFile:
    local_path: Path
    sha256: str
    size_bytes: int
    repo_id: str | None
    requested_revision: str | None
    resolved_revision: str | None
    subfolder: str | None
    weight_name: str

    @property
    def is_local_source(self) -> bool:
        return self.repo_id is None


def _validate_repo_id(repo_id: str) -> str:
    parts = repo_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"expected a Hugging Face model repository ID 'owner/name', got {repo_id!r}")
    if any("\\" in part or "?" in part or "#" in part for part in parts):
        raise ValueError(f"invalid Hugging Face repository ID {repo_id!r}")
    return repo_id


def _safe_hub_path(parts: list[str]) -> str:
    decoded = [unquote(part) for part in parts]
    pure = PurePosixPath(*decoded)
    if not decoded or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("Hugging Face file/subfolder path is empty or unsafe")
    return pure.as_posix()


def _split_hub_file(path: str) -> tuple[str | None, str]:
    pure = PurePosixPath(path)
    parent = None if str(pure.parent) == "." else pure.parent.as_posix()
    return parent, pure.name


def _looks_like_hf_url(value: str) -> bool:
    candidate = value if "://" in value else f"https://{value}"
    return (urlparse(candidate).hostname or "").lower() in _HF_HOSTS


def parse_huggingface_source(source: str | Path) -> ParsedHuggingFaceSource:
    """Parse repository IDs/pages, file links, resolve links, or local files."""

    original = str(source).strip()
    if not original:
        raise ValueError("Hugging Face source must not be empty")
    if "://" in original and not _looks_like_hf_url(original):
        raise ValueError("only huggingface.co URLs and local .safetensors paths are supported")

    # A .safetensors string that is not an HF URL is an explicit local-file
    # source, even before it has been staged into place.
    if not _looks_like_hf_url(original):
        candidate = Path(original).expanduser()
        if candidate.suffix.lower() == ".safetensors":
            return ParsedHuggingFaceSource(
                original=original,
                local_path=candidate.resolve(strict=False),
                weight_name=candidate.name,
            )

    if _looks_like_hf_url(original):
        url_value = original if "://" in original else f"https://{original}"
        parsed_url = urlparse(url_value)
        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError("Hugging Face URLs must use http or https")
        segments = [unquote(part) for part in parsed_url.path.split("/") if part]
        if segments and segments[0] == "models":
            segments = segments[1:]
        if len(segments) < 2:
            raise ValueError("Hugging Face URL must identify a model repository")
        if segments[0] in {"datasets", "spaces"}:
            raise ValueError("only Hugging Face model repositories are supported")
        repo_id = _validate_repo_id("/".join(segments[:2]))
        remainder = segments[2:]
        query = parse_qs(parsed_url.query)
        query_revision = query.get("revision", [None])[0]
        if not remainder:
            return ParsedHuggingFaceSource(
                original=original,
                repo_id=repo_id,
                revision=query_revision,
            )

        marker = remainder[0]
        if marker not in {"blob", "resolve", "tree"}:
            raise ValueError(
                "unsupported Hugging Face URL path; expected a repository, blob, resolve, or tree page"
            )
        if len(remainder) < 2:
            raise ValueError(f"Hugging Face {marker} URL is missing its revision")
        revision = remainder[1]
        if query_revision is not None and query_revision != revision:
            raise ValueError("conflicting revisions in Hugging Face URL path and query")
        asset_parts = remainder[2:]
        if marker == "tree":
            subfolder = _safe_hub_path(asset_parts) if asset_parts else None
            return ParsedHuggingFaceSource(
                original=original,
                repo_id=repo_id,
                revision=revision,
                subfolder=subfolder,
            )
        if not asset_parts:
            raise ValueError(f"Hugging Face {marker} URL is missing its filename")
        file_path = _safe_hub_path(asset_parts)
        subfolder, weight_name = _split_hub_file(file_path)
        return ParsedHuggingFaceSource(
            original=original,
            repo_id=repo_id,
            revision=revision,
            subfolder=subfolder,
            weight_name=weight_name,
        )

    # Bare repository IDs optionally accept the familiar owner/repo@revision
    # shorthand. File paths remain unambiguous because .safetensors was handled
    # above.
    repo_value = original.rstrip("/")
    revision = None
    if "@" in repo_value:
        repo_value, revision = repo_value.rsplit("@", 1)
        if not revision:
            raise ValueError("repository revision after '@' must not be empty")
    return ParsedHuggingFaceSource(
        original=original,
        repo_id=_validate_repo_id(repo_value),
        revision=revision,
    )


def _card_data_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return dict(result) if isinstance(result, dict) else {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def inspect_huggingface_repository(
    source: str | Path | ParsedHuggingFaceSource,
    *,
    revision: str | None = None,
    token: str | None = None,
    api: Any | None = None,
) -> HuggingFaceRepositoryInspection:
    """Inspect a model repository via ``HfApi.model_info``."""

    parsed = source if isinstance(source, ParsedHuggingFaceSource) else parse_huggingface_source(source)
    if parsed.is_local or parsed.repo_id is None:
        raise ValueError("repository inspection requires a Hugging Face repository source")
    if revision is not None and parsed.revision is not None and revision != parsed.revision:
        raise ValueError("explicit revision conflicts with the revision embedded in the source")
    requested_revision = revision or parsed.revision
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:  # pragma: no cover - dependency error in production only
            raise RuntimeError("huggingface-hub is required to inspect remote LoRA assets") from error
        api = HfApi()
    info = api.model_info(
        repo_id=parsed.repo_id,
        revision=requested_revision,
        token=token,
    )
    resolved_revision = getattr(info, "sha", None)
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise RuntimeError("Hugging Face repository response did not include a resolved commit SHA")
    siblings = getattr(info, "siblings", ()) or ()
    files: list[str] = []
    for sibling in siblings:
        filename = (
            sibling.get("rfilename")
            if isinstance(sibling, dict)
            else getattr(sibling, "rfilename", None)
        )
        if isinstance(filename, str) and filename:
            files.append(filename)
    files_tuple = tuple(sorted(set(files)))
    return HuggingFaceRepositoryInspection(
        repo_id=parsed.repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        files=files_tuple,
        safetensors_files=tuple(
            filename for filename in files_tuple if filename.lower().endswith(".safetensors")
        ),
        card_data=_card_data_to_dict(getattr(info, "card_data", None)),
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_source_field(name: str, embedded: str | None, explicit: str | None) -> str | None:
    if embedded is not None and explicit is not None and embedded != explicit:
        raise ValueError(f"explicit {name} conflicts with {name} embedded in the source")
    return explicit or embedded


def resolve_huggingface_file(
    source: str | Path | ParsedHuggingFaceSource,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    weight_name: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    api: Any | None = None,
    download_fn: Callable[..., str] | None = None,
) -> ResolvedHuggingFaceFile:
    """Resolve exactly one safetensors file and record its immutable identity."""

    parsed = source if isinstance(source, ParsedHuggingFaceSource) else parse_huggingface_source(source)
    if parsed.is_local:
        if any(value is not None for value in (revision, subfolder, weight_name)):
            if weight_name is not None and weight_name == parsed.weight_name and revision is None and subfolder is None:
                pass
            else:
                raise ValueError("revision/subfolder/weight_name overrides do not apply to local files")
        assert parsed.local_path is not None
        local_path = parsed.local_path
        if not local_path.is_file():
            raise FileNotFoundError(f"local LoRA file does not exist: {local_path}")
        if local_path.suffix.lower() != ".safetensors":
            raise ValueError("local LoRA files must use the .safetensors format")
        return ResolvedHuggingFaceFile(
            local_path=local_path.resolve(),
            sha256=_sha256_file(local_path),
            size_bytes=local_path.stat().st_size,
            repo_id=None,
            requested_revision=None,
            resolved_revision=None,
            subfolder=None,
            weight_name=local_path.name,
        )

    if parsed.repo_id is None:
        raise ValueError("remote source did not contain a repository ID")
    selected_revision = _merge_source_field("revision", parsed.revision, revision)
    selected_subfolder = _merge_source_field("subfolder", parsed.subfolder, subfolder)
    selected_weight_name = _merge_source_field("weight_name", parsed.weight_name, weight_name)
    if selected_subfolder is not None:
        selected_subfolder = _safe_hub_path(selected_subfolder.strip("/").split("/"))
    inspection = inspect_huggingface_repository(
        ParsedHuggingFaceSource(
            original=parsed.original,
            repo_id=parsed.repo_id,
            revision=selected_revision,
            subfolder=selected_subfolder,
            weight_name=selected_weight_name,
        ),
        token=token,
        api=api,
    )

    prefix = f"{selected_subfolder.strip('/')}/" if selected_subfolder else ""
    if selected_weight_name is not None:
        if "/" in selected_weight_name or "\\" in selected_weight_name:
            raise ValueError("weight_name must be a basename; use subfolder for its directory")
        selected_filename = prefix + selected_weight_name
        if selected_filename not in inspection.files:
            raise FileNotFoundError(
                f"selected file {selected_filename!r} is absent from {parsed.repo_id} "
                f"at revision {inspection.resolved_revision}"
            )
    else:
        candidates = [
            filename
            for filename in inspection.safetensors_files
            if (not prefix or filename.startswith(prefix))
        ]
        if not candidates:
            location = f" under {selected_subfolder!r}" if selected_subfolder else ""
            raise FileNotFoundError(
                f"repository {parsed.repo_id} has no safetensors files{location}"
            )
        if len(candidates) != 1:
            raise ValueError(
                "repository contains multiple safetensors candidates; set lora.weight_name explicitly: "
                + ", ".join(candidates)
            )
        selected_filename = candidates[0]
        selected_subfolder, selected_weight_name = _split_hub_file(selected_filename)

    if not selected_filename.lower().endswith(".safetensors"):
        raise ValueError("selected LoRA file must use the .safetensors format")
    if download_fn is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:  # pragma: no cover - dependency error in production only
            raise RuntimeError("huggingface-hub is required to download remote LoRA assets") from error
        download_fn = hf_hub_download
    downloaded = download_fn(
        repo_id=parsed.repo_id,
        filename=selected_filename,
        revision=inspection.resolved_revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )
    local_path = Path(downloaded).resolve(strict=False)
    if not local_path.is_file():
        raise FileNotFoundError("Hugging Face download API did not return a readable file")
    return ResolvedHuggingFaceFile(
        local_path=local_path,
        sha256=_sha256_file(local_path),
        size_bytes=local_path.stat().st_size,
        repo_id=parsed.repo_id,
        requested_revision=selected_revision,
        resolved_revision=inspection.resolved_revision,
        subfolder=selected_subfolder,
        weight_name=selected_weight_name or Path(selected_filename).name,
    )
