from pathlib import Path

import pytest

from flowmorph_klein.hf_assets import parse_huggingface_source


def test_repository_id() -> None:
    parsed = parse_huggingface_source("owner/adapter")
    assert parsed.repo_id == "owner/adapter"
    assert parsed.revision is None
    assert not parsed.is_local


@pytest.mark.parametrize(
    "source",
    (
        "https://huggingface.co/owner/adapter",
        "huggingface.co/owner/adapter",
        "https://www.huggingface.co/owner/adapter/",
    ),
)
def test_repository_page(source: str) -> None:
    parsed = parse_huggingface_source(source)
    assert parsed.repo_id == "owner/adapter"
    assert parsed.filename is None


def test_blob_file_link_extracts_revision_subfolder_and_weight() -> None:
    parsed = parse_huggingface_source(
        "https://huggingface.co/owner/adapter/blob/a1b2c3/loras/subject.safetensors"
    )
    assert parsed.repo_id == "owner/adapter"
    assert parsed.revision == "a1b2c3"
    assert parsed.subfolder == "loras"
    assert parsed.weight_name == "subject.safetensors"
    assert parsed.filename == "loras/subject.safetensors"


def test_resolve_file_link() -> None:
    parsed = parse_huggingface_source(
        "https://huggingface.co/owner/adapter/resolve/main/deep/path/subject.safetensors?download=true"
    )
    assert parsed.revision == "main"
    assert parsed.subfolder == "deep/path"
    assert parsed.weight_name == "subject.safetensors"


def test_tree_link_extracts_subfolder() -> None:
    parsed = parse_huggingface_source(
        "https://huggingface.co/owner/adapter/tree/release-v2/loras/portrait"
    )
    assert parsed.revision == "release-v2"
    assert parsed.subfolder == "loras/portrait"
    assert parsed.weight_name is None


def test_repository_revision_shorthand() -> None:
    parsed = parse_huggingface_source("owner/adapter@0123456789abcdef")
    assert parsed.repo_id == "owner/adapter"
    assert parsed.revision == "0123456789abcdef"


def test_local_safetensors_path(tmp_path: Path) -> None:
    local = tmp_path / "my adapter.safetensors"
    parsed = parse_huggingface_source(local)
    assert parsed.is_local
    assert parsed.local_path == local.resolve()
    assert parsed.weight_name == "my adapter.safetensors"


def test_non_model_huggingface_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="model repositories"):
        parse_huggingface_source("https://huggingface.co/datasets/owner/data")

