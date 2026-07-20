from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flowmorph_klein.config import BASE_MODEL_ID
from flowmorph_klein.errors import ManifestError
from flowmorph_klein.manifest import (
    InputManifest,
    get_manifest_pair,
    pair_config_overrides,
    validate_manifest,
)


def _manifest_data() -> dict:
    return {
        "project_name": "flowmorph_klein_full",
        "defaults": {
            "model_id": BASE_MODEL_ID,
            "profile": "auto",
            "width": 512,
            "height": 512,
            "frame_count": 20,
            "seed": 42,
            "guidance_scale": 4.0,
            "bridge_prompt": "shared bridge",
        },
        "pairs": [
            {
                "id": "pair_001",
                "source_image": "images/source.png",
                "target_image": "images/target.png",
                "source_prompt": "source-specific",
            }
        ],
    }


def _write_inputs(root: Path) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    (images / "source.png").write_bytes(b"source")
    (images / "target.png").write_bytes(b"target")


def test_manifest_resolves_relative_files_and_defaults(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest_data()), encoding="utf-8")

    manifest = validate_manifest(manifest_path)
    pair = get_manifest_pair(manifest, "pair_001")
    assert pair.source_image == (tmp_path / "images/source.png").resolve()
    assert pair.target_image == (tmp_path / "images/target.png").resolve()

    overrides = pair_config_overrides(manifest, pair)
    assert overrides["input"]["source_prompt"] == "source-specific"
    assert overrides["input"]["bridge_prompt"] == "shared bridge"
    assert overrides["model"]["id"] == BASE_MODEL_ID


def test_manifest_requires_at_least_one_pair() -> None:
    data = _manifest_data()
    data["pairs"] = []
    with pytest.raises(ManifestError, match="at least 1"):
        validate_manifest(data, require_files=False)


def test_manifest_rejects_duplicate_pair_ids() -> None:
    data = _manifest_data()
    data["pairs"].append(dict(data["pairs"][0]))
    with pytest.raises(ManifestError, match="duplicate manifest pair ids"):
        validate_manifest(data, require_files=False)


def test_manifest_rejects_missing_files_before_model_download(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="do not exist"):
        validate_manifest(_manifest_data(), base_dir=tmp_path)


def test_manifest_rejects_relative_path_traversal(tmp_path: Path) -> None:
    data = _manifest_data()
    data["pairs"][0]["source_image"] = "../outside.png"
    with pytest.raises(ManifestError, match="escapes its input root"):
        validate_manifest(data, base_dir=tmp_path, require_files=False)


def test_manifest_rejects_forbidden_model_and_unknown_fields() -> None:
    data = _manifest_data()
    data["defaults"]["model_id"] = "black-forest-labs/FLUX.2-klein-base-4B"
    with pytest.raises(ManifestError, match="Base 9B"):
        validate_manifest(data, require_files=False)

    data = _manifest_data()
    data["pairs"][0]["caption_with_extra_model"] = True
    with pytest.raises(ManifestError, match="Extra inputs are not permitted"):
        validate_manifest(data, require_files=False)


def test_manifest_pair_lookup_has_meaningful_error() -> None:
    manifest = InputManifest.model_validate(_manifest_data())
    with pytest.raises(ManifestError, match="no pair"):
        get_manifest_pair(manifest, "missing")
