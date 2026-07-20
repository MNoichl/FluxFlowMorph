from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from flowmorph_klein.colab_io import sha256_file
from flowmorph_klein.config import ProjectTemplateConfig, resolve_config
from flowmorph_klein.pipeline import FlowMorphRunner, PipelineError, _is_cuda_out_of_memory
from flowmorph_klein.acceptance import RunPhase
from flowmorph_klein.types import HardwareProfile, RunMode


def _resolved_config(tmp_path: Path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (19, 17), (220, 20, 10)).save(source)
    Image.new("RGB", (17, 19), (10, 20, 220)).save(target)
    template = ProjectTemplateConfig.model_validate(
        {
            "paths": {
                "result_root": tmp_path / "results",
                "work_root": tmp_path / "work",
                "input_root": tmp_path,
                "hf_cache": tmp_path / "cache",
                "drive_root": None,
            },
            "input": {"source_image": source, "target_image": target},
        }
    )
    return resolve_config(
        template,
        selected_profile=HardwareProfile.A100_80GB_FULL,
    )


def test_resume_rejects_changed_input_before_overwriting_staged_evidence(
    tmp_path: Path,
) -> None:
    config = _resolved_config(tmp_path)
    run_directory = tmp_path / "results" / "existing-run"
    original = FlowMorphRunner.from_config(config, run_directory=run_directory)
    original._prepare_inputs()
    staged_original = run_directory / "inputs" / "source_original.png"
    staged_processed = run_directory / "inputs" / "source_preprocessed.png"
    evidence_before = (sha256_file(staged_original), sha256_file(staged_processed))

    Image.new("RGB", (19, 17), (1, 2, 3)).save(config.input.source_image)
    resumed = FlowMorphRunner.from_config(config, run_directory=run_directory)
    with pytest.raises(PipelineError, match="before staging"):
        resumed.prepare(resume=True)

    assert (sha256_file(staged_original), sha256_file(staged_processed)) == evidence_before


def test_resume_reuses_verified_preprocessed_images_without_rewriting(
    tmp_path: Path,
) -> None:
    config = _resolved_config(tmp_path)
    run_directory = tmp_path / "results" / "existing-run"
    original = FlowMorphRunner.from_config(config, run_directory=run_directory)
    original._prepare_inputs()
    staged_processed = run_directory / "inputs" / "source_preprocessed.png"
    modified_before = staged_processed.stat().st_mtime_ns

    resumed = FlowMorphRunner.from_config(config, run_directory=run_directory)
    resumed._validate_resume_artifacts_unchanged()
    resumed._prepare_inputs(reuse_persisted=True)

    assert staged_processed.stat().st_mtime_ns == modified_before
    assert resumed.source_preprocessed is not None
    assert resumed.source_preprocessed.image.size == (512, 512)


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (RuntimeError("CUDA out of memory"), True),
        (RuntimeError("CPU out of memory"), False),
        (ValueError("unrelated"), False),
    ),
)
def test_cuda_oom_classification(error: BaseException, expected: bool) -> None:
    assert _is_cuda_out_of_memory(error) is expected


def test_public_resume_routes_prepare_through_explicit_resume_mode() -> None:
    class PrepareSentinel(RuntimeError):
        pass

    runner = object.__new__(FlowMorphRunner)
    runner.config = SimpleNamespace(run_mode=RunMode.REFERENCE)
    runner.phase = RunPhase.CREATED
    runner._prepared = False
    runner._restore_failed_phase_for_resume = lambda: None
    runner._record_failure = lambda error, operation: None
    observed: list[bool] = []

    def prepare(*, resume: bool = False):
        observed.append(resume)
        raise PrepareSentinel

    runner.prepare = prepare
    with pytest.raises(PrepareSentinel):
        runner.resume()

    assert observed == [True]


def test_fresh_resume_session_reruns_backward_probe_despite_durable_phase() -> None:
    class EndpointSentinel(RuntimeError):
        pass

    runner = object.__new__(FlowMorphRunner)
    runner.config = SimpleNamespace(
        run_mode=RunMode.REFERENCE,
        memory=SimpleNamespace(run_production_backward_probe=True),
        lora=SimpleNamespace(fit_scale=1.0),
    )
    runner.phase = RunPhase.TARGET_CHECKPOINTED
    runner._prepared = True
    runner._session_backward_probe_report = None
    runner._restore_failed_phase_for_resume = lambda: None
    runner._record_failure = lambda error, operation: None
    runner._set_lora_scale = lambda scale: None
    calls: list[str] = []

    def probe():
        calls.append("probe")
        runner._session_backward_probe_report = object()

    def fit_endpoint(label: str, *, resume: bool):
        assert label == "source"
        assert resume is True
        raise EndpointSentinel

    runner.run_production_backward_probe = probe
    runner._fit_endpoint = fit_endpoint
    with pytest.raises(EndpointSentinel):
        runner.resume()

    assert calls == ["probe"]
