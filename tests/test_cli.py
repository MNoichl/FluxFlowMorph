from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest
import yaml

from flowmorph_klein import cli


def test_cli_import_does_not_import_pipeline() -> None:
    sys.modules.pop("flowmorph_klein.pipeline", None)
    importlib.reload(cli)
    assert "flowmorph_klein.pipeline" not in sys.modules


def test_run_parser_accepts_documented_safe_overrides() -> None:
    args = cli.build_run_parser().parse_args(
        [
            "--config",
            "configs/full_9b_lora.yaml",
            "--source",
            "/content/source.png",
            "--target",
            "/content/target.png",
            "--lora-source",
            "org/repository",
            "--lora-scale",
            "0.8",
            "--profile",
            "a100_80gb_full",
        ]
    )
    assert args.source == "/content/source.png"
    assert args.target == "/content/target.png"
    assert args.lora_source == "org/repository"
    assert args.lora_scale == pytest.approx(0.8)
    assert args.profile == "a100_80gb_full"


def test_unknown_or_forbidden_profile_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        cli.build_run_parser().parse_args(["--profile", "klein_4b_fallback"])


def test_invalid_inputs_fail_before_pipeline_import(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({}), encoding="utf-8")
    sys.modules.pop("flowmorph_klein.pipeline", None)

    result = cli.run_command(
        [
            "--config",
            str(config_path),
            "--source",
            str(tmp_path / "missing-source.png"),
            "--target",
            str(tmp_path / "missing-target.png"),
            "--profile",
            "a100_80gb_full",
        ]
    )
    assert result == 1
    assert "flowmorph_klein.pipeline" not in sys.modules
    assert "before model download" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "expected_event"),
    [
        (cli.run_command, "run:False"),
        (cli.resume_command, "resume"),
        (cli.validate_colab_command, "probe"),
    ],
)
def test_runner_surfaces_use_facade_in_order(
    monkeypatch: pytest.MonkeyPatch,
    command,
    expected_event: str,
) -> None:
    events: list[str] = []

    class FakeRunner:
        archive_report = None
        run_directory = Path("/tmp/fake-run")

        @classmethod
        def from_config(cls, config):
            events.append("from_config")
            return cls()

        def prepare(self):
            events.append("prepare")

        def run_production_backward_probe(self):
            events.append("probe")
            return {"passed": True}

        def run(self, resume=False):
            events.append(f"run:{resume}")
            return None

        def resume(self):
            events.append("resume")
            return None

    fake_module = types.ModuleType("flowmorph_klein.pipeline")
    fake_module.FlowMorphRunner = FakeRunner
    monkeypatch.setitem(sys.modules, "flowmorph_klein.pipeline", fake_module)
    monkeypatch.setattr(cli, "resolve_runtime_config", lambda args: object())

    assert command([]) == 0
    assert events[:3] == ["from_config", "prepare", "probe"]
    assert expected_event in events
    if command is cli.validate_colab_command:
        assert not any(event.startswith("run:") for event in events)


def test_error_redaction_never_prints_hugging_face_secret(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
    result = cli._guarded(lambda: (_ for _ in ()).throw(RuntimeError(f"Authorization=Bearer {secret}")))
    captured = capsys.readouterr().err
    assert result == 1
    assert secret not in captured
    assert "redacted" in captured

    with pytest.raises(SystemExit):
        cli.build_run_parser().parse_args(["--unknown", secret])
    parsed_error = capsys.readouterr().err
    assert secret not in parsed_error
    assert "redacted" in parsed_error


def test_console_parser_has_all_surface_commands() -> None:
    for name in ("run", "resume", "validate-colab", "inspect-lora", "package-run"):
        if name == "package-run":
            args = cli.build_main_parser().parse_args([name, "--run-directory", "/tmp/run"])
        else:
            args = cli.build_main_parser().parse_args([name])
        assert args.command == name


def test_script_wrappers_keep_package_imports_inside_main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "scripts").glob("*.py")):
        if path.name not in {
            "run_flowmorph.py",
            "resume_flowmorph.py",
            "validate_colab.py",
            "inspect_lora.py",
            "package_run.py",
        }:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top_level_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
        assert top_level_imports == [], path.name
