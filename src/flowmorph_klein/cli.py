"""Safe user-facing command-line surfaces.

This module intentionally imports neither ``pipeline`` nor heavyweight model
dependencies at import time.  Configuration and endpoint files are validated
before the runner facade is imported, so a bad invocation cannot begin a 9B
model download.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_CONFIG = "configs/full_9b_lora.yaml"
PROFILE_CHOICES = (
    "auto",
    "a100_80gb_full",
    "a100_40gb_checkpointed",
    "fp8_9b_experimental",
    "unsupported_low_vram",
)
_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{8,}")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)((?:hf[_-]?token|huggingface[_-]?token|authorization)\s*[:=]\s*)[^\s,;]+"
)
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def redact_secrets(value: Any) -> str:
    """Return a printable message with common Hugging Face secrets removed."""

    text = str(value)
    text = _HF_TOKEN.sub("<redacted-hf-token>", text)
    text = _SECRET_ASSIGNMENT.sub(r"\1<redacted>", text)
    return _BEARER.sub(r"\1<redacted>", text)


class SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that redacts values echoed by parse failures."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {redact_secrets(message)}\n")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    return redact_secrets(value)


def _serialized_report(value: Any) -> str:
    return redact_secrets(json.dumps(_jsonable(value), indent=2, sort_keys=True, default=str))


def _emit_report(value: Any, output: str | Path | None = None) -> None:
    serialized = _serialized_report(value) + "\n"
    if output is None:
        print(serialized, end="")
        return
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    print(f"Report written to {redact_secrets(destination)}")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML configuration template (default: %(default)s)",
    )
    parser.add_argument("--source", help="source endpoint image; overrides input.source_image")
    parser.add_argument("--target", help="target endpoint image; overrides input.target_image")
    parser.add_argument(
        "--lora-source",
        help="optional adapter repository, Hugging Face URL, resolve URL, or local safetensors",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        help="adapter scale used identically for fitting and rendering",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        help="explicit 9B hardware profile; auto never selects FP8 or a smaller model",
    )
    parser.add_argument("--source-prompt", help="source fitting prompt")
    parser.add_argument("--target-prompt", help="target fitting prompt")
    parser.add_argument("--bridge-prompt", help="shared fallback/bridge prompt")
    parser.add_argument("--negative-prompt", help="unconditional CFG prompt")


def build_run_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="run_flowmorph.py",
        description="Run the complete FLUX.2 Klein Base 9B FlowMorph workflow.",
    )
    _add_config_arguments(parser)
    return parser


def build_resume_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="resume_flowmorph.py",
        description="Resume compatible FlowMorph endpoint checkpoints.",
    )
    _add_config_arguments(parser)
    parser.add_argument(
        "--run-directory",
        help=(
            "existing compatible run directory; when omitted, the newest compatible "
            "checkpoint under paths.result_root is selected"
        ),
    )
    return parser


def build_validate_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="validate_colab.py",
        description="Prepare the exact 9B stack and run the production-shape backward probe.",
    )
    _add_config_arguments(parser)
    return parser


def build_inspect_lora_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="inspect_lora.py",
        description="Resolve and statically validate a FLUX.2 Klein Base 9B LoRA.",
    )
    _add_config_arguments(parser)
    parser.add_argument("--lora-revision", help="adapter repository revision override")
    parser.add_argument("--lora-weight-name", help="specific safetensors filename")
    parser.add_argument(
        "--allow-distilled-9b",
        action="store_true",
        help="explicitly allow a distilled-9B adapter warning instead of rejecting it",
    )
    parser.add_argument("--output", help="optional JSON report path")
    return parser


def build_package_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="package_run.py",
        description="Build and verify one compact .flowmorph-klein.zip archive.",
    )
    parser.add_argument("--run-directory", required=True, help="completed run directory")
    parser.add_argument("--run-id", help="archive basename; defaults to directory name")
    return parser


def _configuration_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    aliases = {
        "source": "input.source_image",
        "target": "input.target_image",
        "lora_source": "lora.source",
        "source_prompt": "input.source_prompt",
        "target_prompt": "input.target_prompt",
        "bridge_prompt": "input.bridge_prompt",
        "negative_prompt": "input.negative_prompt",
    }
    for attribute, destination in aliases.items():
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[destination] = value
    lora_scale = getattr(args, "lora_scale", None)
    if lora_scale is not None:
        overrides["lora.fit_scale"] = lora_scale
        overrides["lora.render_scale"] = lora_scale
    return overrides


def select_hardware_profile(requested: Any = "auto") -> str:
    """Resolve ``auto`` using local CUDA facts without downloading a model."""

    value = getattr(requested, "value", requested) or "auto"
    if value not in PROFILE_CHOICES:
        raise ValueError(f"unknown hardware profile {value!r}")
    if value != "auto":
        return str(value)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "profile auto cannot select a production profile because CUDA is unavailable; "
            "the full FLUX.2 Klein Base 9B run is not supported on CPU"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("profile auto requires CUDA BF16 support for the Base-9B reference run")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / (1024**3)
    if total_gib >= 60:
        return "a100_80gb_full"
    if total_gib >= 35:
        return "a100_40gb_checkpointed"
    raise RuntimeError(
        f"profile auto found only {total_gib:.1f} GiB VRAM; no validated Base-9B production "
        "profile is available and no 4B/distilled fallback will be selected"
    )


def resolve_runtime_config(args: argparse.Namespace) -> Any:
    """Validate CLI/YAML inputs completely before importing the model runner."""

    from flowmorph_klein.config import load_config, resolve_config

    template = load_config(args.config, overrides=_configuration_overrides(args))
    selected_profile = select_hardware_profile(args.profile or template.model.profile)
    return resolve_config(template, selected_profile=selected_profile, check_input_files=True)


def _runner_summary(runner: Any, action: str, result: Any = None) -> dict[str, Any]:
    archive = getattr(runner, "archive_report", None)
    return {
        "action": action,
        "status": "completed",
        "run_directory": getattr(runner, "run_directory", None),
        "archive": archive,
        "result": result,
    }


def _discover_resume_directory(config: Any) -> Path:
    """Find the newest checkpointed run with the exact resolved config hash."""

    from flowmorph_klein.config import canonical_config_hash

    root = Path(config.paths.result_root).expanduser().resolve(strict=False)
    expected_hash = canonical_config_hash(config)
    candidates: list[tuple[float, Path]] = []
    if root.is_dir():
        for manifest_path in root.glob("*/run_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            directory = manifest_path.parent
            has_checkpoint = any(
                (directory / "checkpoints" / label / "tensors.safetensors").is_file()
                for label in ("source", "target")
            )
            if manifest.get("config_hash") == expected_hash and has_checkpoint:
                candidates.append((manifest_path.stat().st_mtime, directory))
    if not candidates:
        raise FileNotFoundError(
            f"no compatible checkpointed run was found under {root}; pass --run-directory"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _execute_runner(args: argparse.Namespace, action: str) -> int:
    config = resolve_runtime_config(args)
    # This import must remain after config/path validation.
    from flowmorph_klein.pipeline import FlowMorphRunner

    run_directory = getattr(args, "run_directory", None)
    if action == "resume" and run_directory is None and hasattr(config, "paths"):
        run_directory = _discover_resume_directory(config)
    runner = (
        FlowMorphRunner.from_config(config, run_directory=run_directory)
        if run_directory is not None
        else FlowMorphRunner.from_config(config)
    )
    if action == "resume" and run_directory is not None:
        runner.prepare(resume=True)
    else:
        runner.prepare()
    probe = runner.run_production_backward_probe()
    if action == "validate":
        _emit_report(_runner_summary(runner, "production_backward_probe", probe))
        return 0
    if action == "resume":
        result = runner.resume()
    else:
        result = runner.run(resume=False)
    _emit_report(_runner_summary(runner, action, result))
    return 0


def _guarded(operation: Callable[[], int]) -> int:
    try:
        return operation()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {redact_secrets(error)}", file=sys.stderr)
        return 1


def run_command(argv: Sequence[str] | None = None) -> int:
    args = build_run_parser().parse_args(argv)
    return _guarded(lambda: _execute_runner(args, "run"))


def resume_command(argv: Sequence[str] | None = None) -> int:
    args = build_resume_parser().parse_args(argv)
    return _guarded(lambda: _execute_runner(args, "resume"))


def validate_colab_command(argv: Sequence[str] | None = None) -> int:
    args = build_validate_parser().parse_args(argv)
    return _guarded(lambda: _execute_runner(args, "validate"))


def _inspect_lora(args: argparse.Namespace) -> int:
    from flowmorph_klein.config import load_config

    template = load_config(args.config, overrides=_configuration_overrides(args))
    source = args.lora_source or template.lora.source
    if source is None:
        raise ValueError("no LoRA source was supplied; use --lora-source or set lora.source")

    from flowmorph_klein.environment import resolve_hf_token
    from flowmorph_klein.hf_assets import parse_huggingface_source, resolve_huggingface_file
    from flowmorph_klein.lora import (
        compute_adapter_fingerprint,
        inspect_safetensors_keys,
        validate_flux2_klein_9b_lora,
    )

    parsed = parse_huggingface_source(source)
    token = None
    authentication_source = "not_required"
    if not parsed.is_local:
        authentication = resolve_hf_token()
        token = authentication.token
        authentication_source = authentication.source
    resolved = resolve_huggingface_file(
        parsed,
        revision=args.lora_revision or template.lora.revision,
        subfolder=template.lora.subfolder,
        weight_name=args.lora_weight_name or template.lora.weight_name,
        token=token,
        cache_dir=template.paths.hf_cache,
    )
    inspection = inspect_safetensors_keys(resolved.local_path)
    validation = validate_flux2_klein_9b_lora(
        inspection,
        metadata=getattr(inspection, "metadata", None),
        require_base_9b_provenance=template.lora.require_base_9b_compatibility,
        allow_distilled_9b=args.allow_distilled_9b,
    )
    report = {
        "source": parsed,
        "resolved": resolved,
        "inspection": inspection,
        "validation": validation,
        "fingerprint": compute_adapter_fingerprint(resolved.local_path),
        "authentication_source": authentication_source,
    }
    _emit_report(report, args.output)
    return 0


def inspect_lora_command(argv: Sequence[str] | None = None) -> int:
    args = build_inspect_lora_parser().parse_args(argv)
    return _guarded(lambda: _inspect_lora(args))


def _package_run(args: argparse.Namespace) -> int:
    from flowmorph_klein.packaging import create_run_archive

    run_directory = Path(args.run_directory).expanduser().resolve(strict=False)
    run_id = args.run_id or run_directory.name
    report = create_run_archive(run_directory, run_id)
    _emit_report({"status": "completed", "archive": report})
    return 0


def package_run_command(argv: Sequence[str] | None = None) -> int:
    args = build_package_parser().parse_args(argv)
    return _guarded(lambda: _package_run(args))


def build_main_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(prog="flowmorph-klein")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("run", "run the complete workflow"),
        ("resume", "resume compatible checkpoints"),
        ("validate-colab", "run the production backward probe"),
        ("inspect-lora", "inspect and validate an adapter"),
        ("package-run", "build a compact run archive"),
    ):
        subparser = subparsers.add_parser(name, help=description)
        if name == "package-run":
            subparser.add_argument("--run-directory", required=True)
            subparser.add_argument("--run-id")
        else:
            _add_config_arguments(subparser)
            if name == "resume":
                subparser.add_argument(
                    "--run-directory",
                    help="existing compatible run directory (otherwise auto-discover newest)",
                )
            if name == "inspect-lora":
                subparser.add_argument("--lora-revision")
                subparser.add_argument("--lora-weight-name")
                subparser.add_argument("--allow-distilled-9b", action="store_true")
                subparser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_main_parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "run": lambda value: _execute_runner(value, "run"),
        "resume": lambda value: _execute_runner(value, "resume"),
        "validate-colab": lambda value: _execute_runner(value, "validate"),
        "inspect-lora": _inspect_lora,
        "package-run": _package_run,
    }
    return _guarded(lambda: handlers[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
