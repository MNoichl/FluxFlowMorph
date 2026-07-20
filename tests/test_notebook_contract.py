from __future__ import annotations

import ast
import json
import re
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "FlowMorph_FLUX2_Klein_Base_9B_LoRA_Colab.ipynb"

EXPECTED_SECTIONS = [
    "Runtime identification",
    "GPU and VRAM diagnostics",
    "Path configuration",
    "Repository clone or update",
    "Dependency installation",
    "Hugging Face access mode",
    "Model repository preflight",
    "Optional Google Drive mount",
    "Input upload or staging",
    "Run configuration",
    "Environment validation",
    "Model and LoRA loading",
    "Production backward probe",
    "Source endpoint fitting",
    "Source checkpoint confirmation",
    "Target endpoint fitting",
    "Target checkpoint confirmation",
    "Morph rendering",
    "Metrics",
    "Contact sheet and animation preview",
    "Archive construction",
    "Optional Drive persistence",
    "Final download",
]


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_notebook_is_valid_clean_nbformat() -> None:
    notebook = _load()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_notebook_has_all_23_recommended_sections_in_order() -> None:
    notebook = _load()
    headings: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        first_line = "".join(cell["source"]).splitlines()[0]
        match = re.fullmatch(r"## (\d+)\. (.+)", first_line)
        if match:
            assert int(match.group(1)) == len(headings) + 1
            headings.append(match.group(2))
    assert headings == EXPECTED_SECTIONS


def test_notebook_exposes_plain_exact_reference_variables() -> None:
    code = _code(_load())
    assert 'RUN_MODE = "experimental"' in code
    assert 'MODEL_ID = "Runware/BFL-FLUX.2-klein-base-9B"' in code
    assert "LORA_SOURCE = None" in code
    assert "RESOLUTION = 512" in code
    assert "GUIDANCE_SCALE = 4.0" in code
    assert "SEED = 42" in code
    for variable in (
        "PROJECT_ROOT",
        "INPUT_ROOT",
        "WORK_ROOT",
        "RESULT_ROOT",
        "HF_CACHE_DIR",
        "SOURCE_IMAGE",
        "TARGET_IMAGE",
        "SOURCE_PROMPT",
        "TARGET_PROMPT",
        "BRIDGE_PROMPT",
        "NEGATIVE_PROMPT",
        "PROFILE",
        "OUTPUT_NAME",
    ):
        assert re.search(rf"^{variable}\s*=", code, re.MULTILINE), variable


def test_notebook_can_generate_source_conditioned_runware_endpoints() -> None:
    code = _code(_load())
    assert "GENERATE_TEST_ENDPOINTS = True" in code
    assert "Flux2KleinPipeline.from_pretrained(" in code
    preview_load = code.split("Flux2KleinPipeline.from_pretrained(", 1)[1].split(
        "PREVIEW_PIPE.enable_model_cpu_offload()", 1
    )[0]
    assert "AUTHENTICATION" not in preview_load
    assert "token=" not in preview_load
    assert "prompt=SOURCE_GENERATION_PROMPT" in code
    assert "image=generated_source" in code
    assert "generated_young_woman.png" in code
    assert "generated_older_man.png" in code
    assert "PREVIEW_PIPE.maybe_free_model_hooks()" in code


def test_notebook_is_thin_and_contains_no_algorithm_implementation() -> None:
    code = _code(_load())
    tree = ast.parse(code)
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
    for forbidden in (
        "torch.optim",
        ".backward(",
        "delta_sigma",
        "pipe.transformer",
        "scheduler.step(",
        "load_lora_weights(",
        "slerp(",
    ):
        assert forbidden not in code
    assert "FlowMorphRunner.from_config(CONFIG)" in code
    assert "runner.prepare()" in code
    assert "runner.run_production_backward_probe()" in code
    assert "runner.run(resume=False)" in code


def test_config_resolves_before_pipeline_import() -> None:
    code = _code(_load())
    assert code.index("CONFIG = resolve_config") < code.index("from flowmorph_klein.pipeline import FlowMorphRunner")


def test_colab_only_features_are_guarded_and_plain_path_is_always_printed() -> None:
    code = _code(_load())
    assert "try:\n        from google.colab import drive" in code
    assert "try:\n        from google.colab import files" in code
    assert "except ImportError:" in code
    assert 'print("Final archive path:", archive_path)' in code
    assert "files.download(str(archive_path))" in code
    assert "Drive copy verified:" in code


def test_notebook_contains_no_literal_hugging_face_token() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert re.search(r"hf_[A-Za-z0-9]{20,}", text) is None
    assert "HF_TOKEN =" not in text
