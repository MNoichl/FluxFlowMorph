from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from flowmorph_klein.lora import (
    LoraValidationError,
    compare_lora_velocities,
    inspect_safetensors_keys,
    validate_flux2_klein_9b_lora,
    verify_active_adapter,
)


BASE_METADATA = {"base_model": "black-forest-labs/FLUX.2-klein-base-9B"}


def valid_state() -> dict[str, torch.Tensor]:
    return {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 4096),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(4096, 4),
    }


def test_valid_adapter_key_detection() -> None:
    report = validate_flux2_klein_9b_lora(valid_state(), metadata=BASE_METADATA)
    assert report.valid
    assert report.compatible_tensor_count == 2
    assert report.adapter_format == "diffusers"
    assert report.base_model_provenance == "flux2_klein_base_9b"


def test_safetensors_inspection_reads_metadata_and_shapes(tmp_path) -> None:
    path = tmp_path / "adapter.safetensors"
    save_file(valid_state(), str(path), metadata=BASE_METADATA)
    inspection = inspect_safetensors_keys(path)
    assert set(inspection.keys) == set(valid_state())
    assert inspection.metadata["base_model"].endswith("FLUX.2-klein-base-9B")
    assert inspection.shapes[
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
    ] == (4, 4096)


def test_zero_compatible_keys_causes_failure() -> None:
    unrelated = {
        "unet.down_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 4),
        "unet.down_blocks.0.attn.to_q.lora_B.weight": torch.zeros(4, 2),
    }
    with pytest.raises(LoraValidationError, match="zero compatible"):
        validate_flux2_klein_9b_lora(unrelated, metadata=BASE_METADATA)


@pytest.mark.parametrize("marker", ("loha", "lokr"))
def test_unsupported_adapter_formats_are_rejected(marker: str) -> None:
    state = {f"transformer.transformer_blocks.0.attn.to_q.{marker}.weight": torch.zeros(2, 2)}
    with pytest.raises(LoraValidationError, match="LoHa and LoKr"):
        validate_flux2_klein_9b_lora(state, metadata=BASE_METADATA)


def test_klein_4b_provenance_is_rejected() -> None:
    with pytest.raises(LoraValidationError, match="Klein 4B"):
        validate_flux2_klein_9b_lora(
            valid_state(),
            metadata={"base_model": "black-forest-labs/FLUX.2-klein-base-4B"},
        )


def test_distilled_9b_requires_explicit_override() -> None:
    metadata = {"base_model": "black-forest-labs/FLUX.2-klein-9B", "variant": "distilled"}
    with pytest.raises(LoraValidationError, match="distilled"):
        validate_flux2_klein_9b_lora(valid_state(), metadata=metadata)
    report = validate_flux2_klein_9b_lora(
        valid_state(),
        metadata=metadata,
        allow_distilled_9b=True,
    )
    assert "overridden" in " ".join(report.warnings)


def test_unknown_provenance_can_be_made_strict() -> None:
    with pytest.raises(LoraValidationError, match="provenance"):
        validate_flux2_klein_9b_lora(
            valid_state(),
            require_base_9b_provenance=True,
        )


class TinyTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=128,
            num_layers=8,
            num_single_layers=24,
            attention_head_dim=128,
            num_attention_heads=32,
            joint_attention_dim=12288,
            guidance_embeds=False,
        )
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.to_q = nn.Linear(4096, 4096, bias=False)
        self.transformer_blocks = nn.ModuleList([block])


def test_native_tensor_shapes_map_to_loaded_transformer() -> None:
    report = validate_flux2_klein_9b_lora(
        valid_state(),
        metadata=BASE_METADATA,
        transformer=TinyTarget(),
    )
    assert len(report.shape_verified_keys) == 2


def test_native_shape_mismatch_fails() -> None:
    state = valid_state()
    state["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"] = torch.zeros(4, 2048)
    with pytest.raises(LoraValidationError, match="shape mismatch"):
        validate_flux2_klein_9b_lora(
            state,
            metadata=BASE_METADATA,
            transformer=TinyTarget(),
        )


class FakePipeline:
    def get_active_adapters(self):
        return ["flowmorph_adapter"]

    def get_list_adapters(self):
        return {"transformer": ["flowmorph_adapter"]}


def test_active_adapter_name_is_reported() -> None:
    report = verify_active_adapter(FakePipeline())
    assert report
    assert report.adapter_name == "flowmorph_adapter"


def test_numerical_report_requires_output_change() -> None:
    baseline = torch.zeros(1, 3, 2)
    adapted = baseline.clone()
    adapted[..., 0] = 0.25
    report = compare_lora_velocities(baseline, adapted)
    assert report.changed
    assert report.maximum_absolute_difference == pytest.approx(0.25)
    assert report.mean_absolute_difference > 0
