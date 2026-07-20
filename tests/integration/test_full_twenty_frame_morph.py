from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from flowmorph_klein.acceptance import audit_completed_run
from flowmorph_klein.flow_schedule import build_flowmorph_schedule
from flowmorph_klein.flow_state import FlowMorphEndpoint
from flowmorph_klein.packaging import create_run_archive, validate_archive
from flowmorph_klein.renderer import render_morph


class _StableField:
    def predict_velocity(self, state, timestep, conditioning):
        del timestep, conditioning
        return state * 0.05 + 0.01


def test_offline_exact_twenty_frame_sparse_chain_and_archive(tmp_path) -> None:
    sigmas = torch.linspace(1.0, 0.01, 100, dtype=torch.float32)
    schedule = build_flowmorph_schedule(
        scheduler_points=100,
        image_seq_len=1024,
        timesteps=sigmas * 1000.0,
        sigmas=torch.cat((sigmas, torch.zeros(1))),
    )
    sigma_i = float(schedule.sigmas[35])
    source = FlowMorphEndpoint(
        z=torch.zeros(1, 8, 4),
        delta=torch.full((1, 8, 4), 0.05),
        u=torch.full((1, 8, 4), 0.1),
        sigma_i=sigma_i,
        sigma_last=0.0,
        timestep_i=schedule.timesteps[35],
    )
    target = FlowMorphEndpoint(
        z=torch.ones(1, 8, 4),
        delta=torch.full((1, 8, 4), -0.05),
        u=torch.full((1, 8, 4), -0.15),
        sigma_i=sigma_i,
        sigma_last=0.0,
        timestep_i=schedule.timesteps[35],
    )
    frames = render_morph(
        source,
        target,
        schedule=schedule,
        predictor=_StableField(),
        source_conditioning="source",
        target_conditioning="target",
        frame_count=20,
        render_indices=(35, 55, 75, 95),
    )
    assert len(frames) == 20
    assert frames[0].alpha == 0.0
    assert frames[-1].alpha == 1.0
    assert all(frame.final_latent.shape == (1, 8, 4) for frame in frames)
    assert all(torch.isfinite(frame.final_latent).all() for frame in frames)

    run = tmp_path / "offline_twenty"
    (run / "raw_frames").mkdir(parents=True)
    (run / "display_frames").mkdir(parents=True)
    (run / "config.resolved.yaml").write_text("frame_count: 20\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "offline_contract_only", "frame_count": 20}),
        encoding="utf-8",
    )
    for frame in frames:
        color = int(round(frame.alpha * 255))
        image = Image.new("RGB", (8, 8), (color, 64, 255 - color))
        image.save(run / "raw_frames" / f"frame_{frame.index:03d}.png")
        image.save(run / "display_frames" / f"frame_{frame.index:03d}.png")

    archive = create_run_archive(run, "offline_twenty")
    names = validate_archive(archive.path)
    assert archive.member_count == len(names)
    assert len([name for name in names if name.startswith("raw_frames/frame_")]) == 20
    assert len([name for name in names if name.startswith("display_frames/frame_")]) == 20
    assert "checksums.sha256" in names


@pytest.mark.integration
@pytest.mark.flux2_9b
@pytest.mark.requires_gated_access
@pytest.mark.full_endpoint
@pytest.mark.full_morph
def test_real_full_reference_twenty_frame_morph(integration_harness) -> None:
    integration_harness.require("FLOWMORPH_RUN_FULL_MORPH_INTEGRATION")

    from flowmorph_klein.pipeline import FlowMorphRunner

    with_lora = integration_harness.lora_source is not None
    config = integration_harness.config(
        run_mode="reference",
        frame_count=20,
        source_steps=100,
        target_steps=100,
        with_lora=with_lora,
    )
    runner = FlowMorphRunner.from_config(config)
    runner.prepare()
    runner.run()

    report = audit_completed_run(
        runner.run_directory,
        expected_frames=20,
        require_lora=with_lora,
    )
    assert report.passed, report.failures
    assert len(list((runner.run_directory / "raw_frames").glob("frame_*.png"))) == 20
    assert len(list((runner.run_directory / "display_frames").glob("frame_*.png"))) == 20
    archives = list((runner.run_directory / "artifacts").glob("*.flowmorph-klein.zip"))
    assert len(archives) == 1
    names = validate_archive(archives[0])
    assert "raw_frames/frame_019.png" in names
    assert "display_frames/frame_019.png" in names
