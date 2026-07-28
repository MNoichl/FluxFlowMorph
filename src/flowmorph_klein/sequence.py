"""Sequence-native FlowMorph fitting and rendering.

The research runner deliberately owns one source/target experiment.  Art loops
have a different shape: every image participates in two neighboring gaps, so
loading a pipeline and fitting that image twice is needless.  This module
keeps one prepared :class:`FlowMorphRunner`, fits each unique endpoint into a
durable cache, and renders any number of interior pair positions from those
cached endpoint states.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from PIL import Image

from .checkpoints import (
    LoadedCheckpoint,
    load_endpoint_checkpoint,
    save_endpoint_checkpoint,
    unflatten_optimizer_state,
)
from .conditioning import (
    ConditioningPackage,
    encode_prompt_conditioning,
    interpolate_conditioning_through_midpoint,
    stack_conditioning_packages,
)
from .diagnostics import release_cuda_memory
from .endpoint_optimizer import (
    EndpointOptimizationResult,
    EndpointOptimizerConfig,
    OptimizationStepDiagnostics,
    optimize_endpoint,
    optimize_endpoint_batch,
)
from .flow_schedule import get_render_chain, get_start_state_metadata
from .flow_state import FlowMorphEndpoint
from .flux2_latents import (
    decode_packed_latent,
    encode_image_to_packed_latent,
    preprocess_endpoint_image as flux2_preprocess_endpoint_image,
)
from .image_io import preprocess_endpoint_image
from .pipeline import FlowMorphRunner, PipelineError, _module_dtype, _move_module
from .renderer import RenderedLatentFrame, render_latent_trajectory, render_morph
from .types import PreprocessedImage, RenderConditioningMode


@dataclass(frozen=True, slots=True)
class EncodedSequenceImage:
    """A persisted RGB input and its deterministic packed FLUX.2 latent."""

    key: str
    preprocessed: PreprocessedImage
    latent: torch.Tensor


@dataclass(frozen=True, slots=True)
class SequenceEndpointResult:
    """One reusable endpoint state plus compact fitting provenance."""

    endpoint: FlowMorphEndpoint
    checkpoint_directory: Path
    resumed: bool
    completed_steps: int
    elapsed_seconds: float | None


@dataclass(frozen=True, slots=True)
class SequenceEndpointRequest:
    """One endpoint fit request suitable for grouped sequence optimization."""

    endpoint_key: str
    asset: EncodedSequenceImage
    conditioning: ConditioningPackage
    checkpoint_directory: Path
    resume: bool = True


def _write_history(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FlowMorphSequenceSession:
    """Reuse one prepared model, backward probe, and endpoint cache."""

    def __init__(
        self,
        runner: FlowMorphRunner,
        *,
        render_batch_size: int = 1,
        decode_batch_size: int = 1,
        cfg_execution: str | None = None,
        oom_backoff: bool = True,
    ) -> None:
        if not runner._prepared:  # The runner owns validated model preparation.
            raise PipelineError("FlowMorphSequenceSession requires a prepared runner")
        if render_batch_size < 1 or decode_batch_size < 1:
            raise ValueError("sequence batch sizes must be positive")
        if cfg_execution not in {None, "sequential", "batched"}:
            raise ValueError("cfg_execution must be sequential, batched, or None")
        runner._require_prepared_values()
        self.runner = runner
        self.render_batch_size = int(render_batch_size)
        self.decode_batch_size = int(decode_batch_size)
        self.cfg_execution = cfg_execution
        self.oom_backoff = bool(oom_backoff)
        self.last_render_batch_size = 1
        self.last_decode_batch_size = 1
        self.endpoint_batch_size_limit: int | None = None

    @property
    def device(self) -> torch.device:
        if self.runner.device is None:
            raise PipelineError("prepared runner has no execution device")
        return self.runner.device

    def run_backward_probe_once(self):
        """Execute the runner's production probe once for this live model."""

        return self.runner.run_production_backward_probe()

    def seed_prepared_assets(
        self,
        source_key: str,
        target_key: str,
    ) -> tuple[dict[str, EncodedSequenceImage], dict[str, ConditioningPackage]]:
        """Expose the two images/prompts already encoded during preparation."""

        runner = self.runner
        if any(
            value is None
            for value in (
                runner.source_preprocessed,
                runner.target_preprocessed,
                runner.source_latent,
                runner.target_latent,
                runner.conditioning_cache,
            )
        ):
            raise PipelineError("prepared runner is missing its seed assets")
        assert runner.source_preprocessed is not None
        assert runner.target_preprocessed is not None
        assert runner.source_latent is not None
        assert runner.target_latent is not None
        assert runner.conditioning_cache is not None
        images = {
            source_key: EncodedSequenceImage(
                source_key,
                runner.source_preprocessed,
                runner.source_latent.detach().cpu(),
            ),
            target_key: EncodedSequenceImage(
                target_key,
                runner.target_preprocessed,
                runner.target_latent.detach().cpu(),
            ),
        }
        prompts = {
            str(runner.conditioning_cache.source.prompt): runner.conditioning_cache.source.cpu(),
            str(runner.conditioning_cache.target.prompt): runner.conditioning_cache.target.cpu(),
        }
        return images, prompts

    def encode_missing_assets(
        self,
        *,
        prompts: Sequence[str],
        images: Mapping[str, tuple[str | Path, str | Path]],
    ) -> tuple[dict[str, ConditioningPackage], dict[str, EncodedSequenceImage]]:
        """Encode a round's missing prompts and images with one component swap."""

        runner = self.runner
        pipeline = runner.pipeline
        if pipeline is None or runner.image_ids is None:
            raise PipelineError("prepared runner lacks pipeline/image IDs")
        transformer = pipeline.transformer
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        prompt_results: dict[str, ConditioningPackage] = {}
        image_results: dict[str, EncodedSequenceImage] = {}

        _move_module(transformer, "cpu")
        release_cuda_memory()
        try:
            if prompts:
                _move_module(text_encoder, self.device)
                for prompt in dict.fromkeys(prompts):
                    prompt_results[prompt] = encode_prompt_conditioning(
                        pipeline,
                        prompt,
                        device=self.device,
                    ).cpu()
                _move_module(text_encoder, "cpu")
                release_cuda_memory()

            if images:
                _move_module(vae, self.device)
                vae_dtype = _module_dtype(
                    vae,
                    next(transformer.parameters()).dtype,
                )
                for key, (source, output_path) in images.items():
                    preprocessed = preprocess_endpoint_image(
                        source,
                        width=runner.config.input.width,
                        height=runner.config.input.height,
                        resize_mode=runner.config.input.resize_mode,
                        output_path=output_path,
                        divisibility=16,
                    )
                    tensor = flux2_preprocess_endpoint_image(
                        preprocessed.image,
                        pipeline.image_processor,
                        height=runner.config.input.height,
                        width=runner.config.input.width,
                        resize_mode="default",
                    ).to(device=self.device, dtype=vae_dtype)
                    with torch.inference_mode():
                        latent, image_ids = encode_image_to_packed_latent(
                            tensor,
                            vae,
                            preprocessed=True,
                        )
                    if not torch.equal(image_ids.detach().cpu(), runner.image_ids.detach().cpu()):
                        raise PipelineError(f"sequence image IDs differ for endpoint {key!r}")
                    image_results[key] = EncodedSequenceImage(
                        key,
                        preprocessed,
                        latent.detach().cpu(),
                    )
                    del tensor, latent, image_ids
                _move_module(vae, "cpu")
                release_cuda_memory()
        finally:
            _move_module(text_encoder, "cpu")
            _move_module(vae, "cpu")
            _move_module(transformer, self.device)
            release_cuda_memory()
        return prompt_results, image_results

    def _endpoint_metadata(
        self,
        endpoint_key: str,
        asset: EncodedSequenceImage,
        conditioning: ConditioningPackage,
        settings: EndpointOptimizerConfig,
    ) -> dict[str, object]:
        runner = self.runner
        if runner.schedule is None:
            raise PipelineError("prepared runner lacks a FlowMorph schedule")
        if asset.preprocessed.output_path is None:
            raise PipelineError("sequence endpoint preprocessing was not persisted")
        start = get_start_state_metadata(
            runner.schedule,
            runner.config.flowmorph.start_timestep_index,
        )
        lora_source = None
        lora_revision = None
        lora_sha = None
        if runner.lora_load_report is not None:
            lora_source = runner.lora_load_report.source.repo_id or "local_safetensors"
            lora_revision = runner.lora_load_report.source.resolved_revision
            lora_sha = runner.lora_load_report.source.sha256
        from . import DIFFUSERS_COMMIT, FLOWMORPH_COMMIT, FLUX2_COMMIT
        from .colab_io import sha256_file

        return {
            "schema_version": 1,
            "endpoint": endpoint_key,
            "model_id": runner.config.model.id,
            "model_revision": runner.config.model.revision,
            "lora_source": lora_source,
            "lora_revision": lora_revision,
            "lora_file_sha256": lora_sha,
            "lora_scale": runner.config.lora.fit_scale if lora_source else None,
            "prompt_checksum": conditioning.prompt_sha256,
            "source_image_checksum": asset.preprocessed.original_sha256,
            "processed_image_checksum": sha256_file(asset.preprocessed.output_path),
            "preprocessing_hash": asset.preprocessed.preprocessing_sha256,
            "resize_mode": asset.preprocessed.resize_mode.value,
            "scheduler_configuration": {
                "config": runner.schedule.scheduler_configuration,
                "timesteps": runner.schedule.timesteps.detach().cpu().float().tolist(),
                "sigmas": runner.schedule.sigmas.detach().cpu().float().tolist(),
                "mu": runner.schedule.mu,
                "image_seq_len": runner.schedule.image_seq_len,
            },
            "start_timestep_index": runner.config.flowmorph.start_timestep_index,
            "timestep_i": start.timestep_i,
            "sigma_i": start.sigma_i,
            "sigma_last": start.sigma_last,
            "latent_shape": list(asset.latent.shape),
            "optimizer_configuration": {
                "name": runner.config.flowmorph.optimizer.value,
                "pred_learning_rate": settings.pred_learning_rate,
                "u_learning_rate": settings.u_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            "loss_mode": str(settings.loss_mode.value),
            "guidance_configuration": {
                "enabled": runner.config.guidance.enabled,
                "scale": runner.config.guidance.scale,
                "execution": runner.config.guidance.execution.value,
            },
            "sequence_execution_batching": {
                "cfg_execution_override": self.cfg_execution,
                "render_batch_size": self.render_batch_size,
                "decode_batch_size": self.decode_batch_size,
                "oom_backoff": self.oom_backoff,
            },
            "precision_configuration": {
                "transformer": runner.config.model.transformer_compute_dtype.value,
                "endpoint_parameters": runner.config.model.optimization_parameter_dtype.value,
                "quantization": runner.config.model.quantization.value,
            },
            "diffusers_commit": DIFFUSERS_COMMIT,
            "flowmorph_commit": FLOWMORPH_COMMIT,
            "flux2_commit": FLUX2_COMMIT,
            "optimization_steps_required": settings.optimization_steps,
        }

    def fit_endpoint(
        self,
        *,
        endpoint_key: str,
        asset: EncodedSequenceImage,
        conditioning: ConditioningPackage,
        checkpoint_directory: str | Path,
        resume: bool,
    ) -> SequenceEndpointResult:
        """Fit or resume one endpoint without reloading the model."""

        runner = self.runner
        if runner.schedule is None or runner.model is None:
            raise PipelineError("prepared runner lacks model/schedule")
        runner._set_lora_scale(runner.config.lora.fit_scale)
        settings = EndpointOptimizerConfig(
            optimization_steps=runner.config.flowmorph.optimization_steps_source,
            pred_learning_rate=runner.config.flowmorph.pred_learning_rate,
            u_learning_rate=runner.config.flowmorph.u_learning_rate,
            weight_decay=runner.config.flowmorph.weight_decay,
            loss_mode=runner.config.flowmorph.loss_mode,
            checkpoint_every=runner.config.flowmorph.checkpoint_every,
        )
        directory = Path(checkpoint_directory)
        history_path = directory / "optimization.csv"
        metadata = self._endpoint_metadata(endpoint_key, asset, conditioning, settings)
        loaded: LoadedCheckpoint | None = None
        initial_delta = None
        initial_u = None
        optimizer_state = None
        start_step = 0
        previous_rows = runner._read_csv_rows(history_path)
        if directory.exists():
            if not resume:
                raise PipelineError(f"endpoint checkpoint already exists for {endpoint_key!r}; enable resume")
            loaded = load_endpoint_checkpoint(
                directory,
                expected_metadata=metadata,
                device="cpu",
            )
            start_step = int(loaded.metadata.get("completed_steps", 0))
            if start_step >= settings.optimization_steps:
                endpoint = FlowMorphEndpoint(
                    z=loaded.tensors["z"].float(),
                    delta=loaded.tensors["delta"].float(),
                    u=loaded.tensors["u"].float(),
                    sigma_i=float(metadata["sigma_i"]),
                    sigma_last=float(metadata["sigma_last"]),
                    timestep_i=float(metadata["timestep_i"]),
                )
                elapsed = float(previous_rows[-1]["elapsed_seconds"]) if previous_rows else None
                return SequenceEndpointResult(endpoint, directory, True, start_step, elapsed)
            descriptor = loaded.metadata.get("optimizer_state")
            if not loaded.metadata.get("optimizer_state_saved") or not isinstance(descriptor, Mapping):
                raise PipelineError(f"incomplete endpoint {endpoint_key!r} has no optimizer state")
            initial_delta = loaded.tensors["delta"]
            initial_u = loaded.tensors["u"]
            optimizer_state = unflatten_optimizer_state(loaded.tensors, descriptor)

        z = (loaded.tensors["z"] if loaded is not None else asset.latent).to(
            self.device,
            dtype=torch.float32,
        )
        conditional = conditioning.to(self.device)
        predictor = runner._bound_predictor()
        if self.cfg_execution is not None:
            predictor.cfg_execution = self.cfg_execution
        new_rows: list[dict[str, object]] = []

        def diagnostics_callback(diagnostics: OptimizationStepDiagnostics) -> None:
            new_rows.append(diagnostics.to_dict())

        def checkpoint_callback(
            step: int,
            endpoint: FlowMorphEndpoint,
            optimizer: torch.optim.Optimizer,
            diagnostics: OptimizationStepDiagnostics,
        ) -> None:
            checkpoint_metadata = dict(metadata)
            checkpoint_metadata["completed_steps"] = step
            save_optimizer = step < settings.optimization_steps
            save_endpoint_checkpoint(
                directory,
                {"z": endpoint.z, "delta": endpoint.delta, "u": endpoint.u},
                checkpoint_metadata,
                optimizer=optimizer if save_optimizer else None,
            )
            _write_history(history_path, [*previous_rows, *new_rows])

        start = get_start_state_metadata(
            runner.schedule,
            runner.config.flowmorph.start_timestep_index,
        )
        try:
            result: EndpointOptimizationResult = optimize_endpoint(
                z,
                sigma_i=start.sigma_i,
                sigma_last=start.sigma_last,
                timestep_i=runner.schedule.timesteps[
                    runner.config.flowmorph.start_timestep_index
                ].to(self.device),
                predictor=predictor,
                conditioning=conditional,
                config=settings,
                initial_delta=initial_delta,
                initial_u=initial_u,
                optimizer_state_dict=optimizer_state,
                start_step=start_step,
                predictor_parameters=runner.model.parameters(),
                checkpoint_callback=checkpoint_callback,
                diagnostics_callback=diagnostics_callback,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            if not (
                self.oom_backoff
                and is_oom
                and predictor.cfg_execution == "batched"
            ):
                raise
            self.cfg_execution = "sequential"
            z = torch.empty(0)
            conditional = None
            predictor = None
            release_cuda_memory()
            print("Endpoint fit OOM with batched CFG; retrying with sequential CFG")
            return self.fit_endpoint(
                endpoint_key=endpoint_key,
                asset=asset,
                conditioning=conditioning,
                checkpoint_directory=directory,
                resume=directory.exists(),
            )
        rows = [*previous_rows, *[item.to_dict() for item in result.diagnostics]]
        _write_history(history_path, rows)
        endpoint = result.endpoint.to(device="cpu", dtype=torch.float32)
        elapsed = float(rows[-1]["elapsed_seconds"]) if rows else None
        del z, conditional, predictor, result
        release_cuda_memory()
        return SequenceEndpointResult(
            endpoint,
            directory,
            start_step > 0,
            settings.optimization_steps,
            elapsed,
        )

    def fit_endpoints(
        self,
        requests: Sequence[SequenceEndpointRequest],
        *,
        batch_size: int = 1,
    ) -> dict[str, SequenceEndpointResult]:
        """Fit new endpoints in independent batches and reuse/resume existing ones."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len({request.endpoint_key for request in requests}) != len(requests):
            raise ValueError("endpoint request keys must be unique")
        results: dict[str, SequenceEndpointResult] = {}
        new_requests: list[SequenceEndpointRequest] = []
        for request in requests:
            directory = Path(request.checkpoint_directory)
            if directory.exists() or batch_size == 1:
                results[request.endpoint_key] = self.fit_endpoint(
                    endpoint_key=request.endpoint_key,
                    asset=request.asset,
                    conditioning=request.conditioning,
                    checkpoint_directory=directory,
                    resume=request.resume,
                )
            else:
                new_requests.append(request)

        runner = self.runner
        if not new_requests:
            return results
        if runner.schedule is None or runner.model is None:
            raise PipelineError("prepared runner lacks model/schedule")
        runner._set_lora_scale(runner.config.lora.fit_scale)
        settings = EndpointOptimizerConfig(
            optimization_steps=runner.config.flowmorph.optimization_steps_source,
            pred_learning_rate=runner.config.flowmorph.pred_learning_rate,
            u_learning_rate=runner.config.flowmorph.u_learning_rate,
            weight_decay=runner.config.flowmorph.weight_decay,
            loss_mode=runner.config.flowmorph.loss_mode,
            checkpoint_every=runner.config.flowmorph.checkpoint_every,
        )
        start = get_start_state_metadata(
            runner.schedule,
            runner.config.flowmorph.start_timestep_index,
        )

        effective_batch_size = min(
            batch_size,
            self.endpoint_batch_size_limit or batch_size,
        )
        for chunk_start in range(0, len(new_requests), effective_batch_size):
            chunk = new_requests[chunk_start : chunk_start + effective_batch_size]
            metadata_items = [
                self._endpoint_metadata(
                    request.endpoint_key,
                    request.asset,
                    request.conditioning,
                    settings,
                )
                for request in chunk
            ]
            row_groups: list[list[dict[str, object]]] = [[] for _ in chunk]
            history_paths = [
                Path(request.checkpoint_directory) / "optimization.csv"
                for request in chunk
            ]
            checkpoint_callbacks = []
            diagnostics_callbacks = []
            for index, request in enumerate(chunk):
                directory = Path(request.checkpoint_directory)
                metadata = metadata_items[index]
                rows = row_groups[index]
                history_path = history_paths[index]

                def diagnostics_callback(
                    diagnostics: OptimizationStepDiagnostics,
                    *,
                    rows: list[dict[str, object]] = rows,
                ) -> None:
                    rows.append(diagnostics.to_dict())

                def checkpoint_callback(
                    step: int,
                    endpoint: FlowMorphEndpoint,
                    optimizer: torch.optim.Optimizer,
                    diagnostics: OptimizationStepDiagnostics,
                    *,
                    directory: Path = directory,
                    metadata: dict[str, object] = metadata,
                    rows: list[dict[str, object]] = rows,
                    history_path: Path = history_path,
                ) -> None:
                    checkpoint_metadata = dict(metadata)
                    checkpoint_metadata["completed_steps"] = step
                    save_optimizer = step < settings.optimization_steps
                    save_endpoint_checkpoint(
                        directory,
                        {"z": endpoint.z, "delta": endpoint.delta, "u": endpoint.u},
                        checkpoint_metadata,
                        optimizer=optimizer if save_optimizer else None,
                    )
                    _write_history(history_path, rows)

                diagnostics_callbacks.append(diagnostics_callback)
                checkpoint_callbacks.append(checkpoint_callback)

            predictor = runner._bound_predictor()
            if self.cfg_execution is not None:
                predictor.cfg_execution = self.cfg_execution
            target_tensors = [
                request.asset.latent.to(self.device, dtype=torch.float32)
                for request in chunk
            ]
            conditioning_batch = stack_conditioning_packages(
                [request.conditioning.to(self.device) for request in chunk]
            )
            try:
                optimized = optimize_endpoint_batch(
                    target_tensors,
                    sigma_i=start.sigma_i,
                    sigma_last=start.sigma_last,
                    timestep_i=runner.schedule.timesteps[
                        runner.config.flowmorph.start_timestep_index
                    ].to(self.device),
                    predictor=predictor,
                    conditioning=conditioning_batch,
                    config=settings,
                    predictor_parameters=runner.model.parameters(),
                    checkpoint_callbacks=checkpoint_callbacks,
                    diagnostics_callbacks=diagnostics_callbacks,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                if not (self.oom_backoff and is_oom and len(chunk) > 1):
                    raise
                predictor = None
                target_tensors = []
                conditioning_batch = None
                release_cuda_memory()
                print(
                    f"Endpoint fit OOM for batch_size={len(chunk)}; "
                    "retrying endpoints sequentially"
                )
                self.endpoint_batch_size_limit = 1
                for request in chunk:
                    directory = Path(request.checkpoint_directory)
                    results[request.endpoint_key] = self.fit_endpoint(
                        endpoint_key=request.endpoint_key,
                        asset=request.asset,
                        conditioning=request.conditioning,
                        checkpoint_directory=directory,
                        resume=directory.exists(),
                    )
                continue

            for request, optimization, rows, history_path in zip(
                chunk,
                optimized,
                row_groups,
                history_paths,
                strict=True,
            ):
                _write_history(history_path, rows)
                endpoint = optimization.endpoint.to(device="cpu", dtype=torch.float32)
                elapsed = float(rows[-1]["elapsed_seconds"]) if rows else None
                results[request.endpoint_key] = SequenceEndpointResult(
                    endpoint=endpoint,
                    checkpoint_directory=Path(request.checkpoint_directory),
                    resumed=False,
                    completed_steps=settings.optimization_steps,
                    elapsed_seconds=elapsed,
                )
            predictor = None
            target_tensors = []
            conditioning_batch = None
            del optimized
            release_cuda_memory()
        return results

    def render_midpoints(
        self,
        *,
        source: FlowMorphEndpoint,
        target: FlowMorphEndpoint,
        source_conditioning: ConditioningPackage,
        target_conditioning: ConditioningPackage,
        midpoint_conditionings: Sequence[ConditioningPackage],
        alphas: Sequence[float],
    ) -> tuple[RenderedLatentFrame, ...]:
        """Render only requested interior alphas from cached endpoint states."""

        if len(midpoint_conditionings) != len(alphas) or not alphas:
            raise ValueError("one midpoint conditioning is required per alpha")
        if any(not 0.0 < float(alpha) < 1.0 for alpha in alphas):
            raise ValueError("sequence midpoint alphas must lie strictly inside (0, 1)")
        runner = self.runner
        if runner.schedule is None:
            raise PipelineError("prepared runner lacks schedule")
        runner._set_lora_scale(runner.config.lora.render_scale)
        predictor = runner._bound_predictor()
        if self.cfg_execution is not None:
            predictor.cfg_execution = self.cfg_execution
        piecewise_conditionings = tuple(
            interpolate_conditioning_through_midpoint(
                source_conditioning,
                midpoint_conditioning,
                target_conditioning,
                float(alpha),
            ).to(self.device)
            for midpoint_conditioning, alpha in zip(
                midpoint_conditionings,
                alphas,
                strict=True,
            )
        )
        active_batch_size = min(self.render_batch_size, len(alphas))
        render_backed_off = False
        while True:
            try:
                frames = render_morph(
                    source.to(self.device, dtype=torch.float32),
                    target.to(self.device, dtype=torch.float32),
                    schedule=runner.schedule,
                    predictor=predictor,
                    source_conditioning=source_conditioning.to(self.device),
                    target_conditioning=target_conditioning.to(self.device),
                    frame_conditionings=piecewise_conditionings,
                    alphas=alphas,
                    render_indices=runner.config.flowmorph.render_indices,
                    conditioning_mode=RenderConditioningMode.PROMPT_SCHEDULE,
                    conditioning_batcher=stack_conditioning_packages,
                    batch_size=active_batch_size,
                    output_dtype=torch.float32,
                    use_inference_mode=True,
                )
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                if self.oom_backoff and is_oom and active_batch_size > 1:
                    active_batch_size = max(1, (active_batch_size + 1) // 2)
                    render_backed_off = True
                    release_cuda_memory()
                    print(f"FlowMorph render OOM; retrying with batch_size={active_batch_size}")
                    continue
                if (
                    self.oom_backoff
                    and is_oom
                    and predictor.cfg_execution == "batched"
                ):
                    predictor.cfg_execution = "sequential"
                    self.cfg_execution = "sequential"
                    release_cuda_memory()
                    print("FlowMorph render OOM with batched CFG; retrying sequential CFG")
                    continue
                raise
        self.last_render_batch_size = active_batch_size
        if render_backed_off:
            self.render_batch_size = active_batch_size
        output = tuple(
            RenderedLatentFrame(
                index=frame.index,
                alpha=frame.alpha,
                start_state=frame.start_state.detach().cpu(),
                final_latent=frame.final_latent.detach().cpu(),
                conditioning_mode=frame.conditioning_mode,
            )
            for frame in frames
        )
        del predictor, frames, piecewise_conditionings
        return output

    def render_endpoint_reconstructions(
        self,
        *,
        endpoints: Sequence[FlowMorphEndpoint],
        conditionings: Sequence[ConditioningPackage],
    ) -> tuple[RenderedLatentFrame, ...]:
        """Render canonical fitted endpoints once, batched across a sequence.

        A sequence endpoint participates as alpha=1 in its incoming gap and
        alpha=0 in its outgoing gap. Rendering it once from its cached fitted
        state and reusing the decoded image makes those two roles identical by
        construction.
        """

        if not endpoints or len(endpoints) != len(conditionings):
            raise ValueError(
                "canonical endpoint rendering requires one conditioning per endpoint"
            )
        runner = self.runner
        if runner.schedule is None:
            raise PipelineError("prepared runner lacks schedule")
        runner._set_lora_scale(runner.config.lora.render_scale)
        predictor = runner._bound_predictor()
        if self.cfg_execution is not None:
            predictor.cfg_execution = self.cfg_execution
        render_chain = get_render_chain(
            runner.schedule,
            runner.config.flowmorph.render_indices,
        )
        active_batch_size = min(self.render_batch_size, len(endpoints))
        render_backed_off = False
        while True:
            frames: list[RenderedLatentFrame] = []
            try:
                for chunk_start in range(0, len(endpoints), active_batch_size):
                    chunk_endpoints = endpoints[
                        chunk_start : chunk_start + active_batch_size
                    ]
                    chunk_conditionings = conditionings[
                        chunk_start : chunk_start + active_batch_size
                    ]
                    chunk_states = torch.cat(
                        [
                            endpoint.to(
                                self.device,
                                dtype=torch.float32,
                            ).state
                            for endpoint in chunk_endpoints
                        ],
                        dim=0,
                    )
                    prepared_conditionings = [
                        item.to(self.device) for item in chunk_conditionings
                    ]
                    conditioning = (
                        prepared_conditionings[0]
                        if len(prepared_conditionings) == 1
                        else stack_conditioning_packages(prepared_conditionings)
                    )
                    final_latents = render_latent_trajectory(
                        chunk_states,
                        predictor=predictor,
                        conditioning=conditioning,
                        render_chain=render_chain,
                        frame_index=chunk_start,
                    )
                    for offset in range(len(chunk_endpoints)):
                        frame_slice = slice(offset, offset + 1)
                        frames.append(
                            RenderedLatentFrame(
                                index=chunk_start + offset,
                                alpha=0.0,
                                start_state=chunk_states[
                                    frame_slice
                                ].detach().cpu(),
                                final_latent=final_latents[
                                    frame_slice
                                ].detach().cpu(),
                                conditioning_mode=(
                                    RenderConditioningMode.PROMPT_SCHEDULE
                                ),
                            )
                        )
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = (
                    isinstance(error, torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(error).lower()
                )
                if self.oom_backoff and is_oom and active_batch_size > 1:
                    active_batch_size = max(1, (active_batch_size + 1) // 2)
                    render_backed_off = True
                    release_cuda_memory()
                    print(
                        "Canonical endpoint render OOM; retrying with "
                        f"batch_size={active_batch_size}"
                    )
                    continue
                if (
                    self.oom_backoff
                    and is_oom
                    and predictor.cfg_execution == "batched"
                ):
                    predictor.cfg_execution = "sequential"
                    self.cfg_execution = "sequential"
                    release_cuda_memory()
                    print(
                        "Canonical endpoint render OOM with batched CFG; "
                        "retrying sequential CFG"
                    )
                    continue
                raise
        self.last_render_batch_size = active_batch_size
        if render_backed_off:
            self.render_batch_size = active_batch_size
        del predictor
        return tuple(frames)

    def _decode_token_batch(self, tokens: Sequence[torch.Tensor]) -> list[Image.Image]:
        runner = self.runner
        pipeline = runner.pipeline
        if pipeline is None or runner.image_ids is None:
            raise PipelineError("prepared runner lacks pipeline/image IDs")
        vae_dtype = _module_dtype(
            pipeline.vae,
            next(pipeline.transformer.parameters()).dtype,
        )
        token_batch = torch.cat(
            [token.to(self.device, dtype=vae_dtype) for token in tokens],
            dim=0,
        )
        image_ids = runner.image_ids.to(self.device).expand(
            token_batch.shape[0],
            -1,
            -1,
        )
        decoded = decode_packed_latent(
            token_batch,
            image_ids,
            pipeline.vae,
            image_processor=pipeline.image_processor,
            output_type="pil",
            postprocess=True,
        )
        if isinstance(decoded, Image.Image):
            images = [decoded]
        elif isinstance(decoded, (list, tuple)):
            images = list(decoded)
        else:
            raise PipelineError("batched VAE decode did not return PIL images")
        if len(images) != len(tokens) or not all(isinstance(image, Image.Image) for image in images):
            raise PipelineError("batched VAE decode returned an unexpected image count/type")
        del token_batch, image_ids, decoded
        return [image.convert("RGB") for image in images]

    def decode_frames(
        self,
        frames: Sequence[RenderedLatentFrame],
    ) -> tuple[Image.Image, ...]:
        """Decode a round's CPU latent frames with one VAE/model component swap."""

        runner = self.runner
        pipeline = runner.pipeline
        if pipeline is None:
            raise PipelineError("prepared runner lacks pipeline")
        transformer = pipeline.transformer
        vae = pipeline.vae
        _move_module(transformer, "cpu")
        release_cuda_memory()
        decoded: list[Image.Image] = []
        try:
            _move_module(vae, self.device)
            position = 0
            active_batch_size = min(self.decode_batch_size, len(frames)) if frames else 1
            decode_backed_off = False
            while position < len(frames):
                chunk = frames[position : position + active_batch_size]
                try:
                    decoded.extend(
                        self._decode_token_batch([frame.final_latent for frame in chunk])
                    )
                    position += len(chunk)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                    if not (self.oom_backoff and is_oom and active_batch_size > 1):
                        raise
                    active_batch_size = max(1, (active_batch_size + 1) // 2)
                    decode_backed_off = True
                    release_cuda_memory()
                    print(f"VAE decode OOM; retrying with batch_size={active_batch_size}")
            self.last_decode_batch_size = active_batch_size
            if decode_backed_off:
                self.decode_batch_size = active_batch_size
        finally:
            _move_module(vae, "cpu")
            _move_module(transformer, self.device)
            release_cuda_memory()
        return tuple(decoded)

    def decode_frames_to_paths(
        self,
        frames: Sequence[RenderedLatentFrame],
        output_paths: Sequence[str | Path],
        *,
        restore_transformer: bool = True,
    ) -> tuple[Path, ...]:
        """Decode and persist a round without retaining every RGB image in RAM."""

        if len(frames) != len(output_paths):
            raise ValueError("one output path is required per rendered frame")
        runner = self.runner
        pipeline = runner.pipeline
        if pipeline is None:
            raise PipelineError("prepared runner lacks pipeline")
        transformer = pipeline.transformer
        vae = pipeline.vae
        destinations = tuple(Path(path) for path in output_paths)
        _move_module(transformer, "cpu")
        release_cuda_memory()
        try:
            _move_module(vae, self.device)
            position = 0
            active_batch_size = min(self.decode_batch_size, len(frames)) if frames else 1
            decode_backed_off = False
            while position < len(frames):
                frame_chunk = frames[position : position + active_batch_size]
                destination_chunk = destinations[position : position + active_batch_size]
                try:
                    images = self._decode_token_batch(
                        [frame.final_latent for frame in frame_chunk]
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                    if not (self.oom_backoff and is_oom and active_batch_size > 1):
                        raise
                    active_batch_size = max(1, (active_batch_size + 1) // 2)
                    decode_backed_off = True
                    release_cuda_memory()
                    print(f"VAE decode OOM; retrying with batch_size={active_batch_size}")
                    continue
                for image, destination in zip(images, destination_chunk, strict=True):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.save(destination, format="PNG", compress_level=4)
                    image.close()
                position += len(frame_chunk)
            self.last_decode_batch_size = active_batch_size
            if decode_backed_off:
                self.decode_batch_size = active_batch_size
        finally:
            _move_module(vae, "cpu")
            if restore_transformer:
                _move_module(transformer, self.device)
            release_cuda_memory()
        return destinations


__all__ = [
    "EncodedSequenceImage",
    "FlowMorphSequenceSession",
    "SequenceEndpointRequest",
    "SequenceEndpointResult",
]
