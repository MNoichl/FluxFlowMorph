# FluxFlowMorph

An inspectable, resumable adaptation of
[FlowMorph](https://github.com/VITA-Group/FlowMorph) for FLUX.2 Klein Base 9B,
with one optional user-supplied LoRA. The official repository remains the
reference source; an explicitly labeled experimental art mode supports the
pinned public `Runware/BFL-FLUX.2-klein-base-9B` mirror.

The implementation freezes the transformer, active LoRA, VAE, and text
encoder; fits only FP32 endpoint variables `pred` and `u`; fits source and
target sequentially; interpolates `z`/`delta` linearly and `u` direction/norm
separately; and renders every frame through a verified sparse rectified-flow
chain. It never silently switches to FLUX.1, 4B, distilled 9B, a smaller
resolution, fewer optimization steps, fewer frames, lower CFG, or altered
precision.

## Status

The complete package, Colab facade, CPU unit-test surface, and gated CUDA
integration gates are included. Primary-source inspection is pinned in
[`research_notes.md`](docs/research_notes.md), and every parity decision is in
[`fidelity_matrix.md`](docs/fidelity_matrix.md).

Development in this repository occurred on a CPU-only machine without gated
model credentials and before the user supplied a LoRA. Therefore actual 9B
load, VAE/velocity parity, adapter activation, production backward, A100 memory,
full endpoint fits, and full render are deliberately recorded as **not run**,
not passed. Run `scripts/validate_colab.py` on the intended licensed CUDA
runtime before starting a reproduction.

## Reference contract

- model: FLUX.2 Klein Base 9B, pinned model revision;
- image size: 512×512;
- endpoint fitting: 100 source steps, then 100 target steps;
- optimizer: AdamW, `pred` LR 0.04, `u` LR 0.01, weight decay 0.01;
- loss: released-code unsquared vector L2 (`paper_l2_squared` is opt-in);
- schedule: 100 Klein scheduler points, start index 35;
- render calls: indices 35, 55, 75, and 95, then terminal sigma;
- external CFG: scale 4.0, sequential branches by default;
- morph: 20 inclusive linear alphas with decoupled interpolation;
- LoRA: one native unfused adapter, same scale during fitting/rendering;
- output: raw/display frames, resumable endpoints, metrics, previews, and one
  atomic `RUN_ID.flowmorph-klein.zip`.

## Colab

Open
[`FlowMorph_FLUX2_Klein_Base_9B_LoRA_Colab.ipynb`](notebooks/FlowMorph_FLUX2_Klein_Base_9B_LoRA_Colab.ipynb)
in standard Colab or through the VS Code Colab extension. Its first
configuration cell exposes plain Python variables, including:

```python
RUN_MODE = "experimental"
MODEL_ID = "Runware/BFL-FLUX.2-klein-base-9B"
MODEL_REVISION = "52d7274119d8a2b67f4fba1a43694d9169a44851"
GENERATE_TEST_ENDPOINTS = True
LORA_SOURCE = None  # set later to org/repo, HF file URL, resolve URL, or local safetensors
PROFILE = "auto"
```

The default notebook preview anonymously downloads the pinned public mirror,
generates a young-woman source portrait, creates an older-man target
conditioned on that source, saves both endpoint images, and unloads the preview
pipeline before fitting. Set `GENERATE_TEST_ENDPOINTS = False` to use your own
images. The official gated repository and private LoRAs still require
`HF_TOKEN`; never paste a token into an editable cell.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -m "not integration"
```

The production Colab environment uses `requirements-colab.txt`, which pins
Diffusers to the inspected v0.39.0 commit. It intentionally does not reinstall
PyTorch blindly; the notebook validates the Colab-provided CUDA build.

Run from the command line:

```bash
python scripts/run_flowmorph.py \
  --config configs/full_9b_lora.yaml \
  --source /content/flowmorph_klein_images/max_v1/images/source.png \
  --target /content/flowmorph_klein_images/max_v1/images/target.png \
  --lora-source org/repository \
  --lora-scale 0.8
```

Run the mandatory production-shape preflight separately:

```bash
python scripts/validate_colab.py \
  --config configs/full_9b_lora.yaml \
  --source /content/flowmorph_klein_images/max_v1/images/source.png \
  --target /content/flowmorph_klein_images/max_v1/images/target.png
```

Resume a compatible endpoint run by naming its existing directory:

```bash
python scripts/resume_flowmorph.py \
  --run-directory /content/flowmorph_klein_results/max_v1/full_lora_reproduction_v1/RUN_ID \
  --config configs/full_9b_lora.yaml \
  --source /path/to/the/same/source.png \
  --target /path/to/the/same/target.png
```

Any change
to model revision, LoRA fingerprint/scale, prompt, processed image,
preprocessing, scheduler, start index, latent shape, precision, or Diffusers
commit is rejected.

## Input manifest

```yaml
project_name: flowmorph_klein_full
defaults:
  model_id: black-forest-labs/FLUX.2-klein-base-9B
  profile: auto
  width: 512
  height: 512
  frame_count: 20
  seed: 42
  guidance_scale: 4.0
  lora_source: null
  lora_scale_fit: 1.0
  lora_scale_render: 1.0
pairs:
  - id: pair_001
    source_image: images/source.png
    target_image: images/target.png
    source_prompt: null
    target_prompt: null
    bridge_prompt: a smooth transformation between the two subjects
    negative_prompt: ""
```

Inputs are validated before model download. Multiple pairs reuse the model
only when model revision, LoRA fingerprint/scale, and precision profile match.

## Documentation

- [Architecture](docs/architecture.md)
- [Colab workflow](docs/colab_workflow.md)
- [Memory profiles](docs/memory_profiles.md)
- [LoRA compatibility](docs/lora_compatibility.md)
- [Limitations](docs/limitations.md)
- [License notes](LICENSE_NOTES.md)
- [Implementation report](IMPLEMENTATION_REPORT.md)
