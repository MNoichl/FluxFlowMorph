# Colab workflow

The notebook is deliberately a thin facade over the package. It supports the
browser interface, the VS Code Colab extension, and ordinary Jupyter with
Colab-only imports guarded.

Local ephemeral roots are:

```text
/content/FlowMorphKlein9B
/content/flowmorph_klein_images/max_v1
/content/flowmorph_klein_work/max_v1
/content/flowmorph_klein_results/max_v1/full_lora_reproduction_v1
/content/hf_cache
```

Optional persistence is
`/content/drive/MyDrive/FlowMorphKlein9B`. Images, prompt caches, optimization
states, frames, video, and archive assembly stay under local `/content`. Only
small endpoint checkpoints and the completed verified archive should be copied
to Drive.

Workflow:

1. Edit the plain Python configuration variables at the notebook top.
2. Install the exact project/Diffusers commit without replacing a compatible
   Colab PyTorch build.
3. Resolve `HF_TOKEN` from the environment, Colab secret storage, or the Hub's
   interactive login. The token is never displayed or serialized.
4. Stage and validate the manifest, images, and optional local adapter before
   downloading the base model.
5. Check gated model access once. Authorization failure stops immediately.
6. Detect GPU and choose `cuda:0`. CPU cannot run the production workflow.
7. Load Base 9B once, cache Qwen conditioning, deterministically encode both
   endpoints, and offload text encoder/VAE as configured.
8. Resolve, inspect, load, freeze, activate, and numerically test the optional
   LoRA. When no LoRA is supplied, the run manifest says `not_configured` and
   cannot claim LoRA acceptance.
9. Run a real 512×512 CFG backward probe. An inference-only pass is not enough.
10. Fit source for 100 steps and persist it, then fit target for 100 steps.
11. Interpolate 20 endpoint representations, use four sparse Euler updates per
    frame, decode, and preserve both raw and display sequences.
12. Calculate metrics, previews, manifests, checksums, and the atomic archive.
13. Print archive path, size, and SHA-256. Use `google.colab.files.download`
    only when Colab is available. `/content` is temporary.

The three-frame smoke configuration changes only the frame count and remains
explicitly labeled `smoke`; it is not the final reproduction.

