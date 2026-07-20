# Limitations

- The local development machine had PyTorch CPU only and no access to the
  gated 9B weights. Actual model load, VAE parity, low-level velocity parity,
  LoRA activation, 512×512 input gradients, VRAM use, full endpoint fitting,
  and complete morph rendering remain integration gates.
- A compatible user LoRA has not yet been supplied. The generic resolver and
  validator can be unit-tested, but adapter acceptance cannot be claimed.
- The project is a FLUX.2 adaptation of released FlowMorph behavior, not an
  exact paper reproduction. External CFG, Qwen conditioning, deterministic
  FLUX.2 VAE mode, VAE BN normalization, and FLUX.2 scheduling are required
  changes.
- The original paper/release disagree on squared versus unsquared loss and on
  several implementation details. Both loss modes exist; the default follows
  released code.
- A100 40 GB and the FP8 profile are not supported merely because model
  inference works. Each needs the production backward and full-endpoint gates.
- LPIPS may download its own backbone unless weights are already cached. The
  core metric API requires an explicitly constructed model so this side effect
  is visible to the notebook.
- Display frames replace generated endpoints for visual exactness. Raw endpoint
  reconstructions remain authoritative for metrics.
- Automatic captioning, LoRA training, FLUX.1, 4B/distilled production
  fallbacks, attention manipulation, web hosting, multi-GPU, RIFE, and output
  publication are intentionally out of scope.

