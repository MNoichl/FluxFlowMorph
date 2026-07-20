# Fidelity matrix

The default reference profile prefers released FlowMorph code parity where the
paper and code differ. `required_flux2_adaptation` means the behavior is needed
to run the undistilled FLUX.2 Klein Base 9B model and therefore prevents a
claim of bit-exact original-paper reproduction.

| Behavior | Classification | Reference implementation | Evidence/deviation |
|---|---|---|---|
| Endpoint latent initialization | `flowmorph_code_parity` | `pred=clone(z)`, `u=zeros_like(z)` | Released interpolation path |
| Clean endpoint encoding | `required_flux2_adaptation` | deterministic FLUX.2 VAE posterior mode | Released FLUX.1 code samples; Klein pipeline uses mode |
| Loss definition | `flowmorph_code_parity` | global unsquared L2 norm | Paper squared alternative exposed as `paper_l2_squared` |
| Paper loss alternative | `flowmorph_paper_parity` | mean/squared reconstruction option, explicit opt-in | Not the reference default |
| Optimizer type | `flowmorph_code_parity` | AdamW | Same optimizer as release |
| Pred learning rate | `flowmorph_code_parity` | 0.04 | Separate parameter group |
| u learning rate | `flowmorph_code_parity` | 0.01 | Separate parameter group |
| Weight decay | `flowmorph_code_parity` | 0.01 | PyTorch AdamW default used implicitly upstream; explicit here |
| Optimization steps | `flowmorph_code_parity` | 100 source + 100 target | Never silently reduced |
| Start timestep index | `flowmorph_code_parity` | 35 | Reference profile fixed |
| Scheduler points | `flowmorph_code_parity` | 100 model-evaluation points | Terminal zero is appended |
| Scheduler construction | `required_flux2_adaptation` | Klein custom linear sigmas plus empirical dynamic shift from actual token count | Released code hard-codes FLUX.1 sequence length 16 |
| Render timestep indices | `flowmorph_code_parity` | 35, 55, 75, 95, then terminal | Four velocity evaluations per frame |
| Euler sign | `flowmorph_code_parity` | `x_next=x+(sigma_next-sigma)*v` | Verified from pinned scheduler source; runtime parity gate remains |
| Delta interpolation | `flowmorph_code_parity` | linear | Exact inclusive endpoints |
| u direction interpolation | `flowmorph_code_parity` | SLERP | Deterministic finite fallback for opposite/zero vectors |
| u magnitude interpolation | `flowmorph_code_parity` | linear norm interpolation | Norm calculations in FP32 |
| z interpolation | `flowmorph_code_parity` | linear | Exact inclusive endpoints |
| Prompt conditioning during source fit | `flowmorph_code_parity` | resolved source prompt | Cached package |
| Prompt conditioning during target fit | `flowmorph_code_parity` | resolved target prompt | Cached package |
| Prompt conditioning during reference render | `flowmorph_code_parity` | source package | Released code renders through source wrapper |
| Bridge rendering | `experimental_extension` | shared bridge package | User-facing option, not reference default |
| Interpolated prompt embeddings | `experimental_extension` | only shape-compatible continuous embeddings | Never interpolates token IDs |
| Classifier-free guidance | `required_flux2_adaptation` | external CFG, scale 4.0, sequential branches | Base 9B is undistilled; released Schnell path has no external CFG |
| Configurable negative prompt | `experimental_extension` | same Qwen encoding path | Stock high-level Klein uses empty unconditional text |
| LoRA activation | `required_flux2_adaptation` | one native, unfused transformer LoRA active for fit/render | User requirement beyond original FlowMorph |
| LoRA fit/render scale | `required_flux2_adaptation` | both 1.0 by default | Mismatch is warned experimental behavior |
| Sequential endpoint fitting | `colab_execution_adaptation` | one frozen 9B model, optimizer state released between endpoints | Replaces two full upstream wrappers without changing endpoint equations |
| Float32 optimization variables | `colab_execution_adaptation` | FP32 `pred` and `u`, cast input to model dtype | Released variables inherit BF16; gradient cast parity is tested |
| Frozen model and LoRA | `flowmorph_code_parity` | only `pred` and `u` require gradients | Re-checked after adapter loading |
| Gradient checkpointing | `colab_execution_adaptation` | enabled in the reference candidate; explicitly switchable only in experimental mode | Real output/gradient/VRAM A/B gate remains mandatory before support is claimed |
| Text encoder offload | `colab_execution_adaptation` | cache all prompt packages, then CPU/delete | VRAM compromise, no fitting-equation change |
| VAE offload | `colab_execution_adaptation` | encode endpoints, offload, restore for decode | VRAM compromise, no fitting-equation change |
| Model CPU offload | `experimental_extension` | disabled in reference profile | Input gradients must be separately proven |
| Attention backend | `colab_execution_adaptation` | config `sdpa` maps to pinned Diffusers `native` | Uses PyTorch scaled-dot-product attention |
| Output endpoint replacement | `colab_execution_adaptation` | preserve raw endpoints; replace display frame 0/last with references | Visual exactness without deleting algorithmic reconstructions |
| Checkpoints and resume | `colab_execution_adaptation` | explicit `z`, `delta`, `u`, compatibility metadata, optional AdamW moments | Required for transient Colab sessions |
| LoRA numerical smoke test | `required_flux2_adaptation` | active name plus non-zero deterministic velocity delta | Loading without error is insufficient |
| Production backward probe | `colab_execution_adaptation` | exact 512×512 CFG path with input backward | Required before endpoint fitting |
| FP8 Base 9B | `experimental_extension` | explicit profile only | No BF16/4B/distilled silent substitution |
