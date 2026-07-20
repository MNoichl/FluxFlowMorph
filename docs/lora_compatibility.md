# LoRA compatibility

Version 1 accepts one optional LoRA trained for FLUX.2 Klein Base 9B. Supported
source forms are:

- `org/repository`;
- a Hugging Face repository page;
- a Hugging Face `blob` file page;
- a direct Hugging Face `resolve` URL;
- a local `.safetensors` file.

Hub downloads use `huggingface_hub` APIs with an extracted repository ID,
revision, subfolder, and filename. Manual resolve-URL concatenation is not
used. The resolved local file is hashed and its safetensors metadata/keys are
inspected before native loading.

Accepted upstream conversion families include native Diffusers/PEFT keys,
Kohya LoRA keys, and ai-toolkit/FAL `diffusion_model.*` keys supported by the
pinned `Flux2LoraLoaderMixin`. LoHa, LoKr, unrelated adapters, zero mapped
transformer tensors, FLUX.1 provenance, and 4B provenance are rejected.

The Base and distilled Klein 9B variants share tensor architecture. Shape
matching cannot distinguish them. The validator therefore combines repository
and safetensor metadata, key families, loaded transformer shape matching, and
native loader diagnostics. Distilled/ambiguous provenance requires an explicit
override and emits a report warning.

Loading is unfused under the stable adapter name `flowmorph_adapter`.
`get_active_adapters()` and `get_list_adapters()` must both confirm it. The
mandatory numerical test evaluates the same latent/prompt/timestep with the
adapter disabled and enabled and records maximum and mean absolute velocity
change. Zero change is a failure. After loading, all base and adapter
parameters are frozen again, and fitting asserts they never receive gradients.

Fit and render scales default to 1.0 and must match in reference mode. A scale
mismatch changes the vector field after endpoint fitting and is explicitly
experimental. The downloaded adapter is excluded from the output archive by
default.

