"""FlowMorph adaptation for the undistilled FLUX.2 Klein Base 9B model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flowmorph-klein")
except PackageNotFoundError:  # source tree import
    __version__ = "0.1.0"

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
MIRROR_MODEL_ID = "Runware/BFL-FLUX.2-klein-base-9B"
FP8_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9b-fp8"
DIFFUSERS_COMMIT = "a3608b512ed7248499a44c61d954965ed9bdae4d"
FLOWMORPH_COMMIT = "0db52344ad0ec6963f74a508db831e506058b2f7"
FLUX2_COMMIT = "50fe5162777813d869182b139e83b10743caef15"
MODEL_REVISION = "32773329fbe7e81a90ef971740e8ba4b0364ecf3"
MIRROR_MODEL_REVISION = "52d7274119d8a2b67f4fba1a43694d9169a44851"
FP8_MODEL_REVISION = "9ecf2143d71542449960c5584340269c6d401449"

__all__ = [
    "DIFFUSERS_COMMIT",
    "FLOWMORPH_COMMIT",
    "FLUX2_COMMIT",
    "FP8_MODEL_ID",
    "FP8_MODEL_REVISION",
    "MIRROR_MODEL_ID",
    "MIRROR_MODEL_REVISION",
    "MODEL_ID",
    "MODEL_REVISION",
    "__version__",
]
