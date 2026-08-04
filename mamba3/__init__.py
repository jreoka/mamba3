"""Public API for mamba3."""

import warnings

# TorchInductor's lowering still routes shape assertions through the
# deprecated torch._prims_common.check during torch.compile; silence the
# deprecation notice from that one module so compilation stays quiet.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\._prims_common\.check` is deprecated.*",
    category=FutureWarning,
    module=r"torch\._inductor\.lowering",
)

from .model import Mamba3

__all__ = ["Mamba3"]
