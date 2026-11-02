"""PyTorch accelerator discovery and capability checks."""
from typing import Optional

import torch

def get_accelerator() -> Optional[torch.device]:
    """Return the available PyTorch accelerator, if there is one."""
    accelerator_api = getattr(torch, "accelerator", None)
    if accelerator_api is not None and accelerator_api.is_available():
        return accelerator_api.current_accelerator()

    backends = (
        ("cuda", getattr(torch, "cuda", None)),
        ("mps", getattr(torch.backends, "mps", None)),
        ("xpu", getattr(torch, "xpu", None)),
    )
    for name, backend in backends:
        if backend is not None and backend.is_available():
            return torch.device(name)
    return None

def get_device() -> torch.device:
    """Return the available accelerator device, or CPU when none is available."""
    return get_accelerator() or torch.device("cpu")

def supports_bf16(device: torch.device) -> bool:
    """Check whether the selected accelerator reports bfloat16 support."""
    backend = getattr(torch, device.type, None)
    check = getattr(backend, "is_bf16_supported", None)
    return check is not None and check()