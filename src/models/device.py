from typing import Literal
import torch


def resolve_device(requested: str = "auto") -> str:
    req = (requested or "auto").lower().strip()
    if req in {"cpu", "cuda", "mps"}:
        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was explicitly requested but is not available")
        return req

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
