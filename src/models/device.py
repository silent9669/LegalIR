from typing import Literal
import torch


def resolve_device(requested: str | torch.device = "auto") -> str:
    if isinstance(requested, torch.device):
        requested = str(requested)
    req = str(requested or "auto").lower().strip()

    if req in {"cpu", "cuda", "mps"}:
        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was explicitly requested but is not available")
        return req

    if req.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{req}' was explicitly requested but CUDA is not available")
        try:
            idx = int(req.split(":", 1)[1])
        except ValueError:
            raise RuntimeError(f"Invalid CUDA device specification: {req}")
        if idx < 0 or idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {idx} out of range (available devices: {torch.cuda.device_count()})"
            )
        return f"cuda:{idx}"

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda:0" if torch.cuda.device_count() > 0 else "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

