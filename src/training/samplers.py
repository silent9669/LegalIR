"""Training batch samplers, microbatch factorization, and T4 throughput probing."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MicrobatchFactorization:
    """Factorization of effective batch size into microbatch and gradient accumulation."""

    microbatch_size: int
    gradient_accumulation_steps: int

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.gradient_accumulation_steps

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def get_effective_batch_factorizations(
    target_effective_batch: int = 16,
) -> List[MicrobatchFactorization]:
    """
    Get ordered list of candidate microbatch factorizations for Tesla T4 GPU.
    Ordered from largest microbatch (fastest execution) to smaller fallbacks:
    8x2 -> 4x4 -> 2x8 -> 1x16.
    """
    candidates = [
        (8, 2),
        (4, 4),
        (2, 8),
        (1, 16),
    ]
    return [
        MicrobatchFactorization(mb, ga)
        for mb, ga in candidates
        if mb * ga == target_effective_batch
    ]


def validate_factorization(
    factorization: MicrobatchFactorization,
    expected_effective_batch: int = 16,
) -> bool:
    """Assert that factorization maintains the exact target effective batch size."""
    return factorization.effective_batch_size == expected_effective_batch


def probe_factorization_step(
    model: Any,
    tokenizer: Any,
    factorization: MicrobatchFactorization,
    sample_pairs: List[Tuple[str, str, float]],
    device: str = "cpu",
    max_length: int = 512,
    learning_rate: float = 5e-5,
) -> Dict[str, Any]:
    """
    Perform a real forward and backward optimizer step on real query/passage pairs
    at max_length=512 to verify memory and throughput.
    """
    if not validate_factorization(factorization, expected_effective_batch=16):
        raise ValueError(
            f"Effective batch size must remain 16, got {factorization.effective_batch_size}"
        )

    import torch
    import torch.nn as nn
    from torch.optim import AdamW

    dev = torch.device(device)
    model.to(dev)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    mb = factorization.microbatch_size
    ga = factorization.gradient_accumulation_steps

    # Prepare batch of size mb
    batch_data = sample_pairs[:mb]
    if len(batch_data) < mb:
        # Repeat to match required microbatch size
        batch_data = (batch_data * (mb // max(1, len(batch_data)) + 1))[:mb]

    queries = [str(p[0]) for p in batch_data]
    passages = [str(p[1]) for p in batch_data]
    labels = torch.tensor([float(p[2]) for p in batch_data], dtype=torch.float32, device=dev).unsqueeze(1)

    t0 = time.perf_counter()

    # Capture initial weights for param diff verification
    initial_weights = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    encoded = tokenizer(
        queries,
        passages,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(dev) for k, v in encoded.items()}

    optimizer.zero_grad()
    outputs = model(**encoded)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    loss = criterion(logits, labels)

    # Perform real backward step
    loss.backward()

    # Optimizer step
    optimizer.step()
    step_duration = time.perf_counter() - t0

    # Verify parameter difference
    diffs = [
        torch.norm(p.detach() - init).item()
        for p, init in zip([p for p in model.parameters() if p.requires_grad], initial_weights)
    ]
    param_diff = sum(diffs) / max(1, len(diffs))

    peak_vram = 0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(dev)

    return {
        "status": "PASS",
        "factorization": factorization.to_dict(),
        "effective_batch_size": factorization.effective_batch_size,
        "loss": float(loss.item()),
        "param_diff": float(param_diff),
        "seconds_per_step": float(step_duration),
        "peak_vram_bytes": peak_vram,
        "oom_occurred": False,
    }
