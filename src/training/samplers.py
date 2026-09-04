"""Training batch samplers, microbatch factorization, and throughput measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MicrobatchFactorization:
    """Factorization of effective batch size into microbatch and gradient accumulation."""

    microbatch_size: int
    gradient_accumulation_steps: int

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.gradient_accumulation_steps


def get_effective_batch_factorizations(
    target_effective_batch: int = 16,
) -> List[MicrobatchFactorization]:
    """
    Get ordered list of candidate microbatch factorizations for Tesla T4 GPU.
    Ordered from largest microbatch (fastest execution) to smaller fallbacks.
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
