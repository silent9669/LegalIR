import pytest
from src.training.samplers import (
    MicrobatchFactorization,
    get_effective_batch_factorizations,
    validate_factorization,
)


def test_effective_batch_factorizations():
    factors = get_effective_batch_factorizations(target_effective_batch=16)
    assert len(factors) >= 3

    for f in factors:
        assert f.microbatch_size * f.gradient_accumulation_steps == 16
        assert validate_factorization(f, expected_effective_batch=16) is True

    # Check top candidate is 8x2
    assert factors[0].microbatch_size == 8
    assert factors[0].gradient_accumulation_steps == 2


def test_validate_factorization_invalid():
    bad = MicrobatchFactorization(microbatch_size=8, gradient_accumulation_steps=4)
    assert validate_factorization(bad, expected_effective_batch=16) is False
