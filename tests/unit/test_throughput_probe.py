import pytest
from src.training.samplers import (
    MicrobatchFactorization,
    get_effective_batch_factorizations,
    validate_factorization,
    probe_factorization_step,
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


def test_effective_batch_always_16():
    factors = get_effective_batch_factorizations(16)
    for f in factors:
        assert f.effective_batch_size == 16
        assert validate_factorization(f, 16) is True


def test_validate_factorization_invalid():
    bad = MicrobatchFactorization(microbatch_size=8, gradient_accumulation_steps=4)
    assert validate_factorization(bad, expected_effective_batch=16) is False


def test_t4_probe_performs_real_backward_step():
    import tempfile
    from pathlib import Path
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    config = BertConfig(vocab_size=300, hidden_size=32, num_attention_heads=2, num_hidden_layers=2, max_position_embeddings=512, num_labels=1)
    model = BertForSequenceClassification(config)

    tmp_vocab = Path(tempfile.gettempdir()) / "probe_vocab.txt"
    if not tmp_vocab.exists():
        vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
        tmp_vocab.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(tmp_vocab))

    factorization = MicrobatchFactorization(microbatch_size=2, gradient_accumulation_steps=8)
    sample_pairs = [
        ("câu hỏi 1", "văn bản 1", 1.0),
        ("câu hỏi 2", "văn bản 2", 0.0),
    ]

    result = probe_factorization_step(
        model=model,
        tokenizer=tokenizer,
        factorization=factorization,
        sample_pairs=sample_pairs,
        device="cpu",
        max_length=128,
    )
    assert result["status"] == "PASS"
    assert result["effective_batch_size"] == 16
    assert result["loss"] > 0
    assert result["param_diff"] > 0  # Proves weights were updated by optimizer step
    assert result["seconds_per_step"] > 0
