#!/usr/bin/env python3
"""CLI script to probe Tesla T4 training throughput with real forward/backward steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.training.samplers import (
    get_effective_batch_factorizations,
    probe_factorization_step,
    validate_factorization,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def main():
    parser = argparse.ArgumentParser(description="Probe GPU microbatch throughput with real backward step.")
    parser.add_argument("--effective-batch", type=int, default=16, help="Target effective batch size")
    parser.add_argument("--pairs-file", type=str, default=None, help="Optional training pairs parquet for probe")
    parser.add_argument("--output-report", type=str, default="artifacts/factory/t4_throughput_report.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print(f"[*] Probing factorizations for effective batch size: {args.effective_batch}")
    factorizations = get_effective_batch_factorizations(args.effective_batch)

    resolved_device = "cuda:0" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    if args.device != "auto":
        resolved_device = args.device

    print(f"[*] Probe target device: {resolved_device}")

    # Sample pairs
    sample_pairs = [
        ("Người phạm tội mua bán chiếm đoạt bộ phận cơ thể người bị phạt thế nào?", "Điều 154 Bộ luật Hình sự quy định tội mua bán chiếm đoạt mô hoặc bộ phận cơ thể người...", 1.0),
        ("Quy định xử phạt vi phạm giao thông đường bộ?", "Nghị định 100/2019/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông...", 0.0),
    ]
    if args.pairs_file and Path(args.pairs_file).is_file():
        df = pd.read_parquet(args.pairs_file).head(16)
        sample_pairs = list(zip(df["query_text"].astype(str), df["evidence_text"].astype(str), df["label"].astype(float)))

    # Base model setup
    if resolved_device.startswith("cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        base_model_name = "BAAI/bge-reranker-v2-m3"
        print(f"[*] Loading {base_model_name} on {resolved_device} ...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
    else:
        # Lightweight mock model on CPU for fast CI probe
        import tempfile
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        config = BertConfig(vocab_size=300, hidden_size=32, num_attention_heads=2, num_hidden_layers=2, max_position_embeddings=512, num_labels=1)
        model = BertForSequenceClassification(config)
        tmp_vocab = Path(tempfile.gettempdir()) / "probe_vocab.txt"
        if not tmp_vocab.exists():
            vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
            tmp_vocab.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(tmp_vocab))

    chosen_result = None
    all_reports = []

    for f in factorizations:
        print(f"[*] Testing factorization mb={f.microbatch_size}, ga={f.gradient_accumulation_steps} ...")
        try:
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            step_result = probe_factorization_step(
                model=model,
                tokenizer=tokenizer,
                factorization=f,
                sample_pairs=sample_pairs,
                device=resolved_device,
                max_length=512,
            )
            all_reports.append(step_result)
            print(f"    [+] PASSED: loss={step_result['loss']:.4f}, param_diff={step_result['param_diff']:.6f}, time={step_result['seconds_per_step']:.3f}s")
            chosen_result = step_result
            break
        except Exception as e:
            print(f"    [-] FAILED factorization {f.microbatch_size}x{f.gradient_accumulation_steps}: {e}")
            all_reports.append({
                "factorization": f.to_dict(),
                "status": "FAIL",
                "error": str(e),
                "oom_occurred": "out of memory" in str(e).lower(),
            })

    if chosen_result is None:
        print("[!] All factorizations failed!")
        sys.exit(1)

    out_p = Path(args.output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as fp:
        json.dump({"selected": chosen_result, "history": all_reports}, fp, indent=2)

    stable_f = chosen_result["factorization"]
    print(f"[+] Throughput probe completed. Selected: mb={stable_f['microbatch_size']}, ga={stable_f['gradient_accumulation_steps']}. Report saved to {out_p}")
    sys.exit(0)


if __name__ == "__main__":
    main()
