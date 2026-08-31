"""Train cross-encoder reranker with LoRA/PEFT on fold-safe pairs."""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_reranker import train_reranker


def main():
    parser = argparse.ArgumentParser(description="LegalIR LoRA Reranker Trainer")
    parser.add_argument("--config", type=str, default="configs/experiments/reranker_lora.yaml", help="Path to reranker config")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output checkpoint directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model name or path")
    parser.add_argument("--loss-type", type=str, default=None, help="Loss function (bce, pairwise_logistic, pairwise_margin, listwise_ce)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    print("=" * 60)
    print(f"LegalIR: Training Cross-Encoder Reranker for Fold {args.fold}")
    print("=" * 60)

    report = train_reranker(
        config_path=args.config,
        fold=args.fold,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        base_model_name=args.base_model,
        loss_type=args.loss_type,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print(f"Training status: {report.get('status')}")
    print(f"Trainable parameters: {report.get('trainable_params')} ({report.get('trainable_percent')}%)")
    print(f"Parameter diff (weight update check): {report.get('param_diff')}")
    print(f"Final train loss: {report.get('final_train_loss')}")


if __name__ == "__main__":
    main()
