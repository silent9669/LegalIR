from pathlib import Path
from typing import Any
import argparse
import json
import pandas as pd
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.paths import ProjectPaths
from src.models.device import resolve_device
from src.training.trainer import RerankerTrainer


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    out_dir = str(data.get("output_dir", "artifacts/local/training/checkpoints"))
    if not (out_dir.startswith("artifacts/local/") or "/local/" in out_dir or out_dir.startswith("/kaggle/working/")):
        raise ValueError(f"Training output_dir must be inside artifacts/local/ or /kaggle/working/, got: {out_dir}")

    return data


def train_reranker(
    config_path: str | Path = "configs/experiments/reranker_lora.yaml",
    fold: int = 0,
    output_dir: str | Path | None = None,
    max_steps: int | None = None,
    base_model_name: str | None = None,
    loss_type: str | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    """
    Supervised Cross-Encoder fine-tuning with PEFT LoRA, verified weight updates,
    and checkpoint saving.
    """
    paths = ProjectPaths.from_repo()
    cfg = load_training_config(config_path) if Path(config_path).exists() else {}

    # Command-line / argument overrides
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    if base_model_name is not None:
        cfg["base_model_name"] = base_model_name
    if loss_type is not None:
        cfg["loss_type"] = loss_type
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if learning_rate is not None:
        cfg["learning_rate"] = learning_rate

    device = resolve_device(cfg.get("device", "auto"))

    out_path = (
        Path(output_dir)
        if output_dir
        else paths.repo / cfg.get("output_dir", "artifacts/local/training/checkpoints") / f"fold_{fold}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # Check local base model path or manifest
    model_name_or_path = cfg.get("base_model_name", "BAAI/bge-reranker-v2-m3")
    hf_manifest_path = paths.local_models / "huggingface" / "manifest.json"
    if hf_manifest_path.exists():
        try:
            hf_data = json.loads(hf_manifest_path.read_text(encoding="utf-8"))
            if model_name_or_path in hf_data and "path" in hf_data[model_name_or_path]:
                local_path = Path(hf_data[model_name_or_path]["path"])
                if local_path.is_dir():
                    model_name_or_path = str(local_path)
        except Exception as e:
            print(f"Warning: failed reading HF manifest: {e}")

    # Check pairs file
    pairs_file = paths.repo / "artifacts" / "local" / "training" / "pairs" / f"fold_{fold}" / "reranker_pairs.parquet"
    if not pairs_file.exists():
        print(f"Training pairs not found at {pairs_file}. Generating pairs for fold {fold}...")
        from src.training.build_pairs import build_training_pairs
        build_training_pairs(fold=fold, limit=50)

    pairs_df = pd.read_parquet(pairs_file)
    print(f"Loaded {len(pairs_df)} reranker training pairs for fold {fold}")

    # Split 90/10 train/val from pairs
    unique_qids = pairs_df["query_id"].unique()
    n_val = max(1, int(len(unique_qids) * 0.1))
    val_qids = set(unique_qids[-n_val:]) if len(unique_qids) > 1 else set()
    train_pairs_df = pairs_df[~pairs_df["query_id"].isin(val_qids)]
    val_pairs_df = pairs_df[pairs_df["query_id"].isin(val_qids)] if val_qids else None

    # Load tokenizer and model
    print(f"Loading base model: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1,
    )

    trainer = RerankerTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_data=train_pairs_df,
        val_data=val_pairs_df,
        config=cfg,
        device=device,
    )

    report = trainer.train(output_dir=out_path)
    report["fold"] = fold
    report["base_model"] = model_name_or_path
    report["output_dir"] = str(out_path)

    # Overwrite manifest with full report
    manifest_file = out_path / "training_manifest.json"
    manifest_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Reranker training completed for fold {fold}, manifest saved to {manifest_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description="LegalIR LoRA Reranker Trainer")
    parser.add_argument("--config", type=str, default="configs/experiments/reranker_lora.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--loss-type", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train_reranker(
        config_path=args.config,
        fold=args.fold,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        base_model_name=args.base_model,
        loss_type=args.loss_type,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
