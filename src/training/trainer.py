from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.training.losses import get_loss_function


class RerankerPairDataset(Dataset):
    """PyTorch Dataset for (query, evidence, label) pairs."""

    def __init__(self, data: pd.DataFrame | list[dict[str, Any]]):
        if isinstance(data, pd.DataFrame):
            self.records = data.to_dict(orient="records")
        else:
            self.records = list(data)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        query = str(item.get("query_text") or item.get("question") or "")
        evidence = str(item.get("evidence_text") or item.get("text") or item.get("text_raw") or "")
        label = float(item.get("label", 0.0))
        return {
            "query": query,
            "evidence": evidence,
            "label": label,
            "query_id": str(item.get("query_id", "")),
            "doc_id": str(item.get("doc_id", "")),
        }


class RerankerPairCollator:
    """Collator for dynamic batching and padding with tokenizer."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [item["query"] for item in batch]
        passages = [item["evidence"] for item in batch]
        labels = [item["label"] for item in batch]

        features = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features["labels"] = torch.tensor(labels, dtype=torch.float32)
        features["query_ids"] = [item["query_id"] for item in batch]
        features["doc_ids"] = [item["doc_id"] for item in batch]
        return features


class RerankerGroupDataset(Dataset):
    """
    Dataset grouping 1 positive and N negatives per query for pairwise/listwise ranking.
    """

    def __init__(self, data: pd.DataFrame | list[dict[str, Any]], max_negatives_per_group: int = 7):
        records = data.to_dict(orient="records") if isinstance(data, pd.DataFrame) else list(data)

        # Group by query_id
        groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"query": "", "positives": [], "negatives": []})
        for r in records:
            qid = str(r.get("query_id", ""))
            query = str(r.get("query_text") or r.get("question") or "")
            evidence = str(r.get("evidence_text") or r.get("text") or "")
            label = float(r.get("label", 0.0))
            doc_id = str(r.get("doc_id", ""))

            groups[qid]["query"] = query
            if label > 0.5:
                groups[qid]["positives"].append({"doc_id": doc_id, "evidence": evidence})
            else:
                groups[qid]["negatives"].append({"doc_id": doc_id, "evidence": evidence})

        self.items: list[dict[str, Any]] = []
        for qid, g in groups.items():
            if not g["positives"] or not g["negatives"]:
                continue
            for pos in g["positives"]:
                negs = g["negatives"][:max_negatives_per_group]
                self.items.append({
                    "query_id": qid,
                    "query": g["query"],
                    "positive": pos,
                    "negatives": negs,
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class RerankerGroupCollator:
    """Collator that flattens groups for cross-encoder inference and returns group batching metadata."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_queries = []
        all_passages = []
        group_sizes = []

        for item in batch:
            q = item["query"]
            # First item in group is positive
            all_queries.append(q)
            all_passages.append(item["positive"]["evidence"])

            # Followed by negatives
            for neg in item["negatives"]:
                all_queries.append(q)
                all_passages.append(neg["evidence"])

            group_sizes.append(1 + len(item["negatives"]))

        features = self.tokenizer(
            all_queries,
            all_passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features["group_sizes"] = group_sizes
        return features


def find_target_modules(model: nn.Module, requested_modules: list[str] | None = None) -> list[str]:
    """
    Dynamically discover matching linear layer names in the transformer model for LoRA.
    """
    linear_layer_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_name = name.split(".")[-1]
            linear_layer_names.add(layer_name)

    if requested_modules:
        matched = [m for m in requested_modules if m in linear_layer_names]
        if matched:
            return matched

    # Fallback heuristics for standard transformer architectures
    common_candidates = [
        ["query", "value"],
        ["q_proj", "v_proj"],
        ["query", "key", "value", "dense"],
        ["q_proj", "k_proj", "v_proj", "out_proj"],
        ["dense"],
    ]
    for cand_list in common_candidates:
        matched = [m for m in cand_list if m in linear_layer_names]
        if matched:
            return matched

    # Return all linear layer suffixes if specific names not matched
    return list(linear_layer_names) if linear_layer_names else ["linear"]


def setup_peft_model(
    model: nn.Module,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Applies PEFT LoRA to the model with verified target module inspection.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    matched_targets = find_target_modules(model, target_modules)
    if not matched_targets:
        raise ValueError("Could not find any matching linear layers in the base model for LoRA!")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=matched_targets,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    meta = {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": matched_targets,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": round(trainable_pct, 4),
    }
    return peft_model, meta


class RerankerTrainer:
    """
    Supervised Cross-Encoder Reranker Trainer with PEFT/LoRA, ranking loss objectives,
    and genuine weight update verification.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_data: pd.DataFrame | list[dict[str, Any]],
        val_data: pd.DataFrame | list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config or {}
        self.tokenizer = tokenizer
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.loss_type = self.config.get("loss_type", "bce")
        self.loss_fn = get_loss_function(
            self.loss_type,
            temperature=self.config.get("temperature", 1.0),
            margin=self.config.get("margin", 1.0),
            pos_weight=self.config.get("pos_weight", None),
        )

        self.batch_size = int(self.config.get("batch_size", 16))
        self.max_length = int(self.config.get("max_length", 512))
        self.learning_rate = float(self.config.get("learning_rate", 2e-5))
        self.weight_decay = float(self.config.get("weight_decay", 0.01))
        self.max_steps = self.config.get("max_steps", None)
        self.num_epochs = int(self.config.get("epochs", 2)) if self.max_steps is None else 1
        self.warmup_ratio = float(self.config.get("warmup_ratio", 0.1))
        self.gradient_accumulation_steps = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.fp16 = bool(self.config.get("fp16", True)) and self.device.type == "cuda"

        # LoRA setup
        use_lora = self.config.get("use_lora", True)
        self.peft_meta: dict[str, Any] = {}
        if use_lora:
            lora_cfg = self.config.get("lora", {})
            self.model, self.peft_meta = setup_peft_model(
                model=model,
                lora_r=lora_cfg.get("r", self.config.get("lora_r", 16)),
                lora_alpha=lora_cfg.get("lora_alpha", self.config.get("lora_alpha", 32)),
                lora_dropout=lora_cfg.get("lora_dropout", self.config.get("lora_dropout", 0.05)),
                target_modules=lora_cfg.get("target_modules", self.config.get("target_modules", None)),
            )
        else:
            self.model = model

        if self.config.get("gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "gradient_checkpointing_enable"):
                self.model.base_model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.model.to(self.device)

        # Build datasets and loaders
        if self.loss_type in ("listwise", "listwise_ce", "pairwise_logistic", "pairwise_margin"):
            self.train_dataset = RerankerGroupDataset(train_data)
            self.train_collator = RerankerGroupCollator(self.tokenizer, max_length=self.max_length)
            self.is_group_mode = True
        else:
            self.train_dataset = RerankerPairDataset(train_data)
            self.train_collator = RerankerPairCollator(self.tokenizer, max_length=self.max_length)
            self.is_group_mode = False

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_collator,
        )

        self.val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_ds = RerankerPairDataset(val_data)
            self.val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=RerankerPairCollator(self.tokenizer, max_length=self.max_length),
            )

    def train(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Execute the supervised fine-tuning training loop."""
        start_time = time.time()
        self.model.train()

        # Capture initial trainable parameter values for verification
        initial_params = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        weight_norm_before = sum(
            p.norm().item() for p in self.model.parameters() if p.requires_grad
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Compute total training steps
        steps_per_epoch = max(1, len(self.train_loader) // self.gradient_accumulation_steps)
        total_training_steps = (
            self.max_steps if self.max_steps is not None else steps_per_epoch * self.num_epochs
        )
        total_training_steps = max(1, total_training_steps)
        warmup_steps = int(total_training_steps * self.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        global_step = 0
        loss_history: list[float] = []
        accumulated_loss = 0.0
        optimizer.zero_grad()
        seen_query_ids = set()
        nonfinite_loss_count = 0

        # AMP GradScaler for stable mixed precision training on CUDA
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

        print(f"Starting training: {total_training_steps} steps, device={self.device}, fp16={self.fp16}")

        while global_step < total_training_steps:
            for batch_idx, batch in enumerate(self.train_loader):
                if global_step >= total_training_steps:
                    break

                for qid in batch.get("query_ids", []):
                    if qid:
                        seen_query_ids.add(str(qid))

                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor) and k != "labels"
                }

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                    outputs = self.model(**inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    logits = logits.view(-1)

                    if self.is_group_mode:
                        # Group ranking loss
                        group_sizes = batch.get("group_sizes", [])
                        current_offset = 0
                        group_losses = []
                        for g_size in group_sizes:
                            g_logits = logits[current_offset : current_offset + g_size]
                            if len(g_logits) > 1:
                                if self.loss_type in ("pairwise_logistic", "pairwise_margin"):
                                    pos_score = g_logits[0:1]
                                    neg_scores = g_logits[1:]
                                    g_loss = self.loss_fn(pos_score.expand_as(neg_scores), neg_scores)
                                else:
                                    g_loss = self.loss_fn(g_logits.unsqueeze(0), target_idx=0)
                                group_losses.append(g_loss)
                            current_offset += g_size
                        loss = torch.stack(group_losses).mean() if group_losses else torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        labels = batch["labels"].to(self.device)
                        loss = self.loss_fn(logits, labels)

                    if not torch.isfinite(loss):
                        nonfinite_loss_count += 1
                        raise RuntimeError(f"Non-finite training loss encountered: {loss.item()} at step {global_step}")

                    # Scale for gradient accumulation
                    loss_to_backprop = loss / self.gradient_accumulation_steps

                scaler.scale(loss_to_backprop).backward()
                accumulated_loss += loss.item()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError(f"Non-finite gradient norm encountered: {grad_norm} at step {global_step}")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    avg_loss = accumulated_loss / self.gradient_accumulation_steps
                    loss_history.append(float(avg_loss))
                    accumulated_loss = 0.0

                    if global_step % max(1, total_training_steps // 10) == 0 or global_step == total_training_steps:
                        print(f"Step {global_step}/{total_training_steps} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Post-training weight change verification
        param_diff = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in initial_params:
                diff = (param.detach().cpu() - initial_params[name]).abs().sum().item()
                param_diff += diff

        weight_norm_after = sum(
            p.norm().item() for p in self.model.parameters() if p.requires_grad
        )
        weight_norm_change = abs(weight_norm_after - weight_norm_before)

        # Validation evaluation
        val_metrics = self.evaluate() if self.val_loader is not None else {}

        elapsed_time = time.time() - start_time
        final_train_loss = loss_history[-1] if loss_history else 0.0

        total_input_queries = len(set(r.get("query_id") for r in getattr(self.train_dataset, "records", []))) if hasattr(self.train_dataset, "records") else len(seen_query_ids)
        actual_seen_count = len(seen_query_ids)
        coverage_pct = round((actual_seen_count / max(1, total_input_queries)) * 100.0, 2)

        report = {
            "status": "completed",
            "global_steps": global_step,
            "total_training_steps": total_training_steps,
            "training_time_sec": round(elapsed_time, 2),
            "final_train_loss": round(final_train_loss, 6),
            "loss_history_sample": [round(x, 4) for x in loss_history[:5] + loss_history[-5:]] if len(loss_history) > 10 else [round(x, 4) for x in loss_history],
            "trainable_params": self.peft_meta.get("trainable_params", sum(p.numel() for p in trainable_params)),
            "total_params": self.peft_meta.get("total_params", sum(p.numel() for p in self.model.parameters())),
            "trainable_percent": self.peft_meta.get("trainable_percent", 100.0),
            "param_diff": float(param_diff),
            "actual_unique_queries_seen": actual_seen_count,
            "actual_query_coverage_pct": coverage_pct,
            "actual_examples_seen": global_step * self.batch_size * self.gradient_accumulation_steps,
            "nonfinite_loss_count": nonfinite_loss_count,
            "weight_norm_before": float(weight_norm_before),
            "weight_norm_after": float(weight_norm_after),
            "weight_norm_change": float(weight_norm_change),
            "val_metrics": val_metrics,
            "loss_type": self.loss_type,
            "device": str(self.device),
        }

        # Save checkpoint if output_dir provided
        if output_dir:
            self.save_checkpoint(output_dir, report)

        return report

    def evaluate(self) -> dict[str, Any]:
        """Evaluate model on validation loader."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_val_loss = 0.0
        val_steps = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor) and k != "labels"
                }
                labels = batch["labels"].to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits.view(-1) if hasattr(outputs, "logits") else outputs[0].view(-1)

                loss = nn.BCEWithLogitsLoss()(logits, labels)
                total_val_loss += loss.item()
                val_steps += 1

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_predictions += (preds == labels).sum().item()
                total_samples += len(labels)

        self.model.train()
        avg_val_loss = total_val_loss / max(1, val_steps)
        accuracy = correct_predictions / max(1, total_samples)
        return {
            "val_loss": round(avg_val_loss, 6),
            "accuracy": round(accuracy, 4),
            "total_val_samples": total_samples,
        }

    def save_checkpoint(self, output_dir: str | Path, report: dict[str, Any] | None = None) -> None:
        """Save LoRA weights, tokenizer, and training manifest."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save PEFT adapter or model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(out_path))
        else:
            torch.save(self.model.state_dict(), out_path / "pytorch_model.bin")

        # Save tokenizer
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(out_path))

        # Save manifest
        if report is not None:
            manifest_file = out_path / "training_manifest.json"
            manifest_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Saved LoRA checkpoint and manifest to {out_path}")
