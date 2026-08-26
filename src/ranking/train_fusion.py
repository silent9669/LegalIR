from pathlib import Path
from typing import Any
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths
from src.evaluation.benchmark import build_memory_rows
from src.ranking.fusion import LightGBMRanker
from src.ranking.oof_features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, extract_candidate_features
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import QuestionMemory


def generate_fold_features(
    train_qids: list[str],
    queries_dict: dict[str, str],
    qrels_dict: dict[str, list[str]],
    hybrid_engine: HybridSearchEngine,
    candidate_k: int = 50,
) -> pd.DataFrame:
    all_dfs = []
    for qid in tqdm(train_qids, desc="Extracting candidates & features", leave=False):
        q_text = queries_dict.get(qid, "")
        gold_set = set(qrels_dict.get(qid, []))
        if not q_text or not gold_set:
            continue

        cands = hybrid_engine.search_candidates(q_text, exclude_qid=qid, top_k=candidate_k)
        if not cands:
            continue

        df = extract_candidate_features(query_id=qid, candidate_records=cands)
        df["target"] = df["doc_id"].apply(lambda d: 1 if d in gold_set else 0)
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def train_fusion_models(
    config_path: str | Path = "configs/pipeline.yaml",
    output_dir: str | Path | None = None,
    candidate_k: int = 50,
) -> dict[str, Any]:
    paths = ProjectPaths.from_repo()
    cfg = load_pipeline_config(Path(config_path))

    canonical_dir = paths.canonical
    output_dir = Path(output_dir) if output_dir else paths.repo / "artifacts" / "local" / "training" / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical data for fusion training from {canonical_dir}...")
    docs_df = pd.read_parquet(canonical_dir / "documents.parquet")
    chunks_df = pd.read_parquet(canonical_dir / "chunks.parquet")
    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(canonical_dir / "qrels_train.parquet")

    splits_dir = canonical_dir / "splits"
    random_5fold = json.loads((splits_dir / "random_5fold.json").read_text(encoding="utf-8"))

    queries_dict = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))
    qrels_dict = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()

    bm25_path = paths.local_indexes / "bm25" / "bm25_micro_index.pkl"
    if bm25_path.exists():
        bm25 = BM25MicroRetriever.load(bm25_path)
    else:
        micro_chunks = chunks_df[chunks_df["granularity"] == "micro"].to_dict(orient="records")
        bm25 = BM25MicroRetriever().fit(micro_chunks)

    exact = ExactMatcher(docs_df.to_dict(orient="records"))

    # Generate features for all 5 folds
    fold_feature_dfs = []
    for f_idx, fold_data in enumerate(random_5fold):
        val_qids = [str(x) for x in fold_data.get("val_query_ids", fold_data.get("val", []))]
        train_qids = [str(x) for x in fold_data.get("train_query_ids", fold_data.get("train", []))]

        # Memory uses only fold train queries
        mem_rows = build_memory_rows(train_qids, queries_dict, qrels_dict)
        memory = QuestionMemory(mem_rows, min_similarity=0.82)

        engine = HybridSearchEngine(
            bm25_retriever=bm25,
            exact_matcher=exact,
            question_memory=memory,
            dense_retriever=None,
        )

        print(f"Extracting OOF features for fold {f_idx + 1}/5 ({len(val_qids)} val queries)...")
        val_df = generate_fold_features(val_qids, queries_dict, qrels_dict, engine, candidate_k=candidate_k)
        val_df["fold"] = f_idx
        fold_feature_dfs.append(val_df)

    oof_df = pd.concat(fold_feature_dfs, ignore_index=True)
    oof_df.to_parquet(output_dir / "oof_features.parquet", index=False)

    print(f"Total OOF feature rows: {len(oof_df)}")

    # Train fold-specific models (out-of-fold training)
    trained_models = {}
    for f_idx in range(5):
        train_mask = oof_df["fold"] != f_idx
        train_data = oof_df[train_mask]

        # Group by query_id
        groups = train_data.groupby("query_id").size().values
        X = train_data[FEATURE_COLUMNS]
        y = train_data["target"].values

        ranker = LightGBMRanker()
        ranker.fit(X, y, groups)

        model_file = output_dir / f"model_fold_{f_idx}.txt"
        ranker.save(model_file)
        trained_models[f"fold_{f_idx}"] = str(model_file)
        print(f"Saved fold {f_idx} model to {model_file}")

    # Train full model on all OOF features
    full_groups = oof_df.groupby("query_id").size().values
    full_X = oof_df[FEATURE_COLUMNS]
    full_y = oof_df["target"].values

    full_ranker = LightGBMRanker()
    full_ranker.fit(full_X, full_y, full_groups)
    full_model_file = output_dir / "model_full.txt"
    full_ranker.save(full_model_file)
    trained_models["full"] = str(full_model_file)
    print(f"Saved full-data model to {full_model_file}")

    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "total_rows": len(oof_df),
        "models": trained_models,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Fusion models training completed successfully!")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="LegalIR LightGBM Fusion Trainer")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    train_fusion_models(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
