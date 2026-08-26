from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import pandas as pd
from tqdm import tqdm

from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths
from src.evaluation.benchmark import build_memory_rows
from src.ranking.evidence_pack import EvidencePackBuilder
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import QuestionMemory
from src.training.hard_negative_miner import HardNegativeMiner
from src.training.positive_localizer import PositiveLocalizer


def build_training_pairs(
    config_path: str | Path = "configs/pipeline.yaml",
    fold: int = 0,
    output_dir: str | Path | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ProjectPaths.from_repo()
    cfg = load_pipeline_config(Path(config_path))

    canonical_dir = paths.canonical
    output_dir = Path(output_dir) if output_dir else paths.repo / "artifacts" / "local" / "training" / "pairs" / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical data for pair building from {canonical_dir}...")
    docs_df = pd.read_parquet(canonical_dir / "documents.parquet")
    chunks_df = pd.read_parquet(canonical_dir / "chunks.parquet")
    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(canonical_dir / "qrels_train.parquet")

    splits_dir = canonical_dir / "splits"
    random_5fold = json.loads((splits_dir / "random_5fold.json").read_text(encoding="utf-8"))
    fold_info = random_5fold[fold]
    train_qids = [str(x) for x in fold_info.get("train_query_ids", fold_info.get("train", []))]

    if limit is not None:
        train_qids = train_qids[:limit]

    queries_dict = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))
    qrels_dict = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()
    docs_dict = {str(r["doc_id"]): r for r in docs_df.to_dict(orient="records")}

    macro_chunks = chunks_df[chunks_df["granularity"] == "macro"].to_dict(orient="records")
    localizer = PositiveLocalizer(macro_chunks)
    evidence_builder = EvidencePackBuilder(macro_chunks=macro_chunks, doc_metadata=docs_dict)

    # Load BM25 and exact matcher
    bm25_path = paths.local_indexes / "bm25" / "bm25_micro_index.pkl"
    if bm25_path.exists():
        bm25 = BM25MicroRetriever.load(bm25_path)
    else:
        micro_chunks = chunks_df[chunks_df["granularity"] == "micro"].to_dict(orient="records")
        bm25 = BM25MicroRetriever().fit(micro_chunks)

    exact = ExactMatcher(docs_df.to_dict(orient="records"))
    memory_rows = build_memory_rows(train_qids, queries_dict, qrels_dict)
    memory = QuestionMemory(memory_rows, min_similarity=0.82)

    hybrid_engine = HybridSearchEngine(
        bm25_retriever=bm25,
        exact_matcher=exact,
        question_memory=memory,
        dense_retriever=None,
    )

    # Build false-negative blacklist from near duplicates
    dup_path = canonical_dir / "duplicate_groups.json"
    dup_groups = json.loads(dup_path.read_text(encoding="utf-8")) if dup_path.exists() else {}
    doc_to_dups = defaultdict(set)
    for group in dup_groups.values():
        doc_set = set(str(x) for x in group)
        for did in doc_set:
            doc_to_dups[did].update(doc_set)

    query_blacklist = defaultdict(set)
    for qid in train_qids:
        for gold_id in qrels_dict.get(qid, []):
            query_blacklist[qid].update(doc_to_dups.get(gold_id, set()))

    miner = HardNegativeMiner(false_negative_blacklist=query_blacklist)

    retriever_rows = []
    reranker_rows = []

    print(f"Building training pairs for {len(train_qids)} queries in fold {fold}...")
    for qid in tqdm(train_qids, desc="Mining pairs"):
        q_text = queries_dict.get(qid, "")
        gold_ids = qrels_dict.get(qid, [])
        if not gold_ids or not q_text:
            continue

        cands = hybrid_engine.search_candidates(q_text, exclude_qid=qid, top_k=50)
        neg_ids = miner.mine_negatives(qid, cands, gold_ids, max_negatives=10)

        for gold_id in gold_ids:
            pos_chunk = localizer.localize(q_text, gold_id)
            pos_chunk_id = pos_chunk.get("chunk_id") if pos_chunk else None
            pos_evidence = evidence_builder.format_evidence_text(pos_chunk) if pos_chunk else f"Văn bản {gold_id}"

            # Positive pair for reranker
            reranker_rows.append({
                "query_id": qid,
                "query_text": q_text,
                "doc_id": gold_id,
                "chunk_id": pos_chunk_id,
                "evidence_text": pos_evidence,
                "label": 1.0,
            })

            for neg_id in neg_ids:
                retriever_rows.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "pos_doc_id": gold_id,
                    "pos_chunk_id": pos_chunk_id,
                    "neg_doc_id": neg_id,
                })

                neg_chunk = localizer.localize(q_text, neg_id)
                neg_chunk_id = neg_chunk.get("chunk_id") if neg_chunk else None
                neg_evidence = evidence_builder.format_evidence_text(neg_chunk) if neg_chunk else f"Văn bản {neg_id}"

                reranker_rows.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "doc_id": neg_id,
                    "chunk_id": neg_chunk_id,
                    "evidence_text": neg_evidence,
                    "label": 0.0,
                })

    retriever_df = pd.DataFrame(retriever_rows)
    reranker_df = pd.DataFrame(reranker_rows)

    retriever_df.to_parquet(output_dir / "retriever_pairs.parquet", index=False)
    reranker_df.to_parquet(output_dir / "reranker_pairs.parquet", index=False)

    manifest = {
        "fold": fold,
        "total_queries": len(train_qids),
        "retriever_pairs_count": len(retriever_df),
        "reranker_pairs_count": len(reranker_df),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(retriever_df)} retriever pairs and {len(reranker_df)} reranker pairs in {output_dir}")

    return retriever_df, reranker_df


def main():
    parser = argparse.ArgumentParser(description="LegalIR Training Pairs Generator")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    build_training_pairs(
        config_path=args.config,
        fold=args.fold,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
