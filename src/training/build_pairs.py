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
    negatives_per_positive: int = 10,
    max_evidence_chunks: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build fold-safe positive and multi-band hard negative pairs for cross-encoder training.
    Saves reranker_pairs.parquet, retriever_pairs.parquet, and manifest.json with full provenance.
    """
    paths = ProjectPaths.from_repo()
    cfg = load_pipeline_config(Path(config_path)) if Path(config_path).exists() else {}

    canonical_dir = paths.canonical
    output_dir = Path(output_dir) if output_dir else paths.repo / "artifacts" / "local" / "training" / "pairs" / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical data for pair building from {canonical_dir}...")
    docs_df = pd.read_parquet(canonical_dir / "documents.parquet")
    chunks_df = pd.read_parquet(canonical_dir / "chunks.parquet")
    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(canonical_dir / "qrels_train.parquet")

    splits_dir = canonical_dir / "splits"
    random_5fold_path = splits_dir / "random_5fold.json"
    if random_5fold_path.exists():
        random_5fold = json.loads(random_5fold_path.read_text(encoding="utf-8"))
        fold_info = random_5fold[fold] if isinstance(random_5fold, list) and fold < len(random_5fold) else random_5fold.get(str(fold), {})
        train_qids = [str(x) for x in fold_info.get("train_query_ids", fold_info.get("train", []))]
    else:
        # Fallback to all queries if no split file
        train_qids = [str(x) for x in queries_df["query_id"].unique()]

    if limit is not None:
        train_qids = train_qids[:limit]

    queries_dict = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))
    qrels_dict = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()
    docs_dict = {str(r["doc_id"]): r for r in docs_df.to_dict(orient="records")}

    macro_chunks = chunks_df[chunks_df["granularity"] == "macro"].to_dict(orient="records")
    localizer = PositiveLocalizer(macro_chunks)
    evidence_builder = EvidencePackBuilder(macro_chunks=macro_chunks, doc_metadata=docs_dict)

    # Load or fit BM25 and exact matcher
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

    # Build false-negative blacklist from duplicate groups
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

    retriever_rows: list[dict[str, Any]] = []
    reranker_rows: list[dict[str, Any]] = []

    print(f"Building training pairs for {len(train_qids)} queries in fold {fold}...")
    for qid in tqdm(train_qids, desc=f"Mining pairs fold {fold}"):
        q_text = queries_dict.get(qid, "")
        gold_ids = qrels_dict.get(qid, [])
        if not gold_ids or not q_text:
            continue

        # Multi-band candidate generation
        # 1. Exact matches
        exact_cands = exact.match(q_text, top_k=10) if exact else []
        # 2. BM25 top candidates
        bm25_cands = bm25.search(q_text, top_k=50) if bm25 else []
        # 3. Question memory candidates (fold-safe, excludes current qid)
        mem_cands = memory.query(q_text, exclude_qid=qid, top_k=10) if memory else []
        # 4. Hybrid pool
        hybrid_cands = hybrid_engine.search_candidates(q_text, exclude_qid=qid, top_k=80)

        # Medium negatives from lower-ranked hybrid candidates (ranks 20-80)
        medium_cands = hybrid_cands[20:] if len(hybrid_cands) > 20 else []

        candidates_by_source = {
            "exact": [{"doc_id": c["doc_id"], "score": c.get("exact_score", 1.0), "rank": i + 1} for i, c in enumerate(exact_cands)],
            "bm25": [{"doc_id": c["doc_id"], "score": c.get("bm25_score", 0.0), "rank": i + 1} for i, c in enumerate(bm25_cands[:30])],
            "memory": [{"doc_id": c["doc_id"], "score": c.get("similarity", 0.0), "rank": i + 1} for i, c in enumerate(mem_cands)],
            "hybrid": [{"doc_id": c["doc_id"], "score": c.get("rrf_score", 0.0), "rank": i + 1} for i, c in enumerate(hybrid_cands[:30])],
            "medium_neg": [{"doc_id": c["doc_id"], "score": c.get("rrf_score", 0.0), "rank": i + 21} for i, c in enumerate(medium_cands)],
        }

        mined_neg_records = miner.mine_multi_band_negatives(
            query_id=qid,
            candidates_by_source=candidates_by_source,
            gold_doc_ids=gold_ids,
            per_source_limits={"exact": 2, "bm25": 4, "memory": 2, "hybrid": 4, "medium_neg": 3},
            max_total=negatives_per_positive * len(gold_ids),
        )

        for gold_id in gold_ids:
            pos_chunk = localizer.localize(q_text, gold_id)
            pos_chunk_id = pos_chunk.get("chunk_id") if pos_chunk else None
            pos_evidence = evidence_builder.build_pack(q_text, gold_id, max_chunks=max_evidence_chunks)

            # Positive pair for reranker
            reranker_rows.append({
                "query_id": qid,
                "query_text": q_text,
                "doc_id": gold_id,
                "label": 1.0,
                "negative_source": "gold",
                "retrieval_rank": 0,
                "retrieval_score": 1.0,
                "evidence_chunk_ids": json.dumps([pos_chunk_id] if pos_chunk_id else []),
                "evidence_text": pos_evidence,
                "fold": fold,
            })

            for neg_record in mined_neg_records:
                neg_id = str(neg_record["doc_id"])
                neg_chunk = localizer.localize(q_text, neg_id)
                neg_chunk_id = neg_chunk.get("chunk_id") if neg_chunk else None
                neg_evidence = evidence_builder.build_pack(q_text, neg_id, max_chunks=max_evidence_chunks)

                retriever_rows.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "pos_doc_id": gold_id,
                    "pos_chunk_id": pos_chunk_id,
                    "neg_doc_id": neg_id,
                    "neg_chunk_id": neg_chunk_id,
                    "fold": fold,
                })

                reranker_rows.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "doc_id": neg_id,
                    "label": 0.0,
                    "negative_source": neg_record.get("negative_source", "hybrid"),
                    "retrieval_rank": int(neg_record.get("retrieval_rank", 1)),
                    "retrieval_score": float(neg_record.get("retrieval_score", 0.0)),
                    "evidence_chunk_ids": json.dumps([neg_chunk_id] if neg_chunk_id else []),
                    "evidence_text": neg_evidence,
                    "fold": fold,
                })

    retriever_df = pd.DataFrame(retriever_rows)
    reranker_df = pd.DataFrame(reranker_rows)

    retriever_df.to_parquet(output_dir / "retriever_pairs.parquet", index=False)
    reranker_df.to_parquet(output_dir / "reranker_pairs.parquet", index=False)

    stats = miner.get_stats()
    pos_count = sum(1 for r in reranker_rows if r["label"] == 1.0)
    neg_count = sum(1 for r in reranker_rows if r["label"] == 0.0)

    manifest = {
        "fold": fold,
        "total_queries": len(train_qids),
        "positive_pairs_count": pos_count,
        "negative_pairs_count": neg_count,
        "total_pairs_count": len(reranker_df),
        "retriever_pairs_count": len(retriever_df),
        "counts_by_negative_source": stats["mined_counts_by_source"],
        "excluded_duplicate_cases_count": stats["excluded_duplicates_count"],
        "excluded_gold_cases_count": stats["excluded_golds_count"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {pos_count} positives, {neg_count} negatives in {output_dir}")

    return retriever_df, reranker_df


def main():
    parser = argparse.ArgumentParser(description="LegalIR Training Pairs Generator")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--max-chunks", type=int, default=2)
    args = parser.parse_args()

    build_training_pairs(
        config_path=args.config,
        fold=args.fold,
        output_dir=args.output_dir,
        limit=args.limit,
        negatives_per_positive=args.negatives,
        max_evidence_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
