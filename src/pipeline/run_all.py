from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import pandas as pd
import yaml

from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths
from src.dataset.validator import validate_canonical_dataset
from src.evaluation.benchmark import build_memory_rows
from src.evaluation.submission import package_submission, validate_submission
from src.pipeline.predict import LegalIRPipeline
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import ReciprocalRankFusion, LightGBMRanker
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import QuestionMemory


def run_all(
    config_path: str | Path = "configs/pipeline.yaml",
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    offline: bool = True,
    use_reranker: bool = False,
    use_fusion: str = "rrf",  # rrf | lightgbm
) -> Path:
    paths = ProjectPaths.from_repo()
    cfg = load_pipeline_config(Path(config_path))

    canonical_dir = paths.canonical
    if not (canonical_dir / "documents.parquet").exists():
        print(f"Building canonical v2 dataset at {canonical_dir}...")
        from src.dataset.build_canonical import build_canonical_package
        build_canonical_package(
            raw_contexts_dir=paths.repo / cfg.get("dataset", {}).get("raw_zip", "artifacts/shared/raw/selected-contexts.zip"),
            train_json_path=paths.repo / cfg.get("dataset", {}).get("train_json", "artifacts/shared/raw/train.json"),
            output_dir=canonical_dir,
        )

    val_report = validate_canonical_dataset(canonical_dir)
    if not val_report["is_valid"]:
        raise ValueError(f"Canonical dataset failed validation: {val_report['errors']}")

    print("Loading canonical dataset...")
    docs_df = pd.read_parquet(canonical_dir / "documents.parquet")
    chunks_df = pd.read_parquet(canonical_dir / "chunks.parquet")
    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(canonical_dir / "qrels_train.parquet")

    queries_dict = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))
    qrels_dict = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()
    docs_dict = {str(r["doc_id"]): r for r in docs_df.to_dict(orient="records")}

    # 1. BM25
    bm25_path = paths.local_indexes / "bm25" / "bm25_micro_index.pkl"
    if bm25_path.exists():
        print(f"Loading BM25 index from {bm25_path}...")
        bm25 = BM25MicroRetriever.load(bm25_path)
    else:
        print("Fitting BM25 micro-chunk index...")
        micro_chunks = chunks_df[chunks_df["granularity"] == "micro"].to_dict(orient="records")
        bm25 = BM25MicroRetriever().fit(micro_chunks, show_progress=True)
        bm25.save(bm25_path)

    # 2. Exact Matcher
    print("Initializing ExactMatcher...")
    exact = ExactMatcher(docs_df.to_dict(orient="records"))

    # 3. Question Memory (trained on all queries for final inference)
    print("Initializing full Question Memory...")
    all_train_qids = list(queries_dict.keys())
    memory_rows = build_memory_rows(all_train_qids, queries_dict, qrels_dict)
    memory = QuestionMemory(memory_rows, min_similarity=0.82)

    # 4. Hybrid Search Engine
    hybrid_engine = HybridSearchEngine(
        bm25_retriever=bm25,
        exact_matcher=exact,
        question_memory=memory,
        dense_retriever=None,
    )

    # 5. Evidence Pack Builder
    macro_chunks = chunks_df[chunks_df["granularity"] == "macro"].to_dict(orient="records")
    evidence_builder = EvidencePackBuilder(macro_chunks=macro_chunks, doc_metadata=docs_dict)

    # 6. Reranker (optional)
    reranker = None
    if use_reranker:
        hf_manifest = paths.local_models / "huggingface" / "manifest.json"
        reranker_model = "BAAI/bge-reranker-v2-m3"
        if hf_manifest.exists():
            hf_data = json.loads(hf_manifest.read_text(encoding="utf-8"))
            if reranker_model in hf_data:
                reranker_model = hf_data[reranker_model]["path"]
        reranker = CrossEncoderReranker(model_name=reranker_model)

    # 7. Fusion ranker
    ranker = ReciprocalRankFusion()
    if use_fusion == "lightgbm":
        model_file = paths.repo / "artifacts" / "local" / "training" / "fusion" / "model_full.txt"
        if model_file.exists():
            ranker = LightGBMRanker(model_file=model_file)

    selector = TopKSelector(max_k=5)

    pipeline = LegalIRPipeline(
        hybrid_engine=hybrid_engine,
        evidence_builder=evidence_builder,
        reranker=reranker,
        ranker=ranker,
        selector=selector,
        candidate_k=cfg.get("retrieval", {}).get("hybrid_union", {}).get("top_k_candidates", 150),
        rerank_k=cfg.get("retrieval", {}).get("hybrid_union", {}).get("top_k_for_rerank", 50),
    )

    # 8. Load input queries
    input_path = Path(input_file) if input_file else paths.repo / cfg.get("dataset", {}).get("public_json", "artifacts/shared/raw/public-official.json")
    if not input_path.exists():
        # Fallback to train query sample if public test not found
        input_path = paths.repo / "public-official.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Input query file not found at {input_path}")

    print(f"Loading queries from {input_path}...")
    raw_query_data = json.loads(input_path.read_text(encoding="utf-8"))
    query_texts = {}
    for qid, qobj in raw_query_data.items():
        if isinstance(qobj, dict):
            query_texts[str(qid)] = qobj.get("question", "")
        else:
            query_texts[str(qid)] = str(qobj)

    print(f"Generating predictions for {len(query_texts)} queries...")
    predictions = pipeline.predict_batch(query_texts, show_progress=True)

    # 9. Strict validation
    corpus_doc_ids = set(docs_df["doc_id"].astype(str))
    validate_submission(predictions, set(query_texts.keys()), corpus_doc_ids)
    print("Submission format validation PASSED (100% compliant with Codabench rules).")

    # 10. Package submission
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"submission_run_{timestamp_str}"
    run_dir = Path(output_dir) if output_dir else paths.local_runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "submission.json"
    zip_path = run_dir / "submission.zip"
    package_submission(predictions, json_path, zip_path)
    print(f"Successfully packaged {zip_path} ({zip_path.stat().st_size} bytes)")

    # Write run manifest
    manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp_str,
        "input_file": str(input_path),
        "total_queries": len(predictions),
        "submission_json": str(json_path),
        "submission_zip": str(zip_path),
        "use_reranker": use_reranker,
        "use_fusion": use_fusion,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config.snapshot.yaml").write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="LegalIR End-to-End Orchestrator")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--reranker", action="store_true", default=False)
    parser.add_argument("--fusion", type=str, default="rrf")
    args = parser.parse_args()

    run_all(
        config_path=args.config,
        input_file=args.input,
        output_dir=args.output_dir,
        offline=args.offline,
        use_reranker=args.reranker,
        use_fusion=args.fusion,
    )


if __name__ == "__main__":
    main()
