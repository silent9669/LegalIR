from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import HashingVectorizer

from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths
from src.dataset.validator import validate_canonical_dataset
from src.evaluation.benchmark import build_memory_rows
from src.evaluation.submission import package_submission, validate_submission
from src.pipeline.predict import LegalIRPipeline
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import LightGBMRanker, ReciprocalRankFusion
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import QuestionMemory


DEFAULT_SUBMISSION_DIR = Path("artifacts/task1/submissions")
DEFAULT_FALLBACK_DOC_IDS = ("2113", "58389", "84570")


def _repo_path(paths: ProjectPaths, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else paths.repo / path


def _local_model_path(paths: ProjectPaths, model_name: str) -> str | None:
    manifest_path = paths.local_models / "huggingface" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_info = manifest.get(model_name)
    if not isinstance(model_info, dict):
        return None
    model_path = model_info.get("path")
    if not model_path:
        return None
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path) if path.exists() else None


def _load_dense_branch(
    paths: ProjectPaths,
    dense_cfg: dict[str, Any],
    offline: bool,
    chunks: Any | None = None,
) -> DenseMacroRetriever | None:
    """Build or load the configured DEk21 macro-chunk index."""
    index_dir = paths.local_indexes / "dense"
    required_files = (index_dir / "embeddings.npy", index_dir / "chunks_meta.parquet")
    model_name = str(
        dense_cfg.get("model_name", "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    )
    configured_use_pyvi = bool(dense_cfg.get("use_pyvi", True))
    model_name_or_path = _local_model_path(paths, model_name)

    if not all(path.exists() for path in required_files):
        if chunks is None:
            print("Dense branch index unavailable; continuing with the remaining branches.")
            return None
        if offline and model_name_or_path is None:
            print(
                f"Dense branch index unavailable and {model_name} is not cached; "
                "skipping it in offline mode."
            )
            return None
        try:
            print(f"Building DEk21 dense index at {index_dir}...")
            DenseMacroRetriever.build(
                chunks=chunks,
                output_dir=index_dir,
                model_name=model_name_or_path or model_name,
                batch_size=int(dense_cfg.get("batch_size", 32)),
                max_length=int(dense_cfg.get("max_length", 512)),
                dimension=int(dense_cfg.get("dimension", 768)),
                use_pyvi=configured_use_pyvi,
            )
        except (OSError, ValueError, ImportError) as exc:
            if offline:
                print(f"Unable to build dense branch in offline mode: {exc}")
                return None
            raise
        if not all(path.exists() for path in required_files):
            print("Dense branch build did not produce a complete index; skipping it.")
            return None

    if offline and model_name_or_path is None:
        print(
            f"Dense branch index found, but {model_name} is not cached; "
            "skipping it in offline mode."
        )
        return None

    try:
        return DenseMacroRetriever.load(
            index_dir=index_dir,
            model_name=model_name_or_path or model_name,
            device="cpu" if offline else None,
            use_pyvi=configured_use_pyvi,
        )
    except (OSError, ValueError, ImportError) as exc:
        if offline:
            print(f"Unable to load dense branch in offline mode: {exc}")
            return None
        raise


def _build_offline_dense_fallback(
    docs_records: list[dict[str, Any]],
    chunks_df: pd.DataFrame,
) -> DenseMacroRetriever:
    """Build a small deterministic dense branch when no model index is present.

    The production path uses the configured DEk21 index.  Submission generation
    is also expected to work from a clean offline checkout, so this fallback
    keeps the fourth retrieval branch available without downloading a model or
    materializing embeddings for all 219k macro chunks.
    """
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=256,
        alternate_sign=False,
        norm="l2",
    )
    macro_columns = ["doc_id", "chunk_id", "text_norm", "text_raw"]
    available_columns = [column for column in macro_columns if column in chunks_df.columns]
    macro_df = chunks_df.loc[
        chunks_df["granularity"] == "macro", available_columns
    ]
    representative_text: dict[str, str] = {}
    representative_chunk: dict[str, str] = {}
    for row in macro_df.itertuples(index=False):
        doc_id = str(getattr(row, "doc_id"))
        if doc_id in representative_text:
            continue
        text = getattr(row, "text_norm", None) or getattr(row, "text_raw", None) or ""
        representative_text[doc_id] = str(text)
        representative_chunk[doc_id] = str(getattr(row, "chunk_id"))

    docs_by_id = {str(record["doc_id"]): record for record in docs_records}
    doc_ids = sorted(docs_by_id)
    texts = []
    chunk_ids = []
    for doc_id in doc_ids:
        record = docs_by_id[doc_id]
        metadata = " ".join(
            str(record.get(field) or "") for field in ("title", "legal_number", "name_raw")
        )
        texts.append(f"{metadata} {representative_text.get(doc_id, '')}".strip())
        chunk_ids.append(representative_chunk.get(doc_id, f"{doc_id}_dense"))

    embeddings = vectorizer.transform(texts).toarray().astype(np.float32, copy=False)

    def encode_queries(query_texts: list[str]) -> np.ndarray:
        return vectorizer.transform([str(text) for text in query_texts]).toarray()

    return DenseMacroRetriever.from_arrays(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        doc_ids=doc_ids,
        query_encoder=encode_queries,
        model_name="offline-hash-dense",
        use_pyvi=False,
    )


def _build_reranker(
    paths: ProjectPaths,
    offline: bool = True,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> CrossEncoderReranker:
    manifest_path = paths.local_models / "huggingface" / "manifest.json"
    local_model_path = _local_model_path(paths, model_name)
    return CrossEncoderReranker(
        model_name=local_model_path or model_name,
        manifest_path=manifest_path,
        local_files_only=offline,
    )


def _resolve_use_reranker(cfg: dict[str, Any], requested: bool | None) -> bool:
    if requested is not None:
        return bool(requested)
    ranking_cfg = cfg.get("ranking", {})
    reranker_cfg = ranking_cfg.get("reranker", {})
    return bool(reranker_cfg.get("enabled", True))


def run_all(
    config_path: str | Path = "configs/pipeline.yaml",
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    offline: bool = True,
    use_reranker: bool | None = None,
    use_fusion: str = "rrf",
) -> Path:
    """Run LegalIR retrieval and write a Codabench-compatible submission."""
    paths = ProjectPaths.from_repo()
    config_file = _repo_path(paths, config_path)
    cfg = load_pipeline_config(config_file)
    use_reranker = _resolve_use_reranker(cfg, use_reranker)

    canonical_dir = _repo_path(
        paths,
        cfg.get("paths", {}).get("canonical", "artifacts/shared/canonical/v2"),
    )
    if not (canonical_dir / "documents.parquet").exists():
        print(f"Building canonical v2 dataset at {canonical_dir}...")
        from src.dataset.build_canonical import build_canonical_package

        dataset_cfg = cfg.get("dataset", {})
        build_canonical_package(
            raw_contexts_dir=_repo_path(
                paths,
                dataset_cfg.get(
                    "raw_zip", "artifacts/shared/raw/selected-contexts.zip"
                ),
            ),
            train_json_path=_repo_path(
                paths, dataset_cfg.get("train_json", "artifacts/shared/raw/train.json")
            ),
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

    queries_dict = {
        str(query_id): question
        for query_id, question in zip(
            queries_df["query_id"], queries_df["question_norm"]
        )
    }
    qrels_dict = {
        str(query_id): [str(doc_id) for doc_id in doc_ids]
        for query_id, doc_ids in qrels_df.groupby("query_id")["doc_id"].apply(list).items()
    }
    docs_records = docs_df.to_dict(orient="records")
    docs_dict = {str(record["doc_id"]): record for record in docs_records}
    corpus_doc_ids = set(docs_dict)

    retrieval_cfg = cfg.get("retrieval", {})
    bm25_cfg = retrieval_cfg.get("bm25", {})
    bm25_path = paths.local_indexes / "bm25" / "bm25_micro_index.pkl"
    if bm25_path.exists():
        print(f"Loading BM25 index from {bm25_path}...")
        bm25 = BM25MicroRetriever.load(bm25_path)
    else:
        print("Fitting BM25 micro-chunk index...")
        micro_chunks = chunks_df[
            chunks_df["granularity"] == "micro"
        ].to_dict(orient="records")
        bm25 = BM25MicroRetriever(
            k1=float(bm25_cfg.get("k1", 1.5)),
            b=float(bm25_cfg.get("b", 0.75)),
            field_weights=dict(bm25_cfg.get("field_weights", {})),
        ).fit(micro_chunks, show_progress=True)
        bm25.save(bm25_path)

    print("Initializing ExactMatcher...")
    exact = ExactMatcher(docs_records)

    memory_cfg = retrieval_cfg.get("question_memory", {})
    memory_rows = build_memory_rows(
        list(queries_dict),
        queries_dict,
        qrels_dict,
    )
    memory = QuestionMemory(
        memory_rows,
        min_similarity=float(memory_cfg.get("min_similarity", 0.82)),
    )

    dense_cfg = retrieval_cfg.get("dense_macro", {})
    macro_chunks_df = chunks_df[chunks_df["granularity"] == "macro"]
    dense = _load_dense_branch(
        paths,
        dense_cfg,
        offline=offline,
        chunks=macro_chunks_df,
    )
    dense_mode = "index"
    if dense is None:
        dense = _build_offline_dense_fallback(docs_records, chunks_df)
        dense_mode = "offline_hash_fallback"
        print("Using deterministic offline dense fallback for the dense branch.")
    branch_weights = dict(
        retrieval_cfg.get("hybrid_union", {}).get(
            "initial_weights", HybridSearchEngine.DEFAULT_BRANCH_WEIGHTS
        )
    )
    hybrid_engine = HybridSearchEngine(
        bm25_retriever=bm25,
        exact_matcher=exact,
        question_memory=memory,
        dense_retriever=dense,
        branch_weights=branch_weights,
    )
    active_branches = ["bm25", "memory", "exact"]
    if dense is not None:
        active_branches.insert(1, "dense")
    print(f"Active retrieval branches: {', '.join(active_branches)}")

    evidence_cfg = cfg.get("ranking", {}).get("evidence", {})
    macro_chunks = macro_chunks_df.to_dict(orient="records")
    evidence_builder = EvidencePackBuilder(
        macro_chunks=macro_chunks,
        doc_metadata=docs_dict,
        max_chunks=int(evidence_cfg.get("max_chunks_per_doc", 2)),
        max_chars=int(evidence_cfg.get("max_chars", 1200)),
    )

    reranker_cfg = cfg.get("ranking", {}).get("reranker", {})
    reranker = (
        _build_reranker(
            paths,
            offline=offline,
            model_name=str(reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3")),
        )
        if use_reranker
        else None
    )
    fusion_cfg = cfg.get("ranking", {}).get("fusion", {})
    rrf_k = int(fusion_cfg.get("rrf_k", 60))
    if use_fusion == "rrf":
        ranker = ReciprocalRankFusion(
            k=rrf_k,
            w_bm25=float(branch_weights.get("bm25", 1.0)),
            w_dense=float(branch_weights.get("dense", 1.2)),
            w_memory=float(branch_weights.get("memory", 2.0)),
            w_exact=float(branch_weights.get("exact", 2.5)),
        )
    elif use_fusion == "lightgbm":
        model_file = paths.local / "training" / "fusion" / "model_full.txt"
        ranker = LightGBMRanker(model_file=model_file) if model_file.exists() else ReciprocalRankFusion(k=rrf_k)
    else:
        raise ValueError("use_fusion must be 'rrf' or 'lightgbm'")

    hybrid_candidate_cfg = retrieval_cfg.get("hybrid_union", {})
    selector_cfg = cfg.get("ranking", {}).get("selector", {})
    selector = TopKSelector(
        max_k=int(selector_cfg.get("max_k", 5)),
        min_k=int(selector_cfg.get("min_k", 1)),
    )
    fallback_doc_ids = [
        doc_id
        for doc_id in (*DEFAULT_FALLBACK_DOC_IDS, *sorted(corpus_doc_ids))
        if doc_id in corpus_doc_ids
    ]
    pipeline = LegalIRPipeline(
        hybrid_engine=hybrid_engine,
        evidence_builder=evidence_builder,
        reranker=reranker,
        ranker=ranker,
        selector=selector,
        candidate_k=int(hybrid_candidate_cfg.get("top_k_candidates", 150)),
        rerank_k=int(hybrid_candidate_cfg.get("top_k_for_rerank", 50)),
        fallback_doc_ids=fallback_doc_ids,
        valid_doc_ids=corpus_doc_ids,
    )

    dataset_cfg = cfg.get("dataset", {})
    if input_file is not None:
        input_path = _repo_path(paths, input_file)
    else:
        input_path = _repo_path(
            paths,
            dataset_cfg.get("public_json", "artifacts/shared/raw/public-official.json"),
        )
        if not input_path.exists():
            input_path = paths.repo / "public-official.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Input query file not found at {input_path}")

    print(f"Loading queries from {input_path}...")
    raw_query_data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_query_data, dict):
        raise ValueError("Official query file must contain an object keyed by query ID")
    query_texts = {
        str(query_id): (
            query.get("question", "") if isinstance(query, dict) else str(query)
        )
        for query_id, query in raw_query_data.items()
    }

    print(f"Generating predictions for {len(query_texts)} queries...")
    predictions = pipeline.predict_batch(query_texts, show_progress=True)
    validate_submission(predictions, set(query_texts), corpus_doc_ids)
    print("Submission format validation PASSED (100% compliant with Codabench rules).")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"submission_run_{timestamp_str}"
    run_dir = paths.local_runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_dir = _repo_path(
        paths,
        output_dir if output_dir is not None else DEFAULT_SUBMISSION_DIR,
    )
    json_path = submission_dir / "submission.json"
    zip_path = submission_dir / "submission.zip"
    package_submission(predictions, json_path, zip_path)

    # Keep the conventional top-level files in sync for Codabench tooling.
    top_level_json = paths.repo / "submission.json"
    top_level_zip = paths.repo / "submission.zip"
    if json_path.resolve() != top_level_json.resolve():
        package_submission(predictions, top_level_json, top_level_zip)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp_str,
        "input_file": str(input_path),
        "total_queries": len(predictions),
        "submission_json": str(json_path),
        "submission_zip": str(zip_path),
        "top_level_submission_json": str(top_level_json),
        "top_level_submission_zip": str(top_level_zip),
        "active_branches": active_branches,
        "dense_mode": dense_mode,
        "use_reranker": use_reranker,
        "use_fusion": use_fusion,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    config_snapshot = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    (run_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8"
    )

    print(f"Successfully packaged {zip_path} ({zip_path.stat().st_size} bytes)")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LegalIR End-to-End Orchestrator")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--reranker", action="store_true", default=None)
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
