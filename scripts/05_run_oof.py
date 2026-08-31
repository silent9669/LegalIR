"""Run full 5-fold out-of-fold (OOF) cross-validation for LegalIR."""

import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.oof_runner import OOFRunner


def run_oof_validation(
    data_dir: str = "artifacts/task1/data",
    index_dir: str = "artifacts/task1/indexes",
    splits_path: str = "artifacts/shared/canonical/v2/splits/random_5fold.json",
    output_dir: str = "artifacts/local/cv",
    num_folds: int = 5,
    sample_size: int | None = None,
    candidate_k: int = 150,
    rerank_k: int = 50,
    use_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    device: str | None = None,
    smoke: bool = False,
    doc_disjoint: bool = False,
):
    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=index_dir,
        splits_path=splits_path,
        output_dir=output_dir,
        num_folds=num_folds,
        candidate_k=candidate_k,
        rerank_k=rerank_k,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        device=device,
        smoke=smoke or (sample_size is not None),
        smoke_sample_size=sample_size if sample_size is not None else 20,
        doc_disjoint=doc_disjoint,
    )
    return runner.run()


def main():
    parser = argparse.ArgumentParser(description="LegalIR 5-Fold OOF Runner")
    parser.add_argument("--data-dir", type=str, default="artifacts/task1/data")
    parser.add_argument("--index-dir", type=str, default="artifacts/task1/indexes")
    parser.add_argument("--splits", type=str, default="artifacts/shared/canonical/v2/splits/random_5fold.json")
    parser.add_argument("--output-dir", type=str, default="artifacts/local/cv")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=150)
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke evaluation on 20 queries/fold")
    parser.add_argument("--doc-disjoint", action="store_true", help="Run document-disjoint robustness split evaluation")
    args = parser.parse_args()

    run_oof_validation(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        splits_path=args.splits,
        output_dir=args.output_dir,
        num_folds=args.num_folds,
        sample_size=args.sample_size,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        device=args.device,
        smoke=args.smoke,
        doc_disjoint=args.doc_disjoint,
    )


if __name__ == "__main__":
    main()
