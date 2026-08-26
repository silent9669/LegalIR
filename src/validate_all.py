import argparse
from pathlib import Path
from src.evaluation.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="LegalIR Benchmark Compatibility Entrypoint")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--canonical_dir", type=str, default=None)
    parser.add_argument("--bm25_index", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--label", type=str, default="strict_baseline")
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        fold_limit=args.num_folds,
        label=args.label,
    )


if __name__ == "__main__":
    main()
