import argparse
from pathlib import Path
from src.pipeline.run_all import run_all


def main():
    parser = argparse.ArgumentParser(description="LegalIR Predict Submission CLI (Compatibility Wrapper)")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--input_file", type=str, default="artifacts/shared/raw/public-official.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--output_zip", type=str, default=None)
    parser.add_argument("--canonical_dir", type=str, default=None)
    parser.add_argument("--bm25_index", type=str, default=None)
    parser.add_argument("--reranker", action="store_true", default=False)
    parser.add_argument("--fusion", type=str, default="rrf")
    args = parser.parse_args()

    run_all(
        config_path=args.config,
        input_file=args.input_file,
        use_reranker=args.reranker,
        use_fusion=args.fusion,
    )


if __name__ == "__main__":
    main()
