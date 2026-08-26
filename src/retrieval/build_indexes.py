from pathlib import Path
import argparse
import json
import pandas as pd
from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths
from src.retrieval.bm25_micro import BM25MicroRetriever


def build_retrieval_indexes(
    canonical_dir: str | Path = "artifacts/shared/canonical/v2",
    output_dir: str | Path = "artifacts/local/indexes/bm25",
):
    canonical_dir = Path(canonical_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = canonical_dir / "chunks.parquet"
    print(f"Loading micro chunks from {chunks_path}...")

    chunks_df = pd.read_parquet(chunks_path)
    micro_chunks_df = chunks_df[chunks_df["granularity"] == "micro"]
    print(f"Total micro chunks to index: {len(micro_chunks_df)}")

    micro_chunks = micro_chunks_df.to_dict(orient="records")

    bm25 = BM25MicroRetriever()
    bm25.fit(micro_chunks, show_progress=True)

    bm25_save_path = output_dir / "bm25_micro_index.pkl"
    print(f"Saving BM25 micro index to {bm25_save_path}...")
    bm25.save(bm25_save_path)

    manifest = {
        "index_type": "bm25_micro",
        "total_micro_chunks": len(micro_chunks),
        "avg_len": bm25.avg_len,
        "vocabulary_size": len(bm25.idf),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"BM25 micro index build completed successfully in {output_dir}!")


def main():
    parser = argparse.ArgumentParser(description="LegalIR Retrieval Index Builder")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--canonical_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = {}
    if Path(args.config).exists():
        cfg = load_pipeline_config(Path(args.config))

    canonical_dir = args.canonical_dir or cfg.get("paths", {}).get("canonical", "artifacts/shared/canonical/v2")
    output_dir = args.output_dir or f"{cfg.get('paths', {}).get('local_indexes', 'artifacts/local/indexes')}/bm25"

    build_retrieval_indexes(canonical_dir, output_dir)


if __name__ == "__main__":
    main()
