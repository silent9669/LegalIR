from pathlib import Path
from typing import Any
import argparse
import json
import pandas as pd
from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.dense_macro import DenseMacroRetriever


def build_bm25_index(
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


def build_dense_index(
    canonical_dir: str | Path = "artifacts/shared/canonical/v2",
    output_dir: str | Path = "artifacts/local/indexes/dense",
    model_name_or_path: str = "BAAI/bge-m3",
    batch_size: int = 32,
    max_length: int = 512,
):
    canonical_dir = Path(canonical_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = canonical_dir / "chunks.parquet"
    print(f"Loading macro chunks from {chunks_path}...")

    chunks_df = pd.read_parquet(chunks_path)
    macro_chunks_df = chunks_df[chunks_df["granularity"] == "macro"]
    print(f"Total macro chunks to encode: {len(macro_chunks_df)}")

    macro_chunks = macro_chunks_df.to_dict(orient="records")

    DenseMacroRetriever.build(
        chunks=macro_chunks,
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser(description="LegalIR Retrieval Index Builder")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--bm25", action="store_true", help="Build BM25 micro index")
    parser.add_argument("--dense", action="store_true", help="Build Dense macro index")
    args = parser.parse_args()

    paths = ProjectPaths.from_repo()
    cfg = {}
    if Path(args.config).exists():
        cfg = load_pipeline_config(Path(args.config))

    canonical_dir = paths.canonical

    build_all = not args.bm25 and not args.dense

    if args.bm25 or build_all:
        bm25_out = paths.local_indexes / "bm25"
        build_bm25_index(canonical_dir, bm25_out)

    if args.dense or build_all:
        dense_out = paths.local_indexes / "dense"
        # Find cached model path if available
        hf_manifest_path = paths.local_models / "huggingface" / "manifest.json"
        model_path = "BAAI/bge-m3"
        if hf_manifest_path.exists():
            hf_data = json.loads(hf_manifest_path.read_text(encoding="utf-8"))
            if "BAAI/bge-m3" in hf_data:
                model_path = hf_data["BAAI/bge-m3"]["path"]

        build_dense_index(
            canonical_dir=canonical_dir,
            output_dir=dense_out,
            model_name_or_path=model_path,
            batch_size=32,
            max_length=512,
        )


if __name__ == "__main__":
    main()
