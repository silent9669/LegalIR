import os
import argparse
import pandas as pd
from src.retrieval.bm25_micro import BM25MicroRetriever

def build_retrieval_indexes(canonical_dir: str = "data/task1_canonical/v1", output_dir: str = "indexes"):
    os.makedirs(output_dir, exist_ok=True)
    chunks_path = os.path.join(canonical_dir, "chunks.parquet")
    print(f"Loading micro chunks from {chunks_path}...")

    chunks_df = pd.read_parquet(chunks_path)
    micro_chunks_df = chunks_df[chunks_df["granularity"] == "micro"]
    print(f"Total micro chunks to index: {len(micro_chunks_df)}")

    micro_chunks = micro_chunks_df[["chunk_id", "doc_id", "text_norm"]].to_dict(orient="records")

    bm25 = BM25MicroRetriever()
    bm25.fit(micro_chunks, show_progress=True)

    bm25_save_path = os.path.join(output_dir, "bm25_micro_index.pkl")
    print(f"Saving BM25 micro index to {bm25_save_path}...")
    bm25.save(bm25_save_path)
    print("BM25 micro index build completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_dir", type=str, default="data/task1_canonical/v1")
    parser.add_argument("--output_dir", type=str, default="indexes")
    args = parser.parse_args()

    build_retrieval_indexes(args.canonical_dir, args.output_dir)
