import os
import sys
import time
import pandas as pd
import numpy as np

from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.task1.memory import QuestionMemory

def build_all_indexes(
    data_dir: str = "artifacts/task1/data",
    index_dir: str = "artifacts/task1/indexes",
    device: str = None
):
    print("=" * 60)
    print("UIT-DSC 2026 Task 1: Building High-Recall Indexes")
    print("=" * 60)

    chunks_path = os.path.join(data_dir, "chunks.parquet")
    queries_path = os.path.join(data_dir, "queries_train.parquet")
    qrels_path = os.path.join(data_dir, "qrels_train.parquet")

    if not os.path.exists(chunks_path):
        print(f"Error: Chunks file not found at {chunks_path}")
        return

    print("Loading chunks from parquet...")
    df_chunks = pd.read_parquet(chunks_path)
    print(f"Total chunks loaded: {len(df_chunks):,}")

    # 1. Build Micro BM25 Index
    bm25_dir = os.path.join(index_dir, "bm25")
    print(f"\n[1/3] Building Fielded BM25 Index in {bm25_dir}...")
    micro_chunks = df_chunks[df_chunks["granularity"] == "micro"] if "granularity" in df_chunks.columns else df_chunks
    print(f"Micro chunks for BM25: {len(micro_chunks):,}")

    bm25_corpus = micro_chunks.to_dict("records")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    t0 = time.time()
    bm25.fit(bm25_corpus)
    bm25.save(bm25_dir)
    print(f"BM25 Index built and saved in {time.time() - t0:.2f}s")

    # 2. Build DEk21 Dense Macro Index
    dense_dir = os.path.join(index_dir, "dense_dek21")
    print(f"\n[2/3] Building DEk21 Dense Macro Index on {device or 'auto'}...")
    macro_chunks = df_chunks[df_chunks["granularity"] == "macro"] if "granularity" in df_chunks.columns else df_chunks
    print(f"Macro chunks for DEk21: {len(macro_chunks):,}")

    macro_corpus = macro_chunks.to_dict("records")
    dense = DEk21Retriever(model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device=device, dimension=768)
    t0 = time.time()
    dense.fit(macro_corpus, batch_size=128)
    dense.save(dense_dir)
    print(f"DEk21 Macro Index built and saved in {time.time() - t0:.2f}s")

    # 3. Build Question Memory Index
    mem_dir = os.path.join(index_dir, "question_memory")
    print(f"\n[3/3] Building Question Memory Index in {mem_dir}...")
    if os.path.exists(queries_path) and os.path.exists(qrels_path):
        df_queries = pd.read_parquet(queries_path)
        df_qrels = pd.read_parquet(qrels_path)

        queries_dict = {str(r["query_id"]): str(r["question_raw"]) for r in df_queries.to_dict("records")}
        qrels_dict = {}
        for r in df_qrels.to_dict("records"):
            qid = str(r["query_id"])
            did = str(r["doc_id"])
            if qid not in qrels_dict:
                qrels_dict[qid] = []
            qrels_dict[qid].append(did)

        memory = QuestionMemory(min_similarity=0.82)
        t0 = time.time()
        memory.fit(queries_dict, qrels_dict, dense_retriever=dense)
        memory.save(mem_dir)
        print(f"Question Memory Index built and saved in {time.time() - t0:.2f}s")

    print("\n" + "=" * 60)
    print("All Indexes Built Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    build_all_indexes()
