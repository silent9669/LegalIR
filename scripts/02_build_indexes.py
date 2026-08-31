import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.question_memory import TrainQuestionMemory, QuestionMemory

def build_all_indexes(
    data_dir: str = "artifacts/task1/data",
    index_dir: str = "artifacts/task1/indexes",
    device: str = None
):
    print("=" * 60)
    print("UIT-DSC 2026 Task 1: Building High-Recall Canonical Indexes")
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

    # 1. Build Fielded Legal BM25 Index (Branch A)
    bm25_dir = os.path.join(index_dir, "bm25")
    print(f"\n[1/4] Building Fielded Legal BM25 Micro Index in {bm25_dir}...")
    micro_chunks = df_chunks[df_chunks["granularity"] == "micro"] if "granularity" in df_chunks.columns else df_chunks
    print(f"Micro chunks for Legal BM25: {len(micro_chunks):,}")

    bm25_corpus = micro_chunks.to_dict("records")
    bm25 = BM25MicroRetriever(k1=1.5, b=0.75)
    t0 = time.time()
    bm25.fit(bm25_corpus)
    bm25.save(bm25_dir)
    print(f"Legal BM25 Index built and saved in {time.time() - t0:.2f}s")

    # 2. Build PyVi Segmented BM25 Index (Branch B)
    bm25_pyvi_dir = os.path.join(index_dir, "bm25_pyvi")
    print(f"\n[2/4] Building PyVi BM25 Micro Index in {bm25_pyvi_dir}...")
    bm25_pyvi = BM25PyViRetriever(k1=1.5, b=0.75)
    t0 = time.time()
    bm25_pyvi.fit(bm25_corpus)
    bm25_pyvi.save(bm25_pyvi_dir)
    print(f"PyVi BM25 Index built and saved in {time.time() - t0:.2f}s")

    # 3. Build DEk21 Dense Macro Index
    dense_dir = os.path.join(index_dir, "dense_dek21")
    print(f"\n[3/4] Building DEk21 Dense Macro Index on {device or 'auto'}...")
    macro_chunks = df_chunks[df_chunks["granularity"] == "macro"] if "granularity" in df_chunks.columns else df_chunks
    print(f"Macro chunks for DEk21: {len(macro_chunks):,}")

    macro_corpus = macro_chunks.to_dict("records")
    dense = DenseMacroRetriever(model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device=device, dimension=768)
    t0 = time.time()
    dense.fit(macro_corpus, batch_size=128)
    dense.save(dense_dir)
    print(f"DEk21 Macro Index built and saved in {time.time() - t0:.2f}s")

    # 4. Build Question Memory Index
    mem_dir = os.path.join(index_dir, "question_memory")
    print(f"\n[4/4] Building Question Memory Index in {mem_dir}...")
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

        memory = TrainQuestionMemory(min_similarity=0.82, dense_encoder=dense)
        t0 = time.time()
        memory.fit(queries_dict, qrels_dict)
        memory.save(mem_dir)
        print(f"Question Memory Index built and saved in {time.time() - t0:.2f}s")

    print("\n" + "=" * 60)
    print("All Canonical Indexes Built Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    build_all_indexes()
