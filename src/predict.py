import os
import json
import zipfile
from retrievers.bm25_retriever import BM25Retriever
from retrievers.exact_matcher import ExactMatcher
from retrievers.memory_retriever import TrainQuestionMemory

def generate_predictions(
    input_file: str = "public-official.json",
    output_json: str = "submission.json",
    output_zip: str = "submission.zip",
    bm25_index_path: str = "data/bm25_index.pkl",
    meta_path: str = "data/doc_metadata.json",
    train_path: str = "train.json",
    top_k: int = 5
):
    print(f"Loading input queries from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    print("Loading indices and retrievers...")
    bm25 = BM25Retriever.load(bm25_index_path)
    exact = ExactMatcher(meta_path)
    memory = TrainQuestionMemory(train_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        valid_doc_ids = set(json.load(f).keys())

    predictions = {}
    print(f"Generating predictions for {len(input_data)} queries...")

    for idx, (qid, qobj) in enumerate(input_data.items()):
        q_text = qobj.get("question", "") if isinstance(qobj, dict) else str(qobj)

        # 1. BM25 retrieval
        b_scores = bm25.retrieve(q_text)
        sorted_b = sorted(b_scores.items(), key=lambda x: x[1], reverse=True)
        top_b_docs = [doc for doc, _ in sorted_b[:120]]

        # 2. Exact Matcher
        e_scores = exact.match(q_text)

        # 3. Train Memory
        m_scores = memory.retrieve(q_text)

        # 4. RRF Fusion
        all_candidate_docs = set(top_b_docs) | set(e_scores.keys()) | set(m_scores.keys())
        rrf_scores = {}
        b_ranks = {doc: rank for rank, doc in enumerate(top_b_docs, 1)}

        for doc in all_candidate_docs:
            score = 0.0
            if doc in b_ranks:
                score += 1.0 / (60 + b_ranks[doc])
            if doc in e_scores:
                score += (e_scores[doc] / 15.0) * (1.0 / 25)
            if doc in m_scores:
                score += (m_scores[doc] / 10.0) * (1.0 / 35)
            rrf_scores[doc] = score

        sorted_candidates = [doc for doc, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]

        # Filter valid corpus doc_ids and deduplicate preserving order
        final_docs = []
        seen = set()
        for doc in sorted_candidates:
            doc_str = str(doc)
            if doc_str in valid_doc_ids and doc_str not in seen:
                seen.add(doc_str)
                final_docs.append(doc_str)
            if len(final_docs) == top_k:
                break

        # Fallback if empty
        if not final_docs:
            for fallback_doc, _ in sorted_b[:top_k]:
                if str(fallback_doc) not in seen:
                    final_docs.append(str(fallback_doc))

        # Format according to scoring.py requirements
        predictions[str(qid)] = {
            "answer": final_docs[:top_k]
        }

        if (idx + 1) % 50 == 0 or (idx + 1) == len(input_data):
            print(f"Processed {idx + 1}/{len(input_data)} queries...")

    # Write submission.json
    print(f"\nWriting predictions to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Sanity checks
    print("Running sanity checks on predictions...")
    assert len(predictions) == len(input_data), f"Mismatch count: {len(predictions)} vs {len(input_data)}"
    for qid, obj in predictions.items():
        ans = obj.get("answer")
        assert isinstance(ans, list), f"Query {qid} answer is not a list"
        assert 1 <= len(ans) <= 5, f"Query {qid} has invalid count: {len(ans)}"
        assert len(ans) == len(set(ans)), f"Query {qid} contains duplicate IDs"
        assert all(isinstance(d, str) for d in ans), f"Query {qid} contains non-string IDs"
        assert all(d in valid_doc_ids for d in ans), f"Query {qid} contains invalid corpus doc_id"

    print("All sanity checks passed successfully!")

    # Zip submission.json into submission.zip
    print(f"Packaging {output_json} into {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, arcname=os.path.basename(output_json))

    print(f"Submission package created: {output_zip} (size: {os.path.getsize(output_zip)} bytes)")
    return predictions

if __name__ == "__main__":
    generate_predictions()
