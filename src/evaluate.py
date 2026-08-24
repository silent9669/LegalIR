import json
import numpy as np
from retrievers.bm25_retriever import BM25Retriever
from retrievers.exact_matcher import ExactMatcher
from retrievers.memory_retriever import TrainQuestionMemory

def compute_metrics(predictions: dict, ground_truths: dict):
    """Compute Recall@1, Recall@3, Recall@5, Recall@10, Candidate Recall@50/100, Precision@5."""
    r1, r3, r5, r10, r50, r100, p5 = [], [], [], [], [], [], []

    for qid, gold_docs in ground_truths.items():
        gold_set = set(str(x) for x in gold_docs)
        if not gold_set:
            continue

        preds = [str(x) for x in predictions.get(qid, [])]

        # Top K sets
        top1 = set(preds[:1])
        top3 = set(preds[:3])
        top5 = set(preds[:5])
        top10 = set(preds[:10])
        top50 = set(preds[:50])
        top100 = set(preds[:100])

        r1.append(len(gold_set & top1) / len(gold_set))
        r3.append(len(gold_set & top3) / len(gold_set))
        r5.append(len(gold_set & top5) / len(gold_set))
        r10.append(len(gold_set & top10) / len(gold_set))
        r50.append(len(gold_set & top50) / len(gold_set))
        r100.append(len(gold_set & top100) / len(gold_set))

        # Precision@5: if len(pred) > 5 or len == 0, score is 0
        if 0 < len(preds[:5]) <= 5:
            p5.append(len(gold_set & top5) / len(preds[:5]))
        else:
            p5.append(0.0)

    return {
        "Recall@1": np.mean(r1),
        "Recall@3": np.mean(r3),
        "Recall@5": np.mean(r5),
        "Recall@10": np.mean(r10),
        "Candidate_Recall@50": np.mean(r50),
        "Candidate_Recall@100": np.mean(r100),
        "Precision@5": np.mean(p5),
        "Total_Queries": len(ground_truths)
    }

def run_evaluation(num_samples: int = 500):
    """Run baseline evaluation on a subset or full train queries."""
    with open("train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    bm25 = BM25Retriever.load("data/bm25_index.pkl")
    exact = ExactMatcher("data/doc_metadata.json")
    memory = TrainQuestionMemory("train.json")

    qids = list(train_data.keys())
    if num_samples and num_samples < len(qids):
        # Deterministic sample
        np.random.seed(42)
        sample_qids = np.random.choice(qids, num_samples, replace=False)
    else:
        sample_qids = qids

    print(f"Evaluating {len(sample_qids)} train queries...")

    bm25_preds = {}
    hybrid_preds = {}
    ground_truths = {}

    for idx, qid in enumerate(sample_qids):
        qobj = train_data[qid]
        q_text = qobj.get("question", "")
        gold = [str(x) for x in qobj.get("answer", [])]
        ground_truths[qid] = gold

        # 1. BM25 scores
        b_scores = bm25.retrieve(q_text)
        sorted_b = sorted(b_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_preds[qid] = [doc for doc, _ in sorted_b]

        # 2. Exact Matcher scores
        e_scores = exact.match(q_text)

        # 3. Train Memory scores (exclude exact qid to simulate unseen query)
        m_scores = memory.retrieve(q_text, exclude_qid=qid)

        # 4. Hybrid Reciprocal Rank Fusion (RRF)
        # RRF formula: Score(d) = Σ w_r / (k + rank_r(d))
        all_candidate_docs = set(list(b_scores.keys())[:100]) | set(e_scores.keys()) | set(m_scores.keys())
        rrf_scores = {}

        # BM25 ranks
        b_ranks = {doc: rank for rank, (doc, _) in enumerate(sorted_b[:100], 1)}
        for doc in all_candidate_docs:
            score = 0.0
            if doc in b_ranks:
                score += 1.0 / (60 + b_ranks[doc])
            if doc in e_scores:
                score += (e_scores[doc] / 15.0) * (1.0 / 30)
            if doc in m_scores:
                score += (m_scores[doc] / 10.0) * (1.0 / 40)
            rrf_scores[doc] = score

        sorted_hybrid = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_preds[qid] = [doc for doc, _ in sorted_hybrid]

        if (idx + 1) % 100 == 0:
            print(f"Evaluated {idx + 1}/{len(sample_qids)} queries...")

    print("\n" + "="*50)
    print("--- BM25 Alone Results ---")
    bm25_metrics = compute_metrics(bm25_preds, ground_truths)
    for k, v in bm25_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n" + "="*50)
    print("--- Hybrid (BM25 + Exact Match + Memory RRF) Results ---")
    hybrid_metrics = compute_metrics(hybrid_preds, ground_truths)
    for k, v in hybrid_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation(num_samples=500)
