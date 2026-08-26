import json
from pathlib import Path
from typing import Any
import zipfile


def validate_submission(
    predictions: dict[str, Any],
    expected_qids: set[str],
    corpus_doc_ids: set[str],
) -> None:
    pred_qids = set(str(k) for k in predictions.keys())
    expected_qids = set(str(k) for k in expected_qids)
    corpus_doc_ids = set(str(k) for k in corpus_doc_ids)

    if pred_qids != expected_qids:
        missing = expected_qids - pred_qids
        extra = pred_qids - expected_qids
        raise ValueError(
            f"Submission query keys mismatch: {len(missing)} missing, {len(extra)} extra"
        )

    for qid, qobj in predictions.items():
        if not isinstance(qobj, dict) or "answer" not in qobj:
            raise ValueError(f"Query {qid} prediction must be a dict with 'answer' key")

        answer = qobj["answer"]
        if not isinstance(answer, list):
            raise ValueError(f"Query {qid} answer must be a list of document IDs")

        if not (1 <= len(answer) <= 5):
            raise ValueError(f"Query {qid} answer length must be between 1 to 5, got {len(answer)}")

        if not all(isinstance(x, str) for x in answer):
            raise ValueError(f"Query {qid} answer IDs must all be strings")

        if len(answer) != len(set(answer)):
            raise ValueError(f"Query {qid} contains duplicate document IDs in answer: {answer}")

        unknown_docs = set(answer) - corpus_doc_ids
        if unknown_docs:
            raise ValueError(f"Query {qid} contains unknown document IDs not in corpus: {unknown_docs}")


def package_submission(
    predictions: dict[str, Any],
    json_path: str | Path,
    zip_path: str | Path,
) -> None:
    json_path = Path(json_path)
    zip_path = Path(zip_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    json_bytes = (json.dumps(predictions, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    json_path.write_bytes(json_bytes)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json_bytes)
