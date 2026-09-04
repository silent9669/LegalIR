"""Submission formatting, strict validation against competition rules, and zip packaging."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


def validate_submission(
    submission: Dict[str, List[str]],
    expected_qids: Set[str],
    max_predictions: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Validate submission dictionary against official competition constraints:
    - keysts match expected query IDs exactly
    - 1 to max_predictions predictions per query
    - unique document IDs per query
    """
    errors: List[str] = []
    actual_qids = set(submission.keys())

    missing = expected_qids - actual_qids
    if missing:
        errors.append(f"Missing {len(missing)} query IDs from submission.")

    extra = actual_qids - expected_qids
    if extra:
        errors.append(f"Submission contains {len(extra)} unexpected query IDs.")

    for qid, doc_ids in submission.items():
        if not isinstance(doc_ids, list):
            errors.append(f"Query {qid} predictions must be a list.")
            continue

        if len(doc_ids) < 1:
            errors.append(f"Query {qid} has 0 predictions (at least 1 required).")
        elif len(doc_ids) > max_predictions:
            errors.append(f"Query {qid} has {len(doc_ids)} predictions (exceeds max {max_predictions}).")

        if len(doc_ids) != len(set(doc_ids)):
            errors.append(f"Duplicate document IDs detected in query {qid}: {doc_ids}")

    return len(errors) == 0, errors


def package_submission(
    submission: Dict[str, List[str]],
    out_dir: Union[str, Path],
    filename_prefix: str = "submission",
) -> Tuple[Path, Path]:
    """Save submission.json and compress it into submission.zip."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    json_path = out_p / f"{filename_prefix}.json"
    zip_path = out_p / f"{filename_prefix}.zip"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname=f"{filename_prefix}.json")

    return json_path, zip_path
