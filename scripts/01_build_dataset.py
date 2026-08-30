import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.build_canonical import build_canonical_package
from src.dataset.chunker import ChunkConfig

def build_dataset(
    raw_zip: str = None,
    train_json: str = None,
    out_dir: str = "artifacts/task1/data"
):
    print("=" * 60)
    print("UIT-DSC 2026 Task 1: Building Canonical Dataset")
    print("=" * 60)

    if raw_zip is None:
        for candidate in ["artifacts/raw/selected-contexts.zip", "selected-contexts.zip"]:
            if os.path.exists(candidate):
                raw_zip = candidate
                break
        if raw_zip is None:
            raw_zip = "artifacts/raw/selected-contexts.zip"

    if train_json is None:
        for candidate in ["artifacts/raw/train.json", "train.json"]:
            if os.path.exists(candidate):
                train_json = candidate
                break
        if train_json is None:
            train_json = "artifacts/raw/train.json"

    print(f"Using raw contexts from: {raw_zip}")
    print(f"Using train labels from  : {train_json}")

    chunk_config = ChunkConfig(
        macro_min_tokens=400,
        macro_max_tokens=800,
        micro_min_tokens=100,
        micro_max_tokens=250,
        fallback_min_tokens=700,
        fallback_max_tokens=1200,
        overlap_tokens=150,
    )

    report = build_canonical_package(
        raw_contexts_dir=raw_zip,
        train_json_path=train_json,
        output_dir=out_dir,
        chunk_config=chunk_config,
    )

    return report

if __name__ == "__main__":
    build_dataset()
