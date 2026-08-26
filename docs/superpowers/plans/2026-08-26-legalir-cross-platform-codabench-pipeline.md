# LegalIR Cross-Platform Codabench Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved leakage-free, Codabench-equivalent LegalIR pipeline with shared canonical artifacts, local-only models, cross-platform execution, and score-gated BM25/dense/memory/reranker/LightGBM ranking.

**Architecture:** Establish reproducible paths, configuration, shared artifact manifests, and canonical-v2 data before changing retrieval. Replace the leaky benchmark with fold-scoped component construction and official-scorer equivalence, then add retrieval and ranking stages one at a time, preserving an accepted baseline after every stage. All large model-derived artifacts stay under `artifacts/local`; only contracts, code, tests, checksums, and accepted metric summaries are committed.

**Tech Stack:** Python 3.14 on macOS arm64, Python 3.12+ compatible Windows code, Pandas, PyArrow, NumPy, scikit-learn, PyTorch MPS/CPU/CUDA, Transformers, Hugging Face Hub, LightGBM, PEFT, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-legalir-cross-platform-codabench-pipeline-design.md`

## Global Constraints

- Preserve all 8,532 official document IDs; no external legal corpus, Task 2 data, synthetic augmentation, or online inference API.
- Optimize official macro Recall@5; predictions contain one to five unique official document IDs.
- Use `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`.
- Use `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- Keep total neural parameters below 4B.
- Shared dataset contracts and accepted benchmark summaries live on `main`; model weights, adapters, embeddings, checkpoints, caches, and full runs remain local.
- Canonical Python commands use `python -m ...`, `pathlib.Path`, stable score/doc-ID sorting, and Windows-safe process entrypoints.
- Do not publish BTC data to a public remote without explicit redistribution approval.
- Do not delete a data/index/output copy until checksums prove the retained copy is identical or a validated replacement exists.
- Do not prune unreachable Git objects without separate explicit confirmation.
- Existing benchmark values are labeled `legacy_leaky_baseline` and never used as strict acceptance evidence.

---

## File Structure

### Runtime and artifact contracts

- Create `src/core/paths.py` — repository-root discovery and shared/local path dataclasses.
- Create `src/core/config.py` — load and validate the single pipeline YAML.
- Create `src/core/run_manifest.py` — immutable run provenance records.
- Create `configs/pipeline.yaml` — runtime-loaded shared configuration.
- Create `src/artifacts/checksums.py` — streaming SHA-256 helpers.
- Create `src/artifacts/manifest.py` — artifact inventory creation and verification.
- Create `src/artifacts/cli.py` — `verify`, `inventory`, and cleanup dry-run commands.

### Canonical dataset v2

- Create `src/dataset/source_reader.py` — stream `context_*.json` directly from ZIP.
- Create `src/dataset/schema.py` — required columns, dtypes, and canonical constants.
- Create `src/dataset/legal_parser.py` — hierarchy and metadata parsing.
- Create `src/dataset/chunker.py` — token-aware macro/micro chunk generation.
- Modify `src/dataset/build_canonical.py` — orchestrate v2 build and write auxiliary artifacts.
- Modify `src/dataset/validator.py` — enforce complete v2 invariants.
- Modify `src/evaluation/splits.py` — deterministic checksummed splits.

### Evaluation and submission

- Create `src/evaluation/submission.py` — strict JSON and ZIP validation.
- Create `src/evaluation/codabench_compat.py` — compare internal metrics with official scorer logic.
- Create `src/evaluation/benchmark.py` — fold-scoped benchmark runner and diagnostics.
- Modify `src/evaluation/evaluator.py` — strict metric inputs and candidate metrics at 20/50/100/150.
- Replace `src/validate_all.py` with a thin compatibility CLI.

### Retrieval and ranking

- Create `src/retrieval/types.py` — shared `TypedDict` candidate/evidence contracts.
- Modify `src/retrieval/bm25_micro.py` — fielded scores, second-best aggregation, stable sort, cache manifest.
- Modify `src/retrieval/exact_matcher.py` — complete legal metadata signals.
- Modify `src/retrieval/question_memory.py` — fold-safe lexical and dense memory.
- Modify `src/retrieval/dense_macro.py` — pinned model, local embedding build/load, exact search.
- Modify `src/retrieval/hybrid_search.py` — retain top 150, feature-complete candidate union.
- Modify `src/retrieval/build_indexes.py` — build local BM25 and dense artifacts.
- Modify `src/ranking/evidence_pack.py` — top evidence and required formatted sections.
- Modify `src/ranking/reranker.py` — document-level reranking and evidence features.
- Modify `src/ranking/oof_features.py` — versioned feature schema.
- Modify `src/ranking/fusion.py` — save/load RRF and LightGBM models, stable ranking.
- Modify `src/ranking/selector.py` — enforce legal output constraints.

### Training and end-to-end pipeline

- Create `src/models/bootstrap.py` — pinned minimal model downloads into local artifacts.
- Create `src/models/device.py` — MPS/CUDA/CPU resolution and manifest reporting.
- Modify `src/training/positive_localizer.py` — lexical+dense positive localization.
- Modify `src/training/hard_negative_miner.py` — duplicate/near-query false-negative guard.
- Create `src/training/build_pairs.py` — local retriever/reranker pair artifacts.
- Create `src/training/train_reranker.py` — optional local PEFT/LoRA training.
- Create `src/ranking/train_fusion.py` — fold-safe LightGBM training.
- Create `src/pipeline/predict.py` — one production prediction path shared by benchmark and submission.
- Create `src/pipeline/run_all.py` — verified raw artifacts to validated `submission.zip`.
- Replace `src/predict_submission.py` with a thin compatibility CLI.

### Tests and documentation

- Create focused tests named in each task below.
- Modify `README.md` — truthful active pipeline, cross-platform commands, artifact provenance.
- Modify `.gitignore` — shared/local artifact boundary.
- Modify `requirements.txt`; create `requirements-dev.txt` and `requirements-train.txt`.
- Modify `scripts/01_build_dataset.sh` through `scripts/04_predict_submission.sh` as optional wrappers around Python modules.

---

### Task 1: Runtime Paths, Configuration, and Dependencies

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/paths.py`
- Create: `src/core/config.py`
- Create: `configs/pipeline.yaml`
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `requirements-train.txt`
- Test: `tests/test_core_config.py`

**Interfaces:**
- Produces: `ProjectPaths.from_repo(repo_root: Path | None = None) -> ProjectPaths`
- Produces: `load_pipeline_config(path: Path) -> dict[str, Any]`
- Produces: `resolve_device_name(requested: str) -> str` later consumed by model tasks through config value `runtime.device`.

- [ ] **Step 1: Write failing path and config tests**

```python
from pathlib import Path
import pytest

from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths


def test_project_paths_separate_shared_and_local(tmp_path: Path):
    paths = ProjectPaths.from_repo(tmp_path)
    assert paths.shared == tmp_path / "artifacts" / "shared"
    assert paths.canonical == paths.shared / "canonical" / "v2"
    assert paths.local_models == tmp_path / "artifacts" / "local" / "models"
    assert paths.local_runs == tmp_path / "artifacts" / "local" / "runs"


def test_config_rejects_absolute_project_paths(tmp_path: Path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("paths:\n  canonical: /tmp/illegal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        load_pipeline_config(cfg)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_core_config.py -v`  
Expected: FAIL because `src.core.paths` and `src.core.config` do not exist.

- [ ] **Step 3: Implement path and config contracts**

```python
# src/core/paths.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo: Path
    shared: Path
    canonical: Path
    local: Path
    local_models: Path
    local_indexes: Path
    local_runs: Path

    @classmethod
    def from_repo(cls, repo_root: Path | None = None) -> "ProjectPaths":
        repo = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        shared = repo / "artifacts" / "shared"
        local = repo / "artifacts" / "local"
        return cls(
            repo=repo,
            shared=shared,
            canonical=shared / "canonical" / "v2",
            local=local,
            local_models=local / "models",
            local_indexes=local / "indexes",
            local_runs=local / "runs",
        )
```

```python
# src/core/config.py
from pathlib import Path
from typing import Any
import yaml


def load_pipeline_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, value in data.get("paths", {}).items():
        if Path(str(value)).is_absolute():
            raise ValueError(f"paths.{name} must be relative to repository root")
    return data
```

Create `configs/pipeline.yaml` with exact shared/local defaults, seed `42`, candidate cutoffs `[20, 50, 100, 150]`, model IDs/revisions, and `runtime.device: auto`.

Update direct dependencies to include `pandas`, `pyarrow`, `PyYAML`, `lightgbm`, `huggingface-hub`, and remove unused `rank-bm25` and `FlagEmbedding` unless a later task imports them. Put `pytest` in `requirements-dev.txt`; put `peft` and `accelerate` in `requirements-train.txt`.

- [ ] **Step 4: Run focused and existing tests**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_core_config.py tests/test_evaluation.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core configs/pipeline.yaml requirements*.txt tests/test_core_config.py
git commit -m "feat(core): add cross-platform pipeline configuration" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Shared Artifact Inventory and Checksum Verification

**Files:**
- Create: `src/artifacts/__init__.py`
- Create: `src/artifacts/checksums.py`
- Create: `src/artifacts/manifest.py`
- Create: `src/artifacts/cli.py`
- Test: `tests/test_artifact_manifest.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `build_inventory(root: Path) -> dict[str, ArtifactRecord]`
- Produces: `verify_inventory(root: Path, manifest_path: Path) -> list[str]`, returning error messages and never deleting files.

- [ ] **Step 1: Write failing checksum and mismatch tests**

```python
import json
from pathlib import Path

from src.artifacts.checksums import sha256_file
from src.artifacts.manifest import build_inventory, verify_inventory


def test_inventory_uses_relative_paths_and_detects_changes(tmp_path: Path):
    root = tmp_path / "shared"
    root.mkdir()
    payload = root / "documents.parquet"
    payload.write_bytes(b"canonical")
    inventory = build_inventory(root)
    assert inventory["documents.parquet"]["sha256"] == sha256_file(payload)

    manifest = tmp_path / "artifacts.sha256.json"
    manifest.write_text(json.dumps(inventory), encoding="utf-8")
    payload.write_bytes(b"changed")
    assert verify_inventory(root, manifest) == ["checksum mismatch: documents.parquet"]
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_artifact_manifest.py -v`  
Expected: FAIL because artifact modules do not exist.

- [ ] **Step 3: Implement streaming checksums and inventory**

```python
# src/artifacts/checksums.py
from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
```

```python
# src/artifacts/manifest.py
from pathlib import Path
import json
from src.artifacts.checksums import sha256_file


def build_inventory(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify_inventory(root: Path, manifest_path: Path) -> list[str]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for rel, record in expected.items():
        path = root / rel
        if not path.exists():
            errors.append(f"missing: {rel}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"checksum mismatch: {rel}")
    return errors
```

Implement `python -m src.artifacts.cli inventory|verify` with nonzero exit on verification errors. Update `.gitignore` so `artifacts/local/` is ignored; keep `artifacts/shared/manifests/` and accepted JSON reports trackable. Do not add LFS patterns or push dataset files in this task.

- [ ] **Step 4: Run tests and a repository inventory dry run**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_artifact_manifest.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.artifacts.cli inventory --root artifacts/data --output /tmp/legalir-inventory.json`  
Expected: creates inventory without changing repository files.

- [ ] **Step 5: Commit**

```bash
git add src/artifacts .gitignore tests/test_artifact_manifest.py
git commit -m "feat(artifacts): add verified shared artifact manifests" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: ZIP-Native Official Context Reader

**Files:**
- Create: `src/dataset/source_reader.py`
- Test: `tests/test_source_reader.py`
- Modify: `src/dataset/build_canonical.py`

**Interfaces:**
- Produces: `iter_official_contexts(zip_path: Path) -> Iterator[dict[str, Any]]`
- Guarantees sorted `context_*.json` member order, UTF-8 decoding, unique string IDs, and no extraction directory.

- [ ] **Step 1: Write failing ZIP-order and duplicate-ID tests**

```python
import json
import zipfile
from pathlib import Path
import pytest

from src.dataset.source_reader import iter_official_contexts


def test_reader_streams_sorted_contexts_without_extracting(tmp_path: Path):
    archive = tmp_path / "selected-contexts.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("context_2.json", json.dumps({"id": 2, "passage": "B"}))
        zf.writestr("context_1.json", json.dumps({"id": 1, "passage": "A"}))
    assert [row["id"] for row in iter_official_contexts(archive)] == ["1", "2"]
    assert not (tmp_path / "selected-contexts").exists()


def test_reader_rejects_duplicate_document_ids(tmp_path: Path):
    archive = tmp_path / "selected-contexts.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("context_1.json", json.dumps({"id": 1}))
        zf.writestr("nested/context_2.json", json.dumps({"id": 1}))
    with pytest.raises(ValueError, match="duplicate document ID 1"):
        list(iter_official_contexts(archive))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_source_reader.py -v`  
Expected: FAIL because `source_reader.py` does not exist.

- [ ] **Step 3: Implement ZIP streaming**

```python
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import json
import zipfile


def iter_official_contexts(zip_path: Path) -> Iterator[dict[str, Any]]:
    seen = set()
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if Path(name).name.startswith("context_") and name.endswith(".json")
        )
        if not names:
            raise ValueError(f"no context_*.json members in {zip_path}")
        for name in names:
            row = json.loads(archive.read(name).decode("utf-8"))
            row["id"] = str(row["id"])
            if row["id"] in seen:
                raise ValueError(f"duplicate document ID {row['id']}")
            seen.add(row["id"])
            yield row
```

Change `build_canonical_package()` to accept a raw ZIP path rather than a directory; retain the old CLI flag as a deprecated alias only until scripts migrate.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_source_reader.py tests/test_canonical_dataset.py -v`  
Expected: PASS after adapting fixture calls.

- [ ] **Step 5: Commit**

```bash
git add src/dataset/source_reader.py src/dataset/build_canonical.py tests/test_source_reader.py tests/test_canonical_dataset.py
git commit -m "feat(dataset): stream official contexts from ZIP" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Legal Hierarchy Parser and Token-Aware Chunker

**Files:**
- Create: `src/dataset/legal_parser.py`
- Create: `src/dataset/chunker.py`
- Create: `src/dataset/schema.py`
- Test: `tests/test_legal_chunking.py`
- Modify: `src/dataset/build_canonical.py`

**Interfaces:**
- Produces: `parse_legal_units(text: str) -> list[LegalUnit]`
- Produces: `build_document_chunks(document: dict, config: ChunkConfig) -> list[dict]`
- `LegalUnit` fields: `chapter`, `section`, `article`, `clause`, `point`, `text`.
- `ChunkConfig` fields: macro min/max `400/800`, micro min/max `100/250`, fallback min/max `700/1200`, overlap `150` tokens.

- [ ] **Step 1: Write hierarchy, oversized-article, overlap, and empty tests**

```python
from src.dataset.chunker import ChunkConfig, build_document_chunks
from src.dataset.legal_parser import parse_legal_units


def test_parser_preserves_full_hierarchy():
    text = "Chương I\nMục 1\nĐiều 2. Phạm vi\n1. Nội dung\na) Chi tiết"
    unit = parse_legal_units(text)[0]
    assert (unit.chapter, unit.section, unit.article, unit.clause, unit.point) == (
        "Chương I", "Mục 1", "Điều 2. Phạm vi", "Khoản 1", "Điểm a"
    )


def test_long_article_is_split_and_micro_parents_are_valid():
    doc = {"doc_id": "1", "title": "Luật thử", "passage_norm": "Điều 1. " + "từ " * 2400, "is_empty": False}
    chunks = build_document_chunks(doc, ChunkConfig())
    macros = [c for c in chunks if c["granularity"] == "macro"]
    micros = [c for c in chunks if c["granularity"] == "micro"]
    assert len(macros) > 1
    assert max(c["token_count"] for c in macros) <= 800
    assert {c["parent_chunk_id"] for c in micros} <= {c["chunk_id"] for c in macros}


def test_empty_document_has_one_metadata_chunk():
    doc = {"doc_id": "20", "title": "Văn bản trống", "passage_norm": "", "is_empty": True}
    chunks = build_document_chunks(doc, ChunkConfig())
    assert len(chunks) == 1
    assert chunks[0]["is_empty"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_legal_chunking.py -v`  
Expected: FAIL because parser/chunker contracts do not exist.

- [ ] **Step 3: Implement focused dataclasses and token windows**

```python
# src/dataset/chunker.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    macro_min_tokens: int = 400
    macro_max_tokens: int = 800
    micro_min_tokens: int = 100
    micro_max_tokens: int = 250
    fallback_min_tokens: int = 700
    fallback_max_tokens: int = 1200
    overlap_tokens: int = 150


def sliding_token_windows(tokens: list[str], max_tokens: int, overlap: int) -> list[list[str]]:
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    step = max_tokens - overlap
    return [tokens[start:start + max_tokens] for start in range(0, len(tokens), step)]
```

Implement hierarchy state updates in `legal_parser.py`; keep legal units intact when within size limits. Split oversized units using token windows, derive micro chunks from each macro, and include `chapter`, `section`, `article`, `clause`, `point`, `parent_chunk_id`, `token_count`, and `is_empty` in the schema. Use deterministic IDs derived from document ID and ordinal.

- [ ] **Step 4: Run focused tests and property checks**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_legal_chunking.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_canonical_dataset.py -v`  
Expected: PASS after updating expected v2 columns.

- [ ] **Step 5: Commit**

```bash
git add src/dataset/schema.py src/dataset/legal_parser.py src/dataset/chunker.py src/dataset/build_canonical.py tests/test_legal_chunking.py tests/test_canonical_dataset.py
git commit -m "feat(dataset): add legal-aware token chunking" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Canonical-v2 Builder, Duplicate Groups, and Complete Validator

**Files:**
- Modify: `src/dataset/build_canonical.py`
- Modify: `src/dataset/validator.py`
- Modify: `src/evaluation/splits.py`
- Test: `tests/test_canonical_validator_v2.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces: `build_canonical_package(raw_zip: Path, train_json: Path, output_dir: Path, config: dict) -> dict`
- Produces: `validate_canonical_dataset(canonical_dir: Path, expected_document_count: int = 8532) -> dict`
- Produces deterministic manifests containing source, schema, and split checksums.

- [ ] **Step 1: Write failing invariant tests**

```python
from pathlib import Path
import pandas as pd

from src.dataset.validator import validate_canonical_dataset


def test_validator_reports_duplicate_ids_and_cross_doc_parent(tmp_path: Path):
    pd.DataFrame([
        {"doc_id": "1", "is_empty": False},
        {"doc_id": "1", "is_empty": False},
    ]).to_parquet(tmp_path / "documents.parquet")
    pd.DataFrame([
        {"chunk_id": "m1", "doc_id": "1", "granularity": "macro", "parent_chunk_id": None, "is_empty": False},
        {"chunk_id": "u1", "doc_id": "2", "granularity": "micro", "parent_chunk_id": "m1", "is_empty": False},
    ]).to_parquet(tmp_path / "chunks.parquet")
    pd.DataFrame([{"query_id": "q1", "gold_count": 1}]).to_parquet(tmp_path / "queries_train.parquet")
    pd.DataFrame([{"query_id": "missing", "doc_id": "1", "relevance": 1}]).to_parquet(tmp_path / "qrels_train.parquet")
    report = validate_canonical_dataset(tmp_path, expected_document_count=2)
    assert "duplicate document IDs" in " ".join(report["errors"])
    assert "unknown query IDs" in " ".join(report["errors"])
    assert "cross-document parent" in " ".join(report["errors"])
```

Add deterministic split checksum tests: generating twice with seed 42 must produce byte-identical JSON.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_canonical_validator_v2.py tests/test_splits.py -v`  
Expected: FAIL because current validator checks only four relationships.

- [ ] **Step 3: Implement schema and relationship validation**

Implement named checks for required columns, dtypes, unique IDs, granularity, parent document equality, qrel query/doc coverage, duplicate qrels, relevance, `gold_count`, empty records, duplicate mappings, and manifest hashes. Return every error in one report; do not stop at the first error.

Create `duplicate_groups.json` keyed by normalized passage SHA-256 and `empty_context_ids.json`. Indexing later uses the representative content while preserving every document ID mapping.

Generate JSON with `sort_keys=True`, deterministic list ordering, and a trailing newline so checksums remain stable across Mac and Windows.

- [ ] **Step 4: Run fixture tests, then build canonical v2 into a temporary directory**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_canonical_validator_v2.py tests/test_splits.py tests/test_canonical_dataset.py -v`  
Expected: PASS.

Run:

```bash
PYTHONPATH=. .venv/bin/python -m src.dataset.build_canonical \
  --config configs/pipeline.yaml \
  --raw-zip artifacts/shared/raw/selected-contexts.zip \
  --train-json artifacts/shared/raw/train.json \
  --output-dir artifacts/local/canonical-v2-build
```

Expected: 8,532 documents, 7,000 queries, 7,637 qrels, validator `is_valid=true`, no chunk above documented limits except audited short-unit exceptions.

- [ ] **Step 5: Commit code and tests, not generated Parquet files**

```bash
git add src/dataset src/evaluation/splits.py tests/test_canonical_validator_v2.py tests/test_splits.py tests/test_canonical_dataset.py
git commit -m "feat(dataset): build verified canonical v2 package" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Official Codabench Equivalence and Strict Submission Validation

**Files:**
- Create: `src/evaluation/submission.py`
- Create: `src/evaluation/codabench_compat.py`
- Modify: `src/evaluation/evaluator.py`
- Test: `tests/test_codabench_compat.py`
- Test: `tests/test_submission_compliance.py`

**Interfaces:**
- Produces: `validate_submission(predictions, expected_qids, corpus_doc_ids) -> None`
- Produces: `package_submission(predictions, json_path: Path, zip_path: Path) -> None`
- Produces: `assert_official_equivalence(predictions, ground_truths) -> dict[str, float]`

- [ ] **Step 1: Write failing official-equivalence and ZIP-byte tests**

```python
import json
import zipfile
from pathlib import Path
import pytest

from src.evaluation.codabench_compat import assert_official_equivalence
from src.evaluation.submission import package_submission, validate_submission


def test_internal_metrics_equal_official_scorer():
    truth = {"q1": ["1", "2"], "q2": ["3"]}
    pred = {"q1": {"answer": ["2", "9"]}, "q2": {"answer": ["3"]}}
    metrics = assert_official_equivalence(pred, truth)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.75)


def test_packaged_zip_contains_exact_json_bytes(tmp_path: Path):
    pred = {"q1": {"answer": ["1"]}}
    json_path, zip_path = tmp_path / "submission.json", tmp_path / "submission.zip"
    package_submission(pred, json_path, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["submission.json"]
        assert archive.read("submission.json") == json_path.read_bytes()


def test_validator_rejects_extra_query():
    with pytest.raises(ValueError, match="query keys"):
        validate_submission({"q1": {"answer": ["1"]}, "extra": {"answer": ["1"]}}, {"q1"}, {"1"})
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_codabench_compat.py tests/test_submission_compliance.py -v`  
Expected: FAIL because strict modules do not exist.

- [ ] **Step 3: Implement exact validation and scorer comparison**

`validate_submission` checks exact query-key equality, dict/answer/list types, 1–5 IDs, strings, uniqueness, and corpus membership. `package_submission` writes deterministic UTF-8 JSON once, then inserts those exact bytes as the root ZIP member.

`assert_official_equivalence` imports `eval_retrieval` from `Scoring-Program-Task-LegalIR/scoring.py`, compares official and internal recall/precision within `1e-12`, and raises if they differ. Preserve the official scorer file unchanged.

- [ ] **Step 4: Run compatibility and fuzz cases**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_codabench_compat.py tests/test_submission_compliance.py tests/test_evaluation.py -v`  
Expected: PASS for valid cases and explicit rejection of malformed cases.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation tests/test_codabench_compat.py tests/test_submission_compliance.py tests/test_evaluation.py
git commit -m "feat(evaluation): match official Codabench scoring" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Fold-Scoped Memory and Leakage-Free Benchmark Runner

**Files:**
- Create: `src/evaluation/benchmark.py`
- Modify: `src/retrieval/question_memory.py`
- Modify: `src/validate_all.py`
- Test: `tests/test_fold_isolation.py`
- Test: `tests/test_benchmark_metrics.py`

**Interfaces:**
- Produces: `build_memory_rows(train_query_ids, queries, qrels) -> list[dict]`
- Produces: `run_benchmark(config_path: Path, fold_limit: int | None = None) -> dict`
- `QuestionMemory.training_query_ids: frozenset[str]` is exposed for leakage assertions.

- [ ] **Step 1: Write failing leakage test**

```python
from src.evaluation.benchmark import build_memory_rows
from src.retrieval.question_memory import QuestionMemory


def test_validation_queries_never_enter_question_memory():
    queries = {"train": "câu train", "val": "câu val"}
    qrels = {"train": ["1"], "val": ["2"]}
    rows = build_memory_rows(["train"], queries, qrels)
    memory = QuestionMemory(rows, min_similarity=0.82)
    assert memory.training_query_ids == frozenset({"train"})
    assert "val" not in memory.qid_to_docs
```

Add a benchmark test that expects candidate metrics at 20, 50, 100, and 150 and per-fold mean/std fields.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_fold_isolation.py tests/test_benchmark_metrics.py -v`  
Expected: FAIL because memory construction is global and benchmark module does not exist.

- [ ] **Step 3: Implement fold component factory**

```python
def build_memory_rows(train_query_ids, queries, qrels):
    return [
        {"query_id": qid, "question_norm": queries[qid], "doc_ids": qrels[qid]}
        for qid in sorted(map(str, train_query_ids))
        if qid in queries and qid in qrels
    ]
```

Create a new `QuestionMemory` per fold and a separate one for document-disjoint training IDs. Remove the belief that `exclude_qid` is sufficient isolation. Persist fold predictions and candidate lists under the run directory. Make `src/validate_all.py` a thin CLI delegating to `run_benchmark`.

- [ ] **Step 4: Run unit tests and one strict fold smoke benchmark**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_fold_isolation.py tests/test_benchmark_metrics.py tests/test_retrieval_branches.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.evaluation.benchmark --config configs/pipeline.yaml --fold-limit 1`  
Expected: one new run directory labeled `strict_baseline`, no validation QID in the saved memory manifest, metrics at 20/50/100/150.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark.py src/retrieval/question_memory.py src/validate_all.py tests/test_fold_isolation.py tests/test_benchmark_metrics.py
git commit -m "fix(validation): isolate memory by evaluation fold" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Establish the Accepted Strict Baseline

**Files:**
- Modify: `scripts/03_run_benchmark.sh`
- Modify: `README.md`
- Create: `artifacts/shared/benchmarks/accepted/strict_baseline.json`
- Test: `tests/test_accepted_benchmark_schema.py`

**Interfaces:**
- Produces accepted metric summary with `dataset_sha256`, `split_sha256`, five folds, mean/std, document-disjoint metrics, candidate cutoffs, commit, and `leakage_checks_passed=true`.

- [ ] **Step 1: Write benchmark-summary schema test**

```python
import json
from pathlib import Path


def test_accepted_baseline_contains_all_five_folds():
    report = json.loads(Path("artifacts/shared/benchmarks/accepted/strict_baseline.json").read_text())
    assert len(report["random_5fold"]["folds"]) == 5
    assert report["leakage_checks_passed"] is True
    assert set(report["candidate_cutoffs"]) == {20, 50, 100, 150}
    assert report["label"] == "strict_baseline"
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_accepted_benchmark_schema.py -v`  
Expected: FAIL because no strict accepted baseline exists.

- [ ] **Step 3: Run the full baseline benchmark**

Run: `PYTHONPATH=. .venv/bin/python -m src.evaluation.benchmark --config configs/pipeline.yaml`  
Expected: all five folds plus document-disjoint complete; official scorer equivalence passes; run manifest records checksums and commit.

Copy only the small accepted metrics summary, not full predictions or candidates, into `artifacts/shared/benchmarks/accepted/strict_baseline.json`. Label prior reports `legacy_leaky_baseline` in README and remove their leaderboard-style claims.

- [ ] **Step 4: Run schema and full fast suite**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_accepted_benchmark_schema.py tests -m "not slow" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/03_run_benchmark.sh README.md artifacts/shared/benchmarks/accepted/strict_baseline.json tests/test_accepted_benchmark_schema.py
git commit -m "test(validation): record leakage-free five-fold baseline" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Fielded BM25 and Complete Exact Legal Signals

**Files:**
- Create: `src/retrieval/types.py`
- Modify: `src/retrieval/bm25_micro.py`
- Modify: `src/retrieval/exact_matcher.py`
- Modify: `src/retrieval/build_indexes.py`
- Test: `tests/test_fielded_retrieval.py`

**Interfaces:**
- Produces `CandidateRecord` fields for every branch.
- `BM25MicroRetriever.retrieve(query: str, top_k: int) -> list[dict]` returns document score plus best/second/mean evidence scores and chunk IDs.
- `ExactMatcher.match(query: str) -> dict[str, dict[str, float | bool]]` returns separate legal-number/year/type/article/clause/point/entity features.

- [ ] **Step 1: Write failing field-weight and stable-sort tests**

```python
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher


def test_legal_number_field_outranks_body_only_match():
    chunks = [
        {"chunk_id": "c1", "doc_id": "1", "legal_number": "61/2020/QH14", "title": "Luật", "article": "", "text_norm": "không liên quan", "link": ""},
        {"chunk_id": "c2", "doc_id": "2", "legal_number": "", "title": "", "article": "", "text_norm": "61 2020 qh14 nhắc trong nội dung", "link": ""},
    ]
    retriever = BM25MicroRetriever(field_weights={"legal_number": 5.0, "body": 1.0}).fit(chunks)
    assert retriever.retrieve("61/2020/QH14", top_k=2)[0]["doc_id"] == "1"


def test_exact_match_returns_separate_flags():
    matcher = ExactMatcher([{"doc_id": "1", "legal_number": "61/2020/QH14", "title": "Luật Đầu tư", "year": "2020", "doc_type": "Luật"}])
    result = matcher.match("Điều 2 Luật Đầu tư số 61/2020/QH14")
    assert result["1"]["exact_legal_number"] is True
    assert result["1"]["exact_year"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_fielded_retrieval.py -v`  
Expected: FAIL because retrieval uses a single text field and scalar exact scores.

- [ ] **Step 3: Implement fielded scoring and typed candidate records**

Use one BM25 posting structure per configured field. Combine field scores using config weights. Aggregate each document as `best + 0.1 * second_best`; retain mean as a feature. Stable-sort by `(-score, doc_id)`.

Extend exact regex/features without iterating every title for every query: create normalized title n-gram or token-prefix indexes during matcher initialization. Return all high-confidence hits; the hybrid union decides cutoffs.

Bind the local BM25 cache to canonical checksum, config checksum, and code commit in `index_manifest.json`.

- [ ] **Step 4: Run retrieval tests and benchmark ablation**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_fielded_retrieval.py tests/test_retrieval_branches.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.evaluation.benchmark --config configs/pipeline.yaml --label fielded-bm25-exact`  
Expected: accepted only if strict mean Recall@5 improves or candidate recall improves without violating gates.

- [ ] **Step 5: Commit code and accepted summary if gates pass**

```bash
git add src/retrieval tests/test_fielded_retrieval.py tests/test_retrieval_branches.py configs/pipeline.yaml
git commit -m "feat(retrieval): add fielded legal BM25 signals" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Pinned Local Model Bootstrap and Device Resolution

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/bootstrap.py`
- Create: `src/models/device.py`
- Test: `tests/test_model_bootstrap.py`

**Interfaces:**
- Produces: `resolve_device(requested: str = "auto") -> str`
- Produces: `download_models(config: dict, model_root: Path) -> dict[str, Path]`
- Downloads only required files at immutable revisions into `artifacts/local/models/huggingface`.

- [ ] **Step 1: Write failing device and snapshot-argument tests**

```python
from pathlib import Path
from src.models.bootstrap import required_model_files
from src.models.device import resolve_device


def test_required_files_exclude_onnx_and_openvino():
    files = required_model_files("BAAI/bge-m3")
    assert "pytorch_model.bin" in files
    assert not any(name.startswith("onnx/") for name in files)


def test_explicit_cpu_is_preserved():
    assert resolve_device("cpu") == "cpu"
```

Mock `huggingface_hub.snapshot_download` and assert exact repository ID, revision, `allow_patterns`, and local cache path.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_model_bootstrap.py -v`  
Expected: FAIL because model modules do not exist.

- [ ] **Step 3: Implement pinned minimal downloads**

Use `snapshot_download(repo_id=..., revision=..., allow_patterns=..., cache_dir=model_root)` and set `local_files_only` from CLI/config for offline mode. Required BGE-M3 patterns include the PyTorch model, tokenizer, SentenceTransformers module/config files, and pooling config; reranker patterns include `model.safetensors`, tokenizer, and model config.

Device selection order for `auto` is CUDA, MPS, CPU. Record selected device but do not hide an explicitly unavailable request.

- [ ] **Step 4: Run tests, then download models locally**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_model_bootstrap.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.models.bootstrap --config configs/pipeline.yaml`  
Expected: approximately 4.6 GB under `artifacts/local/models/huggingface`; no root `models/` or second project cache created.

- [ ] **Step 5: Commit code only**

```bash
git add src/models tests/test_model_bootstrap.py configs/pipeline.yaml .gitignore
git commit -m "feat(models): bootstrap pinned local BGE models" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Dense Macro Embedding Build and Exact Search

**Files:**
- Modify: `src/retrieval/dense_macro.py`
- Modify: `src/retrieval/build_indexes.py`
- Test: `tests/test_dense_macro.py`

**Interfaces:**
- Produces: `DenseMacroRetriever.build(chunks, output_dir, batch_size, max_length) -> Path`
- Produces: `DenseMacroRetriever.load(index_dir, model_path, device) -> DenseMacroRetriever`
- Produces: `retrieve(query: str, top_k: int) -> list[dict]` with best/second scores and evidence chunk IDs.

- [ ] **Step 1: Write failing exact-search and manifest tests**

```python
import numpy as np
from pathlib import Path
from src.retrieval.dense_macro import DenseMacroRetriever


def test_exact_dense_search_aggregates_chunks_to_documents(tmp_path: Path):
    np.save(tmp_path / "embeddings.npy", np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float16))
    retriever = DenseMacroRetriever.from_arrays(
        embeddings_path=tmp_path / "embeddings.npy",
        chunk_ids=["a1", "a2", "b1"],
        doc_ids=["A", "A", "B"],
        query_encoder=lambda _: np.array([1.0, 0.0], dtype=np.float32),
    )
    hit = retriever.retrieve("query", top_k=2)[0]
    assert hit["doc_id"] == "A"
    assert hit["dense_best_chunk_id"] == "a1"
    assert hit["dense_second_score"] > 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_dense_macro.py -v`  
Expected: FAIL because build/load/from-arrays contracts do not exist.

- [ ] **Step 3: Implement local float16 arrays and exact batched scoring**

Store normalized embeddings in `embeddings.npy`, chunk/doc IDs in Parquet, and checksum/config/model metadata in `index_manifest.json`. Load arrays with `mmap_mode="r"` on CPU; support a bounded MPS tensor path when memory permits. Exact scoring may process corpus blocks and maintain top chunks with `argpartition`, followed by stable document aggregation.

Pin tokenizer/model loading to local snapshot paths and immutable revisions. Never call the network during benchmark or inference.

- [ ] **Step 4: Run tests and build the real dense index**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_dense_macro.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.retrieval.build_indexes --config configs/pipeline.yaml --dense`  
Expected: 201k-scale macro embeddings under `artifacts/local/indexes/dense`; manifest matches canonical-v2 checksum; M3 peak memory recorded.

- [ ] **Step 5: Commit code and tests**

```bash
git add src/retrieval/dense_macro.py src/retrieval/build_indexes.py tests/test_dense_macro.py
git commit -m "feat(retrieval): add exact BGE macro retrieval" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Dense Question Memory and Feature-Complete Candidate Union

**Files:**
- Modify: `src/retrieval/question_memory.py`
- Modify: `src/retrieval/hybrid_search.py`
- Modify: `src/retrieval/types.py`
- Test: `tests/test_dense_question_memory.py`
- Test: `tests/test_candidate_union.py`

**Interfaces:**
- `QuestionMemory(..., dense_encoder=None, min_similarity=0.82)` returns lexical/dense similarities, vote count, matched QID, and positive frequency.
- `HybridSearchEngine.search_candidates(query, top_k=150) -> list[CandidateRecord]` retains all branch features and stable order.

- [ ] **Step 1: Write failing dense-memory and union tests**

```python
import numpy as np
from src.retrieval.question_memory import QuestionMemory


def test_dense_memory_votes_without_validation_rows():
    rows = [{"query_id": "train", "question_norm": "đầu tư", "doc_ids": ["1"]}]
    encoder = lambda texts: np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)
    memory = QuestionMemory(rows, dense_encoder=encoder, min_similarity=0.82)
    result = memory.retrieve("đầu tư", top_k=5)
    assert result["1"]["dense_similarity"] == 1.0
    assert result["1"]["vote_count"] == 1
    assert memory.training_query_ids == frozenset({"train"})
```

Candidate union tests must prove a dense-only and exact-only document survive into the top-150 records and ties sort by document ID.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_dense_question_memory.py tests/test_candidate_union.py -v`  
Expected: FAIL because memory is lexical-only and union truncates at 50.

- [ ] **Step 3: Implement dual memory and top-150 union**

Normalize dense query embeddings, combine exact lookup, lexical similarity, and dense similarity without leaking labels. Record separate features; do not collapse them to one opaque score. Hybrid union creates one record per document, calculates source count, preserves all branch ranks/scores, and returns top 150 by cheap RRF.

- [ ] **Step 4: Run tests and retrieval ablation**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_dense_question_memory.py tests/test_candidate_union.py tests/test_fold_isolation.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.evaluation.benchmark --config configs/pipeline.yaml --label dense-memory-union`  
Expected: Candidate Recall@100/150 improves over the strict baseline; fold leakage assertions remain true.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/question_memory.py src/retrieval/hybrid_search.py src/retrieval/types.py tests/test_dense_question_memory.py tests/test_candidate_union.py
git commit -m "feat(retrieval): add fold-safe dense memory union" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: Evidence Packs and Zero-Shot Document Reranking

**Files:**
- Modify: `src/ranking/evidence_pack.py`
- Modify: `src/ranking/reranker.py`
- Modify: `src/pipeline/predict.py`
- Test: `tests/test_reranker_integration.py`

**Interfaces:**
- Produces: `EvidencePackBuilder.build(query, candidate, max_chunks=2) -> list[Evidence]`
- Produces: `CrossEncoderReranker.rerank(query, candidates, evidence_builder, top_k=50) -> list[CandidateRecord]`
- Each reranked record includes best, second, margin, best chunk ID, and evidence count.

- [ ] **Step 1: Write failing formatted-evidence and aggregation tests**

```python
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.reranker import CrossEncoderReranker


def test_evidence_contains_required_sections():
    builder = EvidencePackBuilder([{"doc_id": "1", "chunk_id": "c1", "article": "Điều 2", "text_norm": "Nội dung"}], {"1": {"title": "Luật A", "legal_number": "1/2020"}})
    evidence = builder.build("query", {"doc_id": "1"})[0]
    assert "[VĂN BẢN]" in evidence["text"]
    assert "[ĐIỀU KHOẢN]" in evidence["text"]
    assert "[NỘI DUNG]" in evidence["text"]


def test_reranker_aggregates_best_and_second_scores():
    reranker = CrossEncoderReranker(score_fn=lambda pairs: [3.0, 1.0])
    record = reranker.aggregate_document("1", [{"chunk_id": "a"}, {"chunk_id": "b"}], [3.0, 1.0])
    assert record["reranker_best_score"] == 3.0
    assert record["reranker_second_score"] == 1.0
    assert record["reranker_margin"] == 2.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_reranker_integration.py -v`  
Expected: FAIL because document-level APIs do not exist.

- [ ] **Step 3: Implement zero-shot reranking**

Select top evidence using lexical and dense chunk features already present in candidates. Format explicit sections. Score at most the configured top 50 documents in deterministic batches using the pinned local reranker. Add OOM handling only for real MPS batch failures: halve batch size and retry the current batch; never silently switch model or truncate candidate count.

- [ ] **Step 4: Run tests, model smoke, and strict ablation**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_reranker_integration.py tests/test_reranker.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.evaluation.benchmark --config configs/pipeline.yaml --label zero-shot-reranker --reranker`  
Expected: full run records batch size, dtype, runtime, and Recall@5 comparison. Keep reranker enabled only if selection gates pass.

- [ ] **Step 5: Commit**

```bash
git add src/ranking/evidence_pack.py src/ranking/reranker.py src/pipeline/predict.py tests/test_reranker_integration.py tests/test_reranker.py
git commit -m "feat(ranking): add BGE evidence reranking" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: Guarded Positive Localization and Hard-Negative Pair Builder

**Files:**
- Modify: `src/training/positive_localizer.py`
- Modify: `src/training/hard_negative_miner.py`
- Create: `src/training/build_pairs.py`
- Test: `tests/test_training_pairs.py`

**Interfaces:**
- `PositiveLocalizer.localize(query, gold_doc_id, top_k=2) -> list[dict]`
- `HardNegativeMiner.mine_negatives(query_id, candidates, gold_doc_ids, max_negatives=15) -> list[str]`
- `build_training_pairs(...) -> tuple[pd.DataFrame, pd.DataFrame]` for local retriever/reranker Parquet files.

- [ ] **Step 1: Write failing false-negative-guard tests**

```python
from src.training.hard_negative_miner import HardNegativeMiner


def test_miner_excludes_duplicate_and_near_query_positives():
    miner = HardNegativeMiner(false_negative_blacklist={"q1": {"2", "3"}})
    candidates = [{"doc_id": "1"}, {"doc_id": "2"}, {"doc_id": "3"}, {"doc_id": "4"}]
    assert miner.mine_negatives("q1", candidates, ["1"], max_negatives=10) == ["4"]
```

Add localization tests where lexical and dense evidence disagree and configured combined score selects the expected chunk.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_training_pairs.py tests/test_training_prep.py -v`  
Expected: FAIL because localizer returns one lexical chunk and miner lacks corpus-level guards.

- [ ] **Step 3: Implement pair generation**

Build near-query groups from exact normalized questions plus lexical/dense thresholds using only each fold's training partition. Build duplicate-passage guards from canonical-v2 groups. Emit explicit columns including query ID, query text, positive/negative document and chunk IDs, evidence text, source rank, and fold. Write only to `artifacts/local/training/pairs`.

- [ ] **Step 4: Run tests and generate a fold-0 sample**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_training_pairs.py tests/test_training_prep.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.training.build_pairs --config configs/pipeline.yaml --fold 0 --limit 100`  
Expected: local Parquet pair files, zero gold/blacklist negatives, manifest tied to fold checksum.

- [ ] **Step 5: Commit code and tests**

```bash
git add src/training tests/test_training_pairs.py tests/test_training_prep.py
git commit -m "feat(training): build guarded legal ranking pairs" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 15: OOF Feature Schema and LightGBM LambdaMART Fusion

**Files:**
- Modify: `src/ranking/oof_features.py`
- Modify: `src/ranking/fusion.py`
- Create: `src/ranking/train_fusion.py`
- Test: `tests/test_oof_fusion.py`

**Interfaces:**
- Produces: `FEATURE_SCHEMA_VERSION = "v2"`
- Produces: `extract_candidate_features(query_id, candidates) -> pd.DataFrame`
- Produces: `train_fold_ranker(train_features, model_dir) -> LightGBMRanker`
- Produces fold-local model manifests; no committed model files.

- [ ] **Step 1: Write failing feature and fold-exclusion tests**

```python
from src.ranking.oof_features import FEATURE_COLUMNS, extract_candidate_features


def test_oof_features_include_all_required_signals():
    required = {
        "memory_vote_count", "memory_dense_similarity", "exact_legal_number",
        "reranker_best_score", "reranker_second_score", "reranker_margin",
        "evidence_chunk_count", "branch_overlap_count",
    }
    assert required <= set(FEATURE_COLUMNS)


def test_feature_rows_keep_query_group_identity():
    df = extract_candidate_features("q1", [{"doc_id": "1", "bm25_score": 2.0}])
    assert df.loc[0, "query_id"] == "q1"
    assert df.loc[0, "doc_id"] == "1"
```

Add a training test whose validation QID causes an assertion if present in training features.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_oof_fusion.py tests/test_fusion.py -v`  
Expected: FAIL because existing feature schema has only twelve fields and no fold trainer.

- [ ] **Step 3: Implement versioned OOF training**

Create explicit feature columns and deterministic missing-value defaults. For each evaluation fold, train LightGBM only on records from other folds; assert query-ID disjointness. Save local `model.txt` and manifest with schema, dataset, split, and code hashes. Keep RRF as fallback when model validation fails or a feature schema mismatches.

- [ ] **Step 4: Run tests and strict fusion ablation**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_oof_fusion.py tests/test_fusion.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.ranking.train_fusion --config configs/pipeline.yaml`  
Expected: five local fold models and one full-data model; every manifest has disjoint training/evaluation QID checks.

Run: `PYTHONPATH=. .venv/bin/python -m src.evaluation.benchmark --config configs/pipeline.yaml --label lightgbm-fusion --fusion lightgbm`  
Expected: merge/enable only if performance gates pass.

- [ ] **Step 5: Commit code, tests, and accepted small summary only**

```bash
git add src/ranking tests/test_oof_fusion.py tests/test_fusion.py configs/pipeline.yaml
git commit -m "feat(ranking): add fold-safe LambdaMART fusion" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 16: Optional Local LoRA Reranker Experiment

**Files:**
- Create: `src/training/train_reranker.py`
- Create: `configs/experiments/reranker_lora.yaml`
- Test: `tests/test_train_reranker_config.py`

**Interfaces:**
- Produces: `train_reranker(config_path: Path, fold: int, output_dir: Path) -> dict`
- Writes adapter/checkpoint only under `artifacts/local/training/checkpoints`.
- Must run on an experiment branch, not directly on `main`.

- [ ] **Step 1: Create experiment branch and write failing config test**

Run: `git switch -c exp/phuc-bge-reranker-lora`

```python
from pathlib import Path
from src.training.train_reranker import load_training_config


def test_training_config_keeps_outputs_local():
    config = load_training_config(Path("configs/experiments/reranker_lora.yaml"))
    assert config["output_dir"].startswith("artifacts/local/")
    assert config["base_model_revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert config["quantization"] is None
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_train_reranker_config.py -v`  
Expected: FAIL because training entrypoint/config do not exist.

- [ ] **Step 3: Implement MPS-safe PEFT training**

Use PEFT LoRA without bitsandbytes quantization, gradient checkpointing, batch size 1, configurable gradient accumulation, fixed seed, fold-local pairs, and validation Recall@5 early stopping. Guard the CLI against output paths outside `artifacts/local`. Record peak memory and runtime. Do not commit generated adapters.

- [ ] **Step 4: Run dry-run, fold experiment, and ablation**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_train_reranker_config.py -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.training.train_reranker --config configs/experiments/reranker_lora.yaml --fold 0 --max-steps 2`  
Expected: two-step MPS smoke training succeeds and writes only local artifacts.

Run a full fold only after smoke success, then compare zero-shot and LoRA predictions with the strict benchmark. Continue all folds only if fold-0 improves and error analysis shows no leakage or duplicate-negative issue.

- [ ] **Step 5: Commit training code/config, not weights**

```bash
git add src/training/train_reranker.py configs/experiments/reranker_lora.yaml tests/test_train_reranker_config.py requirements-train.txt
git commit -m "feat(training): add local LoRA reranker experiment" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

Merge the code commit back to `main` only after experiment review; keep all adapter files local.

---

### Task 17: Unified Prediction Pipeline and Cross-Platform Orchestrator

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/predict.py`
- Create: `src/pipeline/run_all.py`
- Modify: `src/predict_submission.py`
- Modify: `scripts/01_build_dataset.sh`
- Modify: `scripts/02_build_indexes.sh`
- Modify: `scripts/03_run_benchmark.sh`
- Modify: `scripts/04_predict_submission.sh`
- Test: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Produces: `build_pipeline(config, fold_train_ids=None) -> LegalIRPipeline`
- Produces: `LegalIRPipeline.predict(query_id: str, question: str) -> list[str]`
- Produces: `run_all(config_path: Path, offline: bool = True) -> Path` returning validated ZIP path.

- [ ] **Step 1: Write failing end-to-end fixture test**

```python
import json
import zipfile
from pathlib import Path

from src.pipeline.run_all import run_all


def test_run_all_creates_valid_submission_zip(fixture_project: Path):
    output = run_all(fixture_project / "configs" / "pipeline.yaml", offline=True)
    assert output.name == "submission.zip"
    with zipfile.ZipFile(output) as archive:
        payload = json.loads(archive.read("submission.json"))
    assert set(payload) == {"test-q1", "test-q2"}
    assert all(1 <= len(row["answer"]) <= 5 for row in payload.values())
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_pipeline_e2e.py -v`  
Expected: FAIL because unified pipeline/orchestrator do not exist.

- [ ] **Step 3: Implement one shared production path**

Benchmark and public inference must call the same `LegalIRPipeline.predict`; only memory/fusion training scope differs. Remove hardcoded document `2113` fallback. If no branch returns candidates, use a deterministic corpus-prior list computed from training qrels and recorded in the run manifest.

`run_all` executes artifact verification, canonical validation/build if missing, index validation/build if missing, model local-only checks, inference, strict submission validation, packaging, and final checksum recording. It never downloads data or calls APIs in offline mode.

Make legacy scripts call `python -m` commands and use no `.venv/bin/python` assumption. Windows users run the Python commands directly.

- [ ] **Step 4: Run end-to-end and full fast tests**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_pipeline_e2e.py tests -m "not slow" -v`  
Expected: PASS.

Run: `PYTHONPATH=. .venv/bin/python -m src.pipeline.run_all --config configs/pipeline.yaml --offline`  
Expected: validated ZIP under one run directory, exact JSON bytes inside ZIP, no root submission/report duplicates.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline src/predict_submission.py scripts tests/test_pipeline_e2e.py
git commit -m "feat(pipeline): unify verified LegalIR inference" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 18: Verified Artifact Migration and Working-Tree Cleanup

**Files:**
- Modify: `src/artifacts/cli.py`
- Modify: `.gitignore`
- Test: `tests/test_artifact_cleanup.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `plan_cleanup(repo: Path, retained_manifest: Path) -> list[CleanupAction]`
- Produces: `apply_cleanup(actions, confirmation_token: str) -> None`
- Cleanup refuses any file not proven duplicate, stale by schema, or reproducible from a validated replacement.

- [ ] **Step 1: Write failing safe-cleanup test**

```python
from pathlib import Path
import pytest
from src.artifacts.cli import plan_cleanup, apply_cleanup


def test_cleanup_deletes_verified_duplicate_but_preserves_unknown_file(tmp_path: Path):
    keep = tmp_path / "artifacts/shared/canonical/v2/documents.parquet"
    duplicate = tmp_path / "data/task1_canonical/v1/documents.parquet"
    unknown = tmp_path / "notes.bin"
    for path in (keep, duplicate, unknown):
        path.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    unknown.write_bytes(b"unknown")
    actions = plan_cleanup(tmp_path, retained_manifest=tmp_path / "manifest.json")
    assert duplicate in {a.path for a in actions}
    assert unknown not in {a.path for a in actions}
    with pytest.raises(ValueError, match="confirmation token"):
        apply_cleanup(actions, confirmation_token="")
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_artifact_cleanup.py -v`  
Expected: FAIL because safe cleanup planner does not exist.

- [ ] **Step 3: Implement explicit dry-run and apply modes**

Cleanup candidates are the verified old canonical/index duplicates, stale root `chunks.parquet`, stale root submission/report copies, extracted raw directory after ZIP verification, `.playwright-mcp`, pytest caches, `__pycache__`, and OS metadata. The planner records reason and retained counterpart for every deletion. Git object pruning is excluded.

- [ ] **Step 4: Run dry-run, inspect, then apply only verified actions**

Run: `PYTHONPATH=. .venv/bin/python -m src.artifacts.cli cleanup --repo . --dry-run --output artifacts/local/cleanup-plan.json`  
Expected: every large deletion has a checksum match or validated replacement reason; no source code, shared v2 artifact, or unknown file is listed.

After reviewing the JSON, run the explicit apply command with its generated confirmation token. Then run:

```bash
PYTHONPATH=. .venv/bin/python -m src.artifacts.cli verify --root artifacts/shared
PYTHONPATH=. .venv/bin/pytest tests -m "not slow" -v
git status --short
```

Expected: shared verification and tests pass; no duplicate/orphan working-tree artifacts; Git status contains only intentional tracked changes.

- [ ] **Step 5: Commit cleanup code/docs, not deleted ignored artifacts**

```bash
git add src/artifacts/cli.py .gitignore README.md tests/test_artifact_cleanup.py
git commit -m "chore(artifacts): enforce verified workspace cleanup" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 19: Final Full Benchmark, Ablation Selection, and Submission

**Files:**
- Create: `artifacts/shared/benchmarks/accepted/final_model.json`
- Create: `artifacts/shared/submissions/accepted/submission_manifest.json`
- Modify: `README.md`
- Test: `tests/test_final_acceptance.py`

**Interfaces:**
- Final summary names the selected components and compares every accepted ablation against `strict_baseline` using identical dataset/split checksums.
- Submission manifest contains query count, JSON SHA-256, ZIP SHA-256, model/config/run IDs, and official validation result.

- [ ] **Step 1: Write failing final acceptance test**

```python
import json
from pathlib import Path


def test_final_model_passes_all_acceptance_gates():
    report = json.loads(Path("artifacts/shared/benchmarks/accepted/final_model.json").read_text())
    assert len(report["random_5fold"]["folds"]) == 5
    assert report["official_scorer_equivalent"] is True
    assert report["leakage_checks_passed"] is True
    assert report["random_5fold"]["mean_recall_at_5"] > report["baseline"]["mean_recall_at_5"]
    assert report["document_disjoint"]["recall_at_5"] >= report["baseline"]["document_disjoint_recall_at_5"] - 0.01
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_final_acceptance.py -v`  
Expected: FAIL because final accepted report does not exist.

- [ ] **Step 3: Run complete ablation matrix**

Run the same canonical and split checksums for:

1. strict baseline;
2. fielded BM25 + exact;
3. + dense retrieval;
4. + dual question memory;
5. + zero-shot reranker;
6. + LightGBM fusion;
7. + LoRA reranker only if Task 16 passed its gate.

For every run, preserve per-fold predictions locally and commit only the accepted small summary. Select by mean five-fold Recall@5, then document-disjoint Recall@5, Precision, runtime, and memory as defined by the spec.

- [ ] **Step 4: Train final local components on all training data and package public predictions**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m src.pipeline.run_all \
  --config configs/pipeline.yaml \
  --offline \
  --final-train \
  --input artifacts/shared/raw/public-official.json
```

Expected:

- exactly the public query keys;
- one to five unique official IDs per query, normally five for Recall optimization;
- official submission validator pass;
- ZIP contains exactly the current JSON bytes;
- final run manifest records model, data, config, commit, prediction, and ZIP hashes.

- [ ] **Step 5: Run all acceptance tests and inspect Git hygiene**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests -v
PYTHONPATH=. .venv/bin/python -m src.evaluation.codabench_compat --submission artifacts/local/runs/<final_run>/submission.json
PYTHONPATH=. .venv/bin/python -m src.artifacts.cli verify --root artifacts/shared
git status --short
```

Expected: all tests pass, official equivalence passes, shared artifacts verify, and only final tracked summaries/docs are changed.

- [ ] **Step 6: Commit accepted summaries and documentation**

```bash
git add artifacts/shared/benchmarks/accepted/final_model.json artifacts/shared/submissions/accepted/submission_manifest.json README.md tests/test_final_acceptance.py
git commit -m "feat(legalir): finalize strict high-recall pipeline" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

Do not push shared BTC payloads or Git LFS objects until remote privacy and redistribution authorization are explicitly confirmed.

---

## Execution Order and Review Gates

- Tasks 1–8 establish the only valid benchmark foundation. Neural work must not begin before Task 8 records a leakage-free five-fold baseline.
- Tasks 9–13 improve candidate retrieval and zero-shot reranking. Each task requires a strict ablation.
- Tasks 14–16 create local training/fusion experiments. Model artifacts remain ignored.
- Tasks 17–19 unify production inference, clean verified duplicates, and select the final submission.
- After every task: inspect `git diff`, run the named tests, and review the commit before starting the next task.
- If a task changes canonical data or split checksums, stop and regenerate every downstream benchmark/index; never compare across different checksums.
