"""Pair materializer with duplicate blacklist and strict validation leakage assertions."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pyarrow as pa
import pyarrow.parquet as pq

from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.selector import LazyEvidencePackBuilder, LazyPositiveLocalizer
from src.retrieval.static_cache import StaticCacheReader


TRAIN_PAIRS_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("query_text", pa.string()),
    ("doc_id", pa.string()),
    ("label", pa.int32()),
    ("evidence_text", pa.string()),
    ("source_branch", pa.string()),
])


def build_duplicate_closure(duplicate_groups: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Build transitive duplicate document mapping from duplicate groups."""
    closure: Dict[str, Set[str]] = {}
    for group in duplicate_groups:
        doc_ids = set(group.get("doc_ids", []))
        for did in doc_ids:
            if did not in closure:
                closure[did] = set()
            closure[did].update(doc_ids)
    return closure


class PairMaterializer:
    """
    Generates training pairs for reranker training while strictly enforcing:
    1. Fold train isolation (zero validation queries in training pairs).
    2. Transitive duplicate blacklist for negative candidate filtering.
    3. Memory-bounded streaming to Parquet via PyArrow.
    """

    def __init__(
        self,
        train_qids: Set[str],
        val_qids: Set[str],
        qrels: Dict[str, List[str]],
        duplicate_groups: List[Dict[str, Any]],
        evidence_store: Optional[MacroEvidenceStore] = None,
        doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.train_qids = set(train_qids)
        self.val_qids = set(val_qids)
        self.qrels = {str(k): [str(d) for d in v] for k, v in qrels.items()}
        self.duplicate_closure = build_duplicate_closure(duplicate_groups)
        self.evidence_store = evidence_store
        self.doc_metadata = doc_metadata or {}
        self.pack_builder = (
            LazyEvidencePackBuilder(evidence_store, self.doc_metadata)
            if evidence_store is not None
            else None
        )

        # Assert disjointness at construction
        overlap = self.train_qids.intersection(self.val_qids)
        if overlap:
            raise ValueError(f"Validation leakage detected: train and val qids overlap on {len(overlap)} IDs")

    def assert_fold_isolation(self, qid: str) -> None:
        """Enforce that qid belongs strictly to train and never to val."""
        if qid in self.val_qids:
            raise ValueError(f"Validation leakage detected: query {qid} belongs to validation set")
        if qid not in self.train_qids:
            raise ValueError(f"Query {qid} not in train set")

    def is_negative_allowed(self, qid: str, neg_doc_id: str) -> bool:
        """Check whether neg_doc_id is permitted as a negative for qid."""
        gold_docs = self.qrels.get(str(qid), [])
        neg_doc_id = str(neg_doc_id)

        for gd in gold_docs:
            if neg_doc_id == gd:
                return False
            # Check duplicate closure
            if gd in self.duplicate_closure and neg_doc_id in self.duplicate_closure[gd]:
                return False

        return True

    def build_pairs_to_parquet(
        self,
        output_parquet: Union[str, Path],
        queries_dict: Dict[str, str],
        static_cache_reader: StaticCacheReader,
        negatives_per_positive: int = 7,
        batch_size: int = 5000,
    ) -> int:
        """
        Materialize all train pairs for the fold into Parquet incrementally.
        Returns total number of rows written.
        """
        if self.pack_builder is None:
            raise ValueError("MacroEvidenceStore is required to materialize evidence text.")

        output_p = Path(output_parquet)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(output_p), TRAIN_PAIRS_SCHEMA)

        buffer: List[Dict[str, Any]] = []
        total_rows = 0

        for qid in sorted(self.train_qids):
            self.assert_fold_isolation(qid)
            q_text = queries_dict.get(qid, "")
            gold_docs = self.qrels.get(qid, [])
            if not gold_docs or not q_text:
                continue

            # Add positives
            for gd in gold_docs:
                ev_text = self.pack_builder.build_pack(q_text, gd)
                buffer.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "doc_id": gd,
                    "label": 1,
                    "evidence_text": ev_text,
                    "source_branch": "gold",
                })

            # Sample negatives from static cache candidates
            cands = static_cache_reader.get_query_candidates(qid)
            sampled_negs = 0
            needed_negs = len(gold_docs) * negatives_per_positive

            for cand in cands:
                if sampled_negs >= needed_negs:
                    break
                if self.is_negative_allowed(qid, cand.doc_id):
                    ev_text = self.pack_builder.build_pack(q_text, cand.doc_id)
                    buffer.append({
                        "query_id": qid,
                        "query_text": q_text,
                        "doc_id": cand.doc_id,
                        "label": 0,
                        "evidence_text": ev_text,
                        "source_branch": cand.branch,
                    })
                    sampled_negs += 1

            if len(buffer) >= batch_size:
                tbl = pa.Table.from_pylist(buffer, schema=TRAIN_PAIRS_SCHEMA)
                writer.write_table(tbl)
                total_rows += len(buffer)
                buffer.clear()

        if buffer:
            tbl = pa.Table.from_pylist(buffer, schema=TRAIN_PAIRS_SCHEMA)
            writer.write_table(tbl)
            total_rows += len(buffer)
            buffer.clear()

        writer.close()
        return total_rows
