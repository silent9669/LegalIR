"""
Pair materializer with duplicate blacklist, fold-local question memory,
multi-band hard negative mining, and strict validation leakage assertions.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.memory import release_memory
from src.evaluation.benchmark import build_memory_rows
from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.selector import LazyEvidencePackBuilder, LazyPositiveLocalizer
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.static_cache import StaticCacheReader, StaticCandidateRecord
from src.training.hard_negative_miner import HardNegativeMiner


RERANKER_PAIRS_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("query_text", pa.string()),
    ("doc_id", pa.string()),
    ("label", pa.float32()),
    ("negative_source", pa.string()),
    ("retrieval_rank", pa.int32()),
    ("retrieval_score", pa.float32()),
    ("evidence_chunk_ids", pa.string()),
    ("evidence_text", pa.string()),
    ("fold", pa.int32()),
])

VALIDATION_CANDIDATES_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("query_text", pa.string()),
    ("doc_id", pa.string()),
    ("gold_doc_ids", pa.string()),
    ("rrf_score", pa.float32()),
    ("evidence_text", pa.string()),
    ("fold", pa.int32()),
])


def build_duplicate_closure(duplicate_groups: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Build transitive duplicate document mapping from duplicate groups."""
    closure: Dict[str, Set[str]] = collections.defaultdict(set)

    groups_list: List[List[str]] = []
    if isinstance(duplicate_groups, dict):
        for g in duplicate_groups.values():
            if isinstance(g, list):
                groups_list.append([str(x) for x in g])
    elif isinstance(duplicate_groups, list):
        for g in duplicate_groups:
            if isinstance(g, dict) and "doc_ids" in g:
                groups_list.append([str(x) for x in g["doc_ids"]])
            elif isinstance(g, list):
                groups_list.append([str(x) for x in g])

    for doc_ids in groups_list:
        doc_set = set(doc_ids)
        for did in doc_set:
            closure[did].update(doc_set)

    return dict(closure)


class PairMaterializer:
    """
    Generates fold-safe training pairs and validation candidates.
    Preserves legacy multi-band hard-negative policy, branch limits,
    fold-local Question Memory, and duplicate closures.
    """

    def __init__(
        self,
        train_qids: Set[str],
        val_qids: Set[str],
        qrels: Dict[str, List[str]],
        duplicate_groups: Union[List[Dict[str, Any]], Dict[str, Any]],
        evidence_store: Optional[MacroEvidenceStore] = None,
        doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        fold: int = 0,
        question_memory: Optional[QuestionMemory] = None,
        branch_weights: Optional[Dict[str, float]] = None,
    ):
        self.train_qids = set(map(str, train_qids))
        self.val_qids = set(map(str, val_qids))
        self.qrels = {str(k): [str(d) for d in v] for k, v in qrels.items()}
        self.duplicate_closure = build_duplicate_closure(duplicate_groups)
        self.evidence_store = evidence_store
        self.doc_metadata = doc_metadata or {}
        self.fold = int(fold)
        self.branch_weights = branch_weights or {
            "bm25": 1.0,
            "bm25_pyvi": 1.0,
            "dense": 1.0,
            "exact": 0.5,
            "memory": 1.2,
        }

        # Strict fold isolation assertion
        overlap = self.train_qids.intersection(self.val_qids)
        if overlap:
            raise ValueError(f"Validation leakage detected: train and val qids overlap on {len(overlap)} IDs")

        # Question memory assertion: if memory passed, its keys must be subset of train_qids
        if question_memory is not None:
            mem_qids = set(map(str, getattr(question_memory, "qids", getattr(question_memory, "training_query_ids", getattr(question_memory, "query_ids", [])))))
            if mem_qids:
                if not mem_qids.issubset(self.train_qids):
                    leak = mem_qids.intersection(self.val_qids)
                    raise ValueError(f"Memory leakage detected: {len(leak)} validation qids found in question memory")

        self.question_memory = question_memory

        if evidence_store is not None:
            self.localizer = LazyPositiveLocalizer(evidence_store)
            self.evidence_builder = LazyEvidencePackBuilder(evidence_store, doc_metadata=self.doc_metadata)
        else:
            self.localizer = None
            self.evidence_builder = None

        # Build false negative blacklist
        self.query_blacklist: Dict[str, Set[str]] = collections.defaultdict(set)
        for qid in self.train_qids:
            for gd in self.qrels.get(qid, []):
                self.query_blacklist[qid].add(gd)
                if gd in self.duplicate_closure:
                    self.query_blacklist[qid].update(self.duplicate_closure[gd])

        self.miner = HardNegativeMiner(false_negative_blacklist=self.query_blacklist)

    def assert_fold_isolation(self, qid: str) -> None:
        """Assert that query belongs strictly to train and never to val."""
        qid = str(qid)
        if qid in self.val_qids:
            raise ValueError(f"Validation leakage detected: query {qid} belongs to validation set")
        if qid not in self.train_qids:
            raise ValueError(f"Query {qid} not in train set")

    def is_negative_allowed(self, qid: str, neg_doc_id: str) -> bool:
        """Check whether neg_doc_id is allowed as a negative for qid."""
        qid = str(qid)
        neg_doc_id = str(neg_doc_id)
        if neg_doc_id in self.query_blacklist[qid]:
            return False
        return True

    def _assemble_candidates(
        self,
        qid: str,
        q_text: str,
        static_cache_reader: Optional[StaticCacheReader],
        query_embeddings: Optional[Mapping[str, Any]] = None,
        candidate_k: int = 150,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Assemble candidates across all branches, including fold-local Question Memory and hybrid fusion."""
        candidates_by_source: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        branch_results: Dict[str, Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]] = {}

        if static_cache_reader is not None:
            static_cands = static_cache_reader.get_query_candidates(qid)
            # Group by branch
            by_branch: Dict[str, List[StaticCandidateRecord]] = collections.defaultdict(list)
            for sc in static_cands:
                b = sc.branch
                if b == "bm25_legal":
                    b = "bm25"
                by_branch[b].append(sc)

            for b_name in ["exact", "bm25", "bm25_pyvi", "dense"]:
                items = by_branch.get(b_name, [])
                if items:
                    ranks_dict = {c.doc_id: c.rank for c in items}
                    details_dict = {
                        c.doc_id: {
                            "doc_id": c.doc_id,
                            "score": c.score,
                            f"{b_name}_score": c.score,
                            f"{b_name}_best_chunk_id": c.best_chunk_id,
                        }
                        for c in items
                    }
                    branch_results[b_name] = (ranks_dict, details_dict)
                    candidates_by_source[b_name] = [
                        {
                            "doc_id": c.doc_id,
                            "score": c.score,
                            "rank": c.rank,
                            "retrieval_rank": c.rank,
                            "retrieval_score": c.score,
                            "source": b_name,
                        }
                        for c in items[:30]
                    ]

        # Fold-local Question Memory candidates
        if self.question_memory is not None:
            q_emb = query_embeddings.get(qid) if query_embeddings else None
            mem_items = self.question_memory.query(q_text, exclude_qid=qid, top_k=10, q_emb=q_emb)
            if mem_items:
                m_ranks = {}
                m_details = {}
                m_source_items = []
                for i, m in enumerate(mem_items, start=1):
                    did = str(m["doc_id"])
                    score = float(m.get("similarity", 0.0))
                    m_ranks[did] = i
                    m_details[did] = {"doc_id": did, "score": score, "memory_score": score}
                    m_source_items.append({
                        "doc_id": did,
                        "score": score,
                        "rank": i,
                        "retrieval_rank": i,
                        "retrieval_score": score,
                        "source": "memory",
                    })
                branch_results["memory"] = (m_ranks, m_details)
                candidates_by_source["memory"] = m_source_items

        # Hybrid RRF fusion
        hybrid_cands = []
        if branch_results:
            hybrid_records = HybridSearchEngine._fuse(
                branch_results=branch_results,
                top_k_candidates=candidate_k,
                rrf_k=60,
                branch_weights=self.branch_weights,
            )
            for rank, hr in enumerate(hybrid_records, start=1):
                did = hr.get("doc_id") if isinstance(hr, dict) else hr.doc_id
                score = hr.get("rrf_score", 0.0) if isinstance(hr, dict) else getattr(hr, "rrf_score", 0.0)
                hybrid_cands.append({
                    "doc_id": did,
                    "score": score,
                    "rank": rank,
                    "retrieval_rank": rank,
                    "retrieval_score": score,
                    "source": "hybrid",
                })

        candidates_by_source["hybrid"] = hybrid_cands[:30]
        candidates_by_source["medium_neg"] = hybrid_cands[20:80] if len(hybrid_cands) > 20 else []

        return dict(candidates_by_source), hybrid_cands

    def materialize_train_pairs(
        self,
        output_parquet: Union[str, Path],
        queries_dict: Dict[str, str],
        static_cache_reader: Optional[StaticCacheReader] = None,
        query_embeddings: Optional[Mapping[str, Any]] = None,
        negatives_per_positive: int = 10,
        max_evidence_chunks: int = 3,
        batch_size: int = 5000,
    ) -> int:
        """
        Materialize all train pairs for the fold into Parquet incrementally.
        Guarantees exact negative source policy, duplicate blacklist, and zero leakage.
        """
        if self.evidence_builder is None or self.localizer is None:
            raise ValueError("MacroEvidenceStore is required for pair materialization.")

        output_p = Path(output_parquet)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(output_p), RERANKER_PAIRS_SCHEMA)

        per_source_limits = {
            "exact": 2,
            "bm25": 4,
            "bm25_pyvi": 3,
            "dense": 3,
            "memory": 2,
            "hybrid": 4,
            "medium_neg": 3,
        }

        buffer: List[Dict[str, Any]] = []
        total_rows = 0

        for qid in sorted(self.train_qids):
            self.assert_fold_isolation(qid)
            q_text = queries_dict.get(qid, "")
            gold_ids = self.qrels.get(qid, [])
            if not gold_ids or not q_text:
                continue

            # Multi-band candidates
            cands_by_source, _ = self._assemble_candidates(
                qid=qid,
                q_text=q_text,
                static_cache_reader=static_cache_reader,
                query_embeddings=query_embeddings,
            )

            # Mine hard negatives
            mined_neg_records = self.miner.mine_multi_band_negatives(
                query_id=qid,
                candidates_by_source=cands_by_source,
                gold_doc_ids=gold_ids,
                per_source_limits=per_source_limits,
                max_total=negatives_per_positive * len(gold_ids),
            )

            # Positive pairs
            for gold_id in gold_ids:
                pos_chunk = self.localizer.localize(q_text, gold_id)
                pos_cid = pos_chunk.get("chunk_id") if isinstance(pos_chunk, dict) else getattr(pos_chunk, "chunk_id", None)
                pos_ev = self.evidence_builder.build_pack(q_text, gold_id, max_chunks=max_evidence_chunks)

                buffer.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "doc_id": gold_id,
                    "label": 1.0,
                    "negative_source": "gold",
                    "retrieval_rank": 0,
                    "retrieval_score": 1.0,
                    "evidence_chunk_ids": json.dumps([pos_cid] if pos_cid else []),
                    "evidence_text": pos_ev,
                    "fold": self.fold,
                })

                # Negative pairs
                for neg_rec in mined_neg_records:
                    neg_id = str(neg_rec["doc_id"])
                    neg_chunk = self.localizer.localize(q_text, neg_id)
                    neg_cid = neg_chunk.get("chunk_id") if isinstance(neg_chunk, dict) else getattr(neg_chunk, "chunk_id", None)
                    neg_ev = self.evidence_builder.build_pack(q_text, neg_id, max_chunks=max_evidence_chunks)

                    buffer.append({
                        "query_id": qid,
                        "query_text": q_text,
                        "doc_id": neg_id,
                        "label": 0.0,
                        "negative_source": str(neg_rec.get("negative_source", "hybrid")),
                        "retrieval_rank": int(neg_rec.get("retrieval_rank", 1)),
                        "retrieval_score": float(neg_rec.get("retrieval_score", 0.0)),
                        "evidence_chunk_ids": json.dumps([neg_cid] if neg_cid else []),
                        "evidence_text": neg_ev,
                        "fold": self.fold,
                    })

            if len(buffer) >= batch_size:
                tbl = pa.Table.from_pylist(buffer, schema=RERANKER_PAIRS_SCHEMA)
                writer.write_table(tbl)
                total_rows += len(buffer)
                buffer.clear()

        if buffer:
            tbl = pa.Table.from_pylist(buffer, schema=RERANKER_PAIRS_SCHEMA)
            writer.write_table(tbl)
            total_rows += len(buffer)
            buffer.clear()

        writer.close()
        return total_rows

    def materialize_validation_candidates(
        self,
        output_parquet: Union[str, Path],
        queries_dict: Dict[str, str],
        static_cache_reader: Optional[StaticCacheReader] = None,
        candidate_k: int = 150,
        max_evidence_chunks: int = 2,
        batch_size: int = 5000,
    ) -> int:
        """
        Materialize validation candidates for the fold into Parquet.
        Combines static branch cache + fold-local question memory.
        """
        output_p = Path(output_parquet)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(output_p), VALIDATION_CANDIDATES_SCHEMA)

        buffer: List[Dict[str, Any]] = []
        total_rows = 0

        for qid in sorted(self.val_qids):
            q_text = queries_dict.get(qid, "")
            gold_ids = self.qrels.get(qid, [])
            if not q_text:
                continue

            _, hybrid_cands = self._assemble_candidates(
                qid=qid,
                q_text=q_text,
                static_cache_reader=static_cache_reader,
                candidate_k=candidate_k,
            )

            gold_ids_str = json.dumps(gold_ids)
            for c in hybrid_cands[:candidate_k]:
                did = c["doc_id"]
                score = c["score"]
                ev_text = (
                    self.evidence_builder.build_pack(q_text, did, max_chunks=max_evidence_chunks)
                    if self.evidence_builder
                    else ""
                )
                buffer.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "doc_id": did,
                    "gold_doc_ids": gold_ids_str,
                    "rrf_score": float(score),
                    "evidence_text": ev_text,
                    "fold": self.fold,
                })

            if len(buffer) >= batch_size:
                tbl = pa.Table.from_pylist(buffer, schema=VALIDATION_CANDIDATES_SCHEMA)
                writer.write_table(tbl)
                total_rows += len(buffer)
                buffer.clear()

        if buffer:
            tbl = pa.Table.from_pylist(buffer, schema=VALIDATION_CANDIDATES_SCHEMA)
            writer.write_table(tbl)
            total_rows += len(buffer)
            buffer.clear()

        writer.close()
        return total_rows
