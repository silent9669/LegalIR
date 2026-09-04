"""Static retrieval candidate cache schema, streaming Parquet writer, and reader."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STATIC_CACHE_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("branch", pa.string()),
    ("rank", pa.int32()),
    ("doc_id", pa.string()),
    ("score", pa.float32()),
    ("best_chunk_id", pa.string()),
    ("second_score", pa.float32()),
    ("mean_score", pa.float32()),
    ("extra_json", pa.string()),
])


@dataclass
class StaticCandidateRecord:
    """A single candidate document retrieved by a static retrieval branch."""

    query_id: str
    branch: str
    rank: int
    doc_id: str
    score: float
    best_chunk_id: Optional[str] = None
    second_score: Optional[float] = None
    mean_score: Optional[float] = None
    extra_json: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StaticCacheWriter:
    """
    Streams static retrieval candidates to normalized Parquet in bounded batches.
    Guaranteed label-free: takes no qrels, ground-truth, or fold labels.
    """

    def __init__(self, output_path: Union[str, Path], batch_size: int = 5000):
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.buffer: List[StaticCandidateRecord] = []
        self.writer: Optional[pq.ParquetWriter] = None
        self.total_written = 0

    def _init_writer(self) -> None:
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(str(self.output_path), STATIC_CACHE_SCHEMA)

    def write_record(self, record: StaticCandidateRecord) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def write_records(self, records: List[StaticCandidateRecord]) -> None:
        for r in records:
            self.write_record(r)

    def flush(self) -> None:
        if not self.buffer:
            return
        self._init_writer()

        pydict = {
            "query_id": [r.query_id for r in self.buffer],
            "branch": [r.branch for r in self.buffer],
            "rank": [r.rank for r in self.buffer],
            "doc_id": [r.doc_id for r in self.buffer],
            "score": [float(r.score) for r in self.buffer],
            "best_chunk_id": [r.best_chunk_id for r in self.buffer],
            "second_score": [float(r.second_score) if r.second_score is not None else None for r in self.buffer],
            "mean_score": [float(r.mean_score) if r.mean_score is not None else None for r in self.buffer],
            "extra_json": [r.extra_json for r in self.buffer],
        }

        table = pa.Table.from_pydict(pydict, schema=STATIC_CACHE_SCHEMA)
        self.writer.write_table(table)
        self.total_written += len(self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class StaticCacheReader:
    """Reads static retrieval candidates efficiently without loading underlying models."""

    def __init__(self, cache_path: Union[str, Path]):
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Static cache file not found: {self.cache_path}")
        self._table: Optional[pa.Table] = None
        self._df: Optional[pd.DataFrame] = None

    def _ensure_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pq.read_table(str(self.cache_path)).to_pandas()
        return self._df

    def get_query_ids(self) -> List[str]:
        df = self._ensure_df()
        return df["query_id"].drop_duplicates().tolist()

    def get_query_candidates(
        self, query_id: str, branch: Optional[str] = None
    ) -> List[StaticCandidateRecord]:
        df = self._ensure_df()
        sub = df[df["query_id"] == query_id]
        if branch is not None:
            sub = sub[sub["branch"] == branch]

        records: List[StaticCandidateRecord] = []
        for _, row in sub.iterrows():
            records.append(
                StaticCandidateRecord(
                    query_id=row["query_id"],
                    branch=row["branch"],
                    rank=int(row["rank"]),
                    doc_id=row["doc_id"],
                    score=float(row["score"]),
                    best_chunk_id=row["best_chunk_id"] if pd.notna(row["best_chunk_id"]) else None,
                    second_score=float(row["second_score"]) if pd.notna(row["second_score"]) else None,
                    mean_score=float(row["mean_score"]) if pd.notna(row["mean_score"]) else None,
                    extra_json=row["extra_json"] if pd.notna(row["extra_json"]) else None,
                )
            )
        return records

    def get_dataframe(self) -> pd.DataFrame:
        return self._ensure_df()
