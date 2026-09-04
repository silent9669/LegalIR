"""Arrow-backed lazy MacroEvidenceStore with bounded LRU memory caching."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class MacroChunk:
    """A single macro chunk representing an article or major section."""

    doc_id: str
    chunk_id: str
    chunk_index: int
    text: str
    approx_tokens: int = 0


@dataclass
class PreprocessedDoc:
    """Preprocessed document representation with parsed macro chunks."""

    doc_id: str
    chunks: List[MacroChunk]
    full_text: str
    byte_size: int = 0


class MacroEvidenceStore:
    """
    Arrow-backed lazy evidence store.
    Instead of holding all 219k macro chunks in Python object memory, it maintains
    an Arrow table and lazily populates an LRU cache of document chunks bounded by count & bytes.
    """

    def __init__(
        self,
        chunks_path: Union[str, Path],
        max_cache_bytes: int = 512 * 1024 * 1024,  # 512 MB default
        max_cached_docs: int = 512,
    ):
        self.chunks_path = Path(chunks_path)
        if not self.chunks_path.is_file():
            raise FileNotFoundError(f"Chunks parquet not found: {self.chunks_path}")

        self.max_cache_bytes = max_cache_bytes
        self.max_cached_docs = max_cached_docs
        self._cache: collections.OrderedDict[str, PreprocessedDoc] = collections.OrderedDict()
        self._cache_bytes_total: int = 0

        # Read only required columns from chunks table
        columns = ["doc_id", "chunk_id", "chunk_type", "text"]
        # Check if chunk_index exists
        pq_meta = pq.read_metadata(str(self.chunks_path))
        existing_cols = pq_meta.schema.names
        if "chunk_index" in existing_cols:
            columns.append("chunk_index")

        full_table = pq.read_table(str(self.chunks_path), columns=columns)
        # Filter strictly to macro chunks
        macro_mask = pc.equal(full_table["chunk_type"], "macro")
        self.macro_table = full_table.filter(macro_mask)

        # Build compact doc_id -> list of table row indices
        self.doc_index: Dict[str, List[int]] = collections.defaultdict(list)
        doc_ids = self.macro_table["doc_id"].to_pylist()
        for idx, did in enumerate(doc_ids):
            self.doc_index[did].append(idx)

    def cache_bytes(self) -> int:
        """Return current bytes occupied by cached documents."""
        return self._cache_bytes_total

    def clear_cache(self) -> None:
        """Evict all cached documents."""
        self._cache.clear()
        self._cache_bytes_total = 0

    def _evict_if_needed(self) -> None:
        """Evict least recently used documents when over capacity."""
        while len(self._cache) > self.max_cached_docs or (
            self._cache_bytes_total > self.max_cache_bytes and len(self._cache) > 1
        ):
            _, evicted_doc = self._cache.popitem(last=False)
            self._cache_bytes_total -= evicted_doc.byte_size

    def get_preprocessed_doc(self, doc_id: str) -> PreprocessedDoc:
        """Retrieve preprocessed document with macro chunks, utilizing LRU cache."""
        if doc_id in self._cache:
            # Mark as recently used
            doc = self._cache.pop(doc_id)
            self._cache[doc_id] = doc
            return doc

        row_indices = self.doc_index.get(doc_id, [])
        chunks: List[MacroChunk] = []
        texts: List[str] = []
        doc_bytes = 0

        for row_idx in row_indices:
            cid = self.macro_table["chunk_id"][row_idx].as_py()
            c_text = self.macro_table["text"][row_idx].as_py() or ""
            c_idx = (
                self.macro_table["chunk_index"][row_idx].as_py()
                if "chunk_index" in self.macro_table.column_names
                else len(chunks)
            )

            # Fast approx token count: word count
            approx_tokens = len(c_text.split())
            chunk = MacroChunk(
                doc_id=doc_id,
                chunk_id=cid,
                chunk_index=c_idx,
                text=c_text,
                approx_tokens=approx_tokens,
            )
            chunks.append(chunk)
            texts.append(c_text)
            doc_bytes += len(c_text.encode("utf-8")) + 128  # approx overhead per chunk

        full_text = "\n\n".join(texts)
        doc_bytes += len(full_text.encode("utf-8")) + 256

        prep = PreprocessedDoc(
            doc_id=doc_id,
            chunks=chunks,
            full_text=full_text,
            byte_size=doc_bytes,
        )

        self._cache[doc_id] = prep
        self._cache_bytes_total += doc_bytes
        self._evict_if_needed()
        return prep

    def get_doc_chunks(self, doc_id: str) -> List[MacroChunk]:
        """Convenience method returning macro chunks for a document."""
        return self.get_preprocessed_doc(doc_id).chunks
