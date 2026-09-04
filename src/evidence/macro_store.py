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
    text_raw: str = ""
    text_norm: str = ""
    article: str = ""
    clause: str = ""
    point: str = ""
    chapter: str = ""
    section: str = ""
    approx_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        raw = self.text_raw or self.text
        norm = self.text_norm or self.text
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "text_raw": raw,
            "text_norm": norm,
            "article": self.article,
            "clause": self.clause,
            "point": self.point,
            "chapter": self.chapter,
            "section": self.section,
            "body": raw or norm or self.text,
            "granularity": "macro",
        }


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

        # Determine dynamic schema columns
        pq_meta = pq.read_metadata(str(self.chunks_path))
        existing_cols = pq_meta.schema.names

        self.type_col = "granularity" if "granularity" in existing_cols else "chunk_type"
        self.text_col = (
            "text_raw"
            if "text_raw" in existing_cols
            else ("text_norm" if "text_norm" in existing_cols else "text")
        )

        possible_cols = [
            "doc_id", "chunk_id", self.type_col,
            "text_raw", "text_norm", "text",
            "article", "clause", "point", "chapter", "section", "chunk_index"
        ]
        columns = [c for c in possible_cols if c in existing_cols]

        full_table = pq.read_table(str(self.chunks_path), columns=columns)
        # Filter strictly to macro chunks
        macro_mask = pc.equal(full_table[self.type_col], "macro")
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
            c_text = self.macro_table[self.text_col][row_idx].as_py() or ""
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
                text_raw=str(self.macro_table["text_raw"][row_idx].as_py() or "") if "text_raw" in self.macro_table.column_names else "",
                text_norm=str(self.macro_table["text_norm"][row_idx].as_py() or "") if "text_norm" in self.macro_table.column_names else "",
                article=str(self.macro_table["article"][row_idx].as_py() or "") if "article" in self.macro_table.column_names else "",
                clause=str(self.macro_table["clause"][row_idx].as_py() or "") if "clause" in self.macro_table.column_names else "",
                point=str(self.macro_table["point"][row_idx].as_py() or "") if "point" in self.macro_table.column_names else "",
                chapter=str(self.macro_table["chapter"][row_idx].as_py() or "") if "chapter" in self.macro_table.column_names else "",
                section=str(self.macro_table["section"][row_idx].as_py() or "") if "section" in self.macro_table.column_names else "",
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
