from typing import Final

CANONICAL_DOCUMENTS_COLUMNS: Final[list[str]] = [
    "doc_id",
    "name_raw",
    "title",
    "link",
    "passage_raw",
    "passage_norm",
    "legal_number",
    "year",
    "doc_type",
    "is_empty",
]

CANONICAL_CHUNKS_COLUMNS: Final[list[str]] = [
    "chunk_id",
    "doc_id",
    "granularity",
    "chapter",
    "section",
    "article",
    "clause",
    "point",
    "text_raw",
    "text_norm",
    "parent_chunk_id",
    "token_count",
    "is_empty",
]

CANONICAL_QUERIES_COLUMNS: Final[list[str]] = [
    "query_id",
    "question_raw",
    "question_norm",
    "gold_count",
]

CANONICAL_QRELS_COLUMNS: Final[list[str]] = [
    "query_id",
    "doc_id",
    "relevance",
]
