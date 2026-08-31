from typing import Optional, TypedDict


class CandidateRecord(TypedDict, total=False):
    doc_id: str
    bm25_score: float
    bm25_rank: int
    bm25_raw_score: float
    bm25_best_score: float
    bm25_second_score: float
    bm25_mean_score: float
    bm25_best_chunk_id: Optional[str]
    bm25_legal_boost: float
    bm25_pyvi_score: float
    bm25_pyvi_rank: int
    bm25_pyvi_best_score: float
    bm25_pyvi_second_score: float
    bm25_pyvi_mean_score: float
    bm25_pyvi_best_chunk_id: Optional[str]
    exact_score: float
    exact_match_score: float
    exact_legal_number: bool
    exact_article: bool
    exact_clause: bool
    exact_point: bool
    exact_year: bool
    exact_doc_type: bool
    exact_title: bool
    exact_title_overlap: float
    memory_score: float
    memory_rank: int
    memory_lexical_similarity: float
    memory_dense_similarity: float
    memory_vote_count: int
    dense_score: float
    dense_rank: int
    dense_best_score: float
    dense_second_score: float
    dense_mean_score: float
    dense_best_chunk_id: Optional[str]
    rrf_score: float
    source_count: int
    branch_ranks: dict[str, int]
    branch_contributions: dict[str, float]
    branch_metadata: dict[str, dict[str, object]]
