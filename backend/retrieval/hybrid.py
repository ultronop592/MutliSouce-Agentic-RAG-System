"""
Hybrid Search Engine — Reciprocal Rank Fusion (RRF)
----------------------------------------------------
Runs TWO completely independent searches and fuses the ranked results:

  1. Semantic Search  — Qdrant vector (cosine) search
                        finds conceptually / semantically similar chunks
                        even when exact keywords differ.

  2. BM25 Search      — Full-corpus keyword search via BM25Okapi
                        finds chunks with high keyword / term overlap.
                        Runs on ALL documents in the collection (not just
                        the top-N from vector search), so it's truly
                        independent.

  3. Fusion           — Reciprocal Rank Fusion (RRF) combines both ranked
                        lists into a single ranking without needing to
                        normalise scores from different scales.

Why RRF?
  Vector scores are cosine similarities (0-1 range, distribution varies).
  BM25 scores are TF-IDF-like (0-∞ range, collection-dependent).
  RRF avoids score normalisation by working purely on RANK positions:

      RRF(doc) = Σ  1 / (k + rank_i)      k = 60 (standard constant)

  A document ranked #1 by BOTH searches gets ~2 × (1/61) ≈ 0.033,
  outscoring anything ranked high by only one leg.
"""

from retrieval.bm25_index import bm25_manager

# RRF constant — 60 is the standard value from the original RRF paper
RRF_K: int = 60

# How many results to request from each search leg
SEMANTIC_TOP_K: int = 10
BM25_TOP_K: int = 10

# Final number of fused documents to surface
FINAL_TOP_K: int = 6


def reciprocal_rank_fusion(
    semantic_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        semantic_results: [(text, score), ...] sorted descending by vector score
        bm25_results:     [(text, score), ...] sorted descending by BM25 score
        k:                RRF smoothing constant (default 60)

    Returns:
        [(text, rrf_score), ...] sorted descending by fused RRF score
    """
    rrf_scores: dict[str, float] = {}

    # Accumulate RRF score from the semantic (vector) leg
    for rank, (text, _score) in enumerate(semantic_results, start=1):
        rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)

    # Accumulate RRF score from the BM25 leg
    for rank, (text, _score) in enumerate(bm25_results, start=1):
        rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)

    # Sort by combined RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused  # [(text, rrf_score), ...]


def hybrid_search_collection(
    collection: str,
    query: str,
    semantic_hits: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """
    Run BM25 search on `collection` and fuse with the semantic hits using RRF.

    Args:
        collection:    Qdrant collection name
        query:         rewritten search query string
        semantic_hits: [(text, vector_score), ...] already retrieved from Qdrant

    Returns:
        [(text, rrf_score), ...] fused and sorted, capped to FINAL_TOP_K
    """
    # BM25 search — independent, searches the FULL corpus index
    bm25_hits = bm25_manager.search(collection, query, top_k=BM25_TOP_K)

    if not bm25_hits:
        # No BM25 index yet (e.g., empty collection) — fall back to semantic only
        return [(t, s) for t, s in semantic_hits[:FINAL_TOP_K]]

    # RRF fusion
    fused = reciprocal_rank_fusion(semantic_hits, bm25_hits)
    return fused[:FINAL_TOP_K]


def fuse_all_collections(
    per_collection_semantic: dict[str, list[tuple[str, float]]],
    query: str,
) -> list[tuple[str, float]]:
    """
    Run hybrid search across ALL selected collections and merge into one
    globally-ranked list.

    Args:
        per_collection_semantic: {collection: [(text, score), ...]}
        query:                   rewritten query string

    Returns:
        [(text, rrf_score), ...] global ranking, capped to FINAL_TOP_K
    """
    all_fused: list[tuple[str, float]] = []

    for collection, semantic_hits in per_collection_semantic.items():
        col_results = hybrid_search_collection(collection, query, semantic_hits)
        all_fused.extend(col_results)

    # Global RRF — deduplicate across collections by text identity,
    # keeping the highest RRF score when the same chunk appears in multiple collections
    deduped: dict[str, float] = {}
    for text, score in all_fused:
        if text not in deduped or score > deduped[text]:
            deduped[text] = score

    ranked = sorted(deduped.items(), key=lambda x: x[1], reverse=True)
    return ranked[:FINAL_TOP_K]
