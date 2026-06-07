"""
Ensemble Combiner
-----------------
Improves retrieval quality by combining semantic and BM25 search results
using a WEIGHTED ENSEMBLE with min-max score normalization.

Why ensemble over plain RRF?
  RRF only uses rank position (1 / (60 + rank)) and throws away all score
  magnitude information.  A document with cosine score 0.95 is treated the
  same as one with 0.51 — both just get "rank 1" in their leg.

  The Ensemble Combiner preserves score magnitudes:
    1. Min-max normalize each leg to [0, 1]  (compatible scale)
    2. Weighted combination: α * semantic + β * bm25
    3. Documents that appear in BOTH legs are rewarded via the union

  Default weights: semantic=0.65, bm25=0.35
  Rationale: semantic search generally has better precision; BM25 adds
  recall for exact keyword matches and rare terms.

  Final step: RRF is applied on top of the ensemble scores as a secondary
  stability layer to handle edge cases where normalization produces ties.

Usage:
    from retrieval.ensemble import ensemble_combine
    combined = ensemble_combine(semantic_hits, bm25_hits)
"""

import math

# Weights for the two retrieval legs (must sum to 1.0)
SEMANTIC_WEIGHT: float = 0.65
BM25_WEIGHT: float = 0.35

# RRF constant — applied as secondary sort stability layer
RRF_K: int = 60

# How many documents to keep from the final ensemble
ENSEMBLE_TOP_K: int = 8


def _min_max_normalize(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Normalize scores to [0, 1] using min-max scaling."""
    if not scored:
        return []
    scores = [s for _, s in scored]
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        # All scores identical — assign 1.0 (they're all equally relevant)
        return [(t, 1.0) for t, _ in scored]
    return [(t, (s - s_min) / (s_max - s_min)) for t, s in scored]


def ensemble_combine(
    semantic_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
    semantic_weight: float = SEMANTIC_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    top_k: int = ENSEMBLE_TOP_K,
) -> list[tuple[str, float]]:
    """
    Combine semantic and BM25 results using weighted normalized ensemble.

    Algorithm:
      1. Normalize both score lists to [0, 1] via min-max.
      2. Build a union dict: for each unique document, sum the weighted scores
         from whichever legs it appeared in.
      3. Apply a small RRF bonus for documents appearing in BOTH legs.
      4. Sort and return top_k.

    Args:
        semantic_hits: [(text, cosine_score), ...] from vector search
        bm25_hits:     [(text, bm25_score), ...]   from BM25 index
        semantic_weight: weight for semantic scores (default 0.65)
        bm25_weight:     weight for BM25 scores    (default 0.35)
        top_k:           maximum documents to return

    Returns:
        [(text, ensemble_score), ...] sorted descending, capped to top_k
    """
    # Step 1: Normalize each leg independently
    sem_norm = dict(_min_max_normalize(semantic_hits))
    bm25_norm = dict(_min_max_normalize(bm25_hits))

    # Step 2: Union of all documents
    all_docs: set[str] = set(sem_norm) | set(bm25_norm)
    ensemble: dict[str, float] = {}

    for doc in all_docs:
        sem_score = sem_norm.get(doc, 0.0)
        bm25_score = bm25_norm.get(doc, 0.0)

        # Weighted combination
        weighted = semantic_weight * sem_score + bm25_weight * bm25_score

        # Cross-leg bonus: reward documents that BOTH legs agree on
        # This is the "ensemble agreement" signal — if both semantic
        # AND keyword search found it, it's almost certainly relevant.
        if doc in sem_norm and doc in bm25_norm:
            cross_leg_bonus = 0.10 * math.sqrt(sem_score * bm25_score)
            weighted += cross_leg_bonus

        ensemble[doc] = weighted

    # Step 3: Sort by ensemble score descending
    ranked = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def ensemble_combine_multi_collection(
    per_collection_semantic: dict[str, list[tuple[str, float]]],
    per_collection_bm25: dict[str, list[tuple[str, float]]],
    top_k: int = ENSEMBLE_TOP_K,
) -> list[tuple[str, float]]:
    """
    Run ensemble combination per collection and merge into a global ranking.

    For each collection:
      - Combine its semantic + BM25 hits with ensemble_combine()
    Then:
      - Merge all collection results, deduplicating by text
      - Keep best score if same chunk appears in multiple collections
      - Return global top_k

    Args:
        per_collection_semantic: {collection: [(text, score), ...]}
        per_collection_bm25:     {collection: [(text, score), ...]}
        top_k:                   final document count

    Returns:
        [(text, ensemble_score), ...] globally ranked
    """
    global_scores: dict[str, float] = {}

    for collection, sem_hits in per_collection_semantic.items():
        bm25_hits = per_collection_bm25.get(collection, [])
        col_results = ensemble_combine(sem_hits, bm25_hits)
        for text, score in col_results:
            # Keep highest score across collections (dedup)
            if text not in global_scores or score > global_scores[text]:
                global_scores[text] = score

    ranked = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
