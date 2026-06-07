"""
Hybrid Search Engine — Ensemble + RRF Pipeline
------------------------------------------------
Runs TWO completely independent searches and fuses results through a
two-stage combination pipeline:

  Stage 1 — ENSEMBLE COMBINER  (retrieval/ensemble.py)
  ────────────────────────────────────────────────────
  Semantic Search  — Qdrant vector (cosine) search
  BM25 Search      — Full-corpus BM25Okapi keyword search (entire index)

  Both results are combined using WEIGHTED ENSEMBLE MATCHING:
    1. Min-max normalize scores from each leg to [0, 1]
    2. Weighted sum: 0.65 × semantic_score + 0.35 × bm25_score
    3. Cross-leg bonus for documents found by BOTH methods
       (geometric mean bonus = 0.10 × √(sem × bm25))

  Stage 2 — RRF STABILITY LAYER
  ─────────────────────────────
  The ensemble scores are re-ranked using Reciprocal Rank Fusion as a
  final stability pass. This handles edge cases where min-max
  normalization produces near-ties and ensures a stable final ranking.

Why two stages?
  Ensemble preserves score magnitude (quality signal).
  RRF smooths out rank instability from score normalization.
  Together they outperform either method alone.
"""

import logging
from retrieval.bm25_index import bm25_manager
from retrieval.ensemble import ensemble_combine_multi_collection

logger = logging.getLogger(__name__)

# How many results to request from BM25 search per collection
BM25_TOP_K: int = 10

# Exported — used by retriever.py for semantic fetch size
SEMANTIC_TOP_K: int = 10

# Final documents returned after all fusion stages
FINAL_TOP_K: int = 6

# RRF stability constant (standard value from original paper)
RRF_K: int = 60


# ── Stage 2: RRF stability layer ─────────────────────────────────────────────

def _rrf_stability(
    results: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Apply RRF as a re-ranking stability pass on already-combined results.
    Input is a pre-sorted list; RRF converts rank positions back to smooth scores.
    """
    return [
        (text, 1.0 / (k + rank))
        for rank, (text, _) in enumerate(results, start=1)
    ]


# ── BM25 collection search ────────────────────────────────────────────────────

def _bm25_search_all(
    selected: list[str], query: str
) -> dict[str, list[tuple[str, float]]]:
    """Run BM25 search on all selected collections independently."""
    per_col: dict[str, list[tuple[str, float]]] = {}
    for collection in selected:
        hits = bm25_manager.search(collection, query, top_k=BM25_TOP_K)
        per_col[collection] = hits
        if hits:
            logger.debug("BM25 '%s': %d hits (top=%.4f)", collection, len(hits), hits[0][1])
        else:
            logger.debug("BM25 '%s': no hits (index empty or no keyword overlap)", collection)
    return per_col


# ── Main fusion entry point ───────────────────────────────────────────────────

def fuse_all_collections(
    per_collection_semantic: dict[str, list[tuple[str, float]]],
    query: str,
) -> list[tuple[str, float]]:
    """
    Full two-stage fusion: Ensemble Combiner → RRF Stability Layer.

    Args:
        per_collection_semantic: {collection: [(text, semantic_score), ...]}
                                  Results from Qdrant vector search (post-threshold).
        query:                    Rewritten search query (used for BM25 search).

    Returns:
        [(text, final_score), ...] globally ranked and capped to FINAL_TOP_K.
    """
    selected = list(per_collection_semantic.keys())

    # ── BM25 independent search across all collections ───────────────────────
    per_collection_bm25 = _bm25_search_all(selected, query)

    total_bm25 = sum(len(h) for h in per_collection_bm25.values())
    total_sem = sum(len(h) for h in per_collection_semantic.values())
    logger.info(
        "Fusion input: semantic=%d docs, BM25=%d docs across %d collections",
        total_sem, total_bm25, len(selected),
    )

    # ── Stage 1: Weighted Ensemble Combination ───────────────────────────────
    ensemble_results = ensemble_combine_multi_collection(
        per_collection_semantic=per_collection_semantic,
        per_collection_bm25=per_collection_bm25,
        top_k=FINAL_TOP_K * 2,   # give RRF stage more candidates to work with
    )

    if not ensemble_results:
        # Both legs returned nothing
        logger.warning("Ensemble returned 0 results — both retrieval legs empty")
        return []

    # ── Stage 2: RRF Stability Layer ─────────────────────────────────────────
    final = _rrf_stability(ensemble_results)
    final.sort(key=lambda x: x[1], reverse=True)
    final = final[:FINAL_TOP_K]

    logger.info(
        "Fusion complete: %d final docs | top=%.6f | bottom=%.6f",
        len(final),
        final[0][1] if final else 0,
        final[-1][1] if final else 0,
    )
    return final
