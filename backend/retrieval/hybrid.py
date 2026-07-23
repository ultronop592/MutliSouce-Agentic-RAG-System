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
    selected: list[str], query: str, source_filename: str | None = None
) -> dict[str, list[tuple[str, float]]]:
    """Run BM25 search on all selected collections independently.
    When source_filename is set, the BM25 index now filters to only
    chunks from that specific file (using stored source_filename metadata).
    This prevents cross-document keyword bleed when multiple PDFs exist
    in the same collection.
    """
    per_col: dict[str, list[tuple[str, float]]] = {}
    for collection in selected:
        # Pass source_filename so bm25_manager.search() restricts results
        # to only chunks that belong to the active uploaded PDF.
        hits = bm25_manager.search(
            collection, query, top_k=BM25_TOP_K, source_filename=source_filename
        )
        per_col[collection] = hits
        if hits:
            logger.debug(
                "BM25 '%s': %d hits (top=%.4f, source_filter=%s)",
                collection, len(hits), hits[0][1], source_filename or "none",
            )
        else:
            logger.debug(
                "BM25 '%s': no hits (index empty, no keyword overlap, or source_filter='%s')",
                collection, source_filename or "none",
            )
    return per_col


# ── Main fusion entry point ───────────────────────────────────────────────────

def _jaccard(a: str, b: str, max_tokens: int = 200) -> float:
    """Fast Jaccard token overlap for near-duplicate detection.
    Operates on the first max_tokens words to keep it O(1) per pair.
    Returns 0.0–1.0. Values > 0.80 indicate near-identical chunks.
    """
    ta = set(a.lower().split()[:max_tokens])
    tb = set(b.lower().split()[:max_tokens])
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _deduplicate(
    docs: list[tuple[str, float]],
    jaccard_threshold: float = 0.80,
) -> list[tuple[str, float]]:
    """
    Remove near-duplicate chunks from the fused result list.

    When multiple PDFs or overlapping chunks produce nearly identical text
    segments, sending them all to the LLM wastes context window and can
    cause the model to focus on repeated content at the expense of diversity.

    Algorithm: greedy O(n²) over the small FINAL_TOP_K list.
    For each chunk, compare Jaccard similarity against all already-kept chunks.
    If similarity > threshold, drop it (the higher-ranked version is already kept).
    """
    kept: list[tuple[str, float]] = []
    for text, score in docs:
        is_dup = any(_jaccard(text, kept_text) > jaccard_threshold for kept_text, _ in kept)
        if not is_dup:
            kept.append((text, score))
    return kept


def fuse_all_collections(
    per_collection_semantic: dict[str, list[tuple[str, float]]],
    query: str,
    source_filename: str | None = None,
) -> list[tuple[str, float]]:
    """
    Full three-stage fusion: Ensemble Combiner → RRF Stability → Deduplication.

    Args:
        per_collection_semantic: {collection: [(text, semantic_score), ...]}\
                                  Results from Qdrant vector search (post-threshold).
        query:                    Rewritten search query (used for BM25 search).
        source_filename:          Optional: passed to BM25 search for logging/context.

    Returns:
        [(text, final_score), ...] globally ranked, deduplicated, capped to FINAL_TOP_K.
    """
    selected = list(per_collection_semantic.keys())

    # ── BM25 independent search across all collections ───────────────────────
    per_collection_bm25 = _bm25_search_all(selected, query, source_filename=source_filename)

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
        top_k=FINAL_TOP_K * 3,   # fetch extra candidates so dedup doesn't shrink below FINAL_TOP_K
    )

    if not ensemble_results:
        # Both legs returned nothing
        logger.warning("Ensemble returned 0 results — both retrieval legs empty")
        return []

    # ── Stage 2: RRF Stability Layer ─────────────────────────────────────────
    final = _rrf_stability(ensemble_results)
    final.sort(key=lambda x: x[1], reverse=True)

    # ── Stage 3: Near-duplicate deduplication ─────────────────────────────────
    # Chunks with > 80% Jaccard token overlap are collapsed to the highest-ranked one.
    # This prevents the LLM context window from being saturated with repeated content
    # (common when multiple PDFs or overlapping chunk windows land in the same result set).
    before_dedup = len(final)
    final = _deduplicate(final, jaccard_threshold=0.80)
    final = final[:FINAL_TOP_K]
    if len(final) < before_dedup:
        logger.info(
            "Deduplication: removed %d near-duplicate chunks (%d → %d)",
            before_dedup - len(final), before_dedup, len(final),
        )

    logger.info(
        "Fusion complete: %d final docs | top=%.6f | bottom=%.6f",
        len(final),
        final[0][1] if final else 0,
        final[-1][1] if final else 0,
    )
    return final
