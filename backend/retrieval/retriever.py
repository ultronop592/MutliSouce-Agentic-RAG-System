"""
Retriever — True Hybrid Search Orchestrator
--------------------------------------------
Coordinates the full retrieval pipeline:

  Step 1  Query rewriting    — Gemini rewrites the user question into a
                               cleaner search query for better recall.

  Step 2  Semantic search    — Parallel Qdrant vector (cosine) search
                               across all selected collections.
                               MIN_SCORE threshold discards low-relevance chunks
                               before they can poison the LLM context.

  Step 3  BM25 search        — Independent keyword search via BM25Okapi on the
                               FULL corpus stored in each collection's index.
                               This is NOT reranking the vector results — it is
                               a completely separate retrieval leg.

  Step 4  RRF Fusion         — Reciprocal Rank Fusion merges the two ranked
                               lists into one unified ranking.

  Step 5  Returns             (top_docs, docs_with_scores) for the caller:
                               - top_docs         → text list for the LLM prompt
                               - docs_with_scores → fused scores for the
                                                    hallucination guard
"""

import asyncio
import logging
from core.qdrant_client import qdrant
from core.embeddings import embeddings
from core.llm import llm
from retrieval.hybrid import fuse_all_collections, SEMANTIC_TOP_K

logger = logging.getLogger(__name__)

# ── Tunable constants ────────────────────────────────────────────────────────

# Cosine similarity threshold — chunks below this score are discarded BEFORE
# they reach the hybrid fusion stage.  This is the primary guard against
# "garbage-in → hallucination-out".
MIN_SCORE: float = 0.35

# How many vector candidates to pull per collection (before threshold filter)
VECTOR_FETCH_K: int = 12

COLLECTION_CONFIDENCE = {
    "research_papers": 1.0,
    "knowledge_base": 0.8,
    "code_docs": 1.2,
    "faq_data": 0.7,
}


# ── Query Rewriting ──────────────────────────────────────────────────────────

def rewrite_query(question: str) -> str:
    """Use Gemini to rewrite the user question into a better search query."""
    try:
        prompt = (
            "Rewrite the following question into ONE clear standalone search query.\n\n"
            "Return ONLY the rewritten query. Do NOT explain. Do NOT give options.\n\n"
            f"Question: {question}"
        )
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        if rewritten and len(rewritten) > 3:
            return rewritten
    except Exception:
        pass
    return question


# ── Semantic Search Leg ──────────────────────────────────────────────────────

async def _semantic_search_collection(
    collection: str, query_vector: list[float]
) -> list[tuple[str, float]]:
    """
    Vector search a single Qdrant collection.
    Applies MIN_SCORE threshold and collection confidence weighting.
    Returns [(text, weighted_score), ...].
    """
    try:
        results = qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            limit=VECTOR_FETCH_K,
        )
        confidence = COLLECTION_CONFIDENCE.get(collection, 1.0)
        hits = []
        for point in results.points:
            if "text" not in point.payload:
                continue
            if point.score < MIN_SCORE:
                # ── Score gate: low-relevance chunk discarded ──────────────
                continue
            hits.append((point.payload["text"], point.score * confidence))
        return hits
    except Exception as e:
        logger.warning("Semantic search failed for '%s': %s", collection, e)
        return []


# ── Main Entry Point ─────────────────────────────────────────────────────────

async def hybrid_retrieve(
    query: str, selected: list[str]
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query:    raw user question
        selected: list of Qdrant collection names to search

    Returns:
        (top_docs, docs_with_scores)
        - top_docs         : list[str]                — texts for LLM prompt
        - docs_with_scores : list[tuple[str, float]]  — for hallucination guard
    """
    # ── Step 1: Query rewriting ──────────────────────────────────────────────
    rewritten = rewrite_query(query)
    logger.debug("Query rewritten: '%s' → '%s'", query, rewritten)

    # ── Step 2: Single embedding, parallel semantic search ───────────────────
    query_vector = embeddings.embed_query(rewritten)

    tasks = [_semantic_search_collection(c, query_vector) for c in selected]
    semantic_results_per_col = await asyncio.gather(*tasks)

    # Map collection → semantic hits
    per_collection: dict[str, list[tuple[str, float]]] = {
        col: hits
        for col, hits in zip(selected, semantic_results_per_col)
    }

    total_semantic = sum(len(h) for h in per_collection.values())
    logger.debug(
        "Semantic search: %d docs across %d collections (post-threshold)",
        total_semantic, len(selected),
    )

    # ── Step 3 + 4: BM25 search + RRF fusion ────────────────────────────────
    # hybrid.py runs BM25 independently on each collection's full corpus,
    # then merges with RRF.
    fused: list[tuple[str, float]] = fuse_all_collections(per_collection, rewritten)

    if not fused:
        # Both legs returned nothing — return empty for guard to catch
        return [], []

    top_docs = [text for text, _ in fused]
    logger.debug("Hybrid retrieval final result: %d docs", len(top_docs))

    return top_docs, fused