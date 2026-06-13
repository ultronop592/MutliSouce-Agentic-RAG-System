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

import os
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
# Raised from 0.20 → 0.35 to prevent near-irrelevant chunks from poisoning the
# LLM context with garbage that triggers hallucinations.
# Real on-topic chunks score 0.55+ even for vague queries; 0.35 filters
# low-relevance noise while still allowing broad/general questions through.
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

# Minimum word count before we bother calling Gemini to rewrite.
# Short queries like "what is RAG?" are already clear — no rewrite needed.
_REWRITE_MIN_WORDS: int = 5

# Vague reference words that always trigger a rewrite regardless of length
_VAGUE_TRIGGERS = {"pdf", "document", "file", "this", "that", "it", "about"}


def rewrite_query(question: str) -> str:
    """Use Gemini to rewrite the user question into a better search query.

    Bypass logic: skip the rewrite (save one LLM call) when the query is
    already short and specific, unless it contains vague reference words.
    """
    words = question.lower().split()
    has_vague = bool(set(words) & _VAGUE_TRIGGERS)
    too_short = len(words) < _REWRITE_MIN_WORDS

    # Skip rewrite for short, specific queries (no vague words)
    if too_short and not has_vague:
        logger.debug("Query rewrite skipped (short+specific): '%s'", question[:60])
        return question

    try:
        prompt = (
            "Your task: rewrite the user's question into ONE clear, specific, "
            "standalone search query optimized for vector similarity search.\n\n"
            "Rules:\n"
            "- If the question is vague (e.g. 'tell me about the pdf', 'what is this about'), "
            "expand it into a more specific information-seeking query.\n"
            "- If it references 'the pdf', 'the document', 'the file', treat it as asking "
            "for a general overview or summary of the document content.\n"
            "- Remove filler words. Keep domain-specific terms.\n"
            "- Return ONLY the rewritten query. No explanation.\n\n"
            f"Question: {question}\n"
            "Rewritten search query:"
        )
        response = llm.invoke(prompt)
        rewritten = response.content.strip().strip('"\'')
        if rewritten and len(rewritten) > 3:
            logger.info("Query rewritten: '%s' -> '%s'", question[:60], rewritten[:60])
            return rewritten
    except Exception as e:
        logger.warning("Query rewrite failed: %s", e)
    return question


# ── Semantic Search Leg ──────────────────────────────────────────────────────

async def _semantic_search_collection(
    collection: str,
    query_vector: list[float],
    source_filename: str | None = None,
) -> list[tuple[str, float]]:
    """
    Vector search a single Qdrant collection.
    Applies MIN_SCORE threshold and collection confidence weighting.
    Returns [(text_with_source_header, weighted_score), ...].

    source_filename: when provided, adds a Qdrant payload filter so ONLY
    chunks whose source_file path contains this filename are returned.
    This is the primary guard against cross-document chunk bleed when
    multiple PDFs have been uploaded to the same collection.

    Each chunk is prefixed with a [Source: filename | Page: N] header so the
    LLM can distinguish content from different uploaded PDFs.  Without this
    header the LLM blends chunks from unrelated documents and hallucinates.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        # Build payload filter for source_filename when a specific file is active.
        # Uses the 'source_filename' payload field (bare basename stored during ingestion)
        # with MatchValue for exact matching — more reliable than MatchText on full paths.
        query_filter = None
        if source_filename:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_filename",
                        match=MatchValue(value=source_filename),
                    )
                ]
            )
            logger.debug(
                "Semantic search: filtering to source_filename='%s'",
                source_filename,
            )

        results = qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            limit=VECTOR_FETCH_K,
            query_filter=query_filter,
        )
        confidence = COLLECTION_CONFIDENCE.get(collection, 1.0)
        hits = []
        for point in results.points:
            if "text" not in point.payload:
                continue
            if point.score < MIN_SCORE:
                # ── Score gate: low-relevance chunk discarded ──────────────
                continue

            # ── Build source header ───────────────────────────────────────
            raw_path = point.payload.get("source_file", "")
            filename = os.path.basename(raw_path) if raw_path else "unknown"
            page = point.payload.get("page", 0)
            # Human-readable: page stored 0-indexed by PyPDFLoader → show +1
            page_label = int(page) + 1 if isinstance(page, (int, float)) else page
            source_header = f"[Source: {filename} | Page: {page_label}]"

            chunk_text = f"{source_header}\n{point.payload['text']}"
            hits.append((chunk_text, point.score * confidence))
        return hits
    except Exception as e:
        logger.warning("Semantic search failed for '%s': %s", collection, e)
        return []


# ── Main Entry Point ─────────────────────────────────────────────────────────

async def hybrid_retrieve(
    query: str,
    selected: list[str],
    source_filename: str | None = None,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query:           raw user question
        selected:        list of Qdrant collection names to search
        source_filename: optional PDF basename to restrict retrieval to.
                         When set, both Qdrant vector search and BM25 results
                         are filtered to only chunks from this specific file.
                         This prevents the Universal tab from mixing chunks
                         from multiple PDFs that share the same collection.

    Returns:
        (top_docs, docs_with_scores)
        - top_docs         : list[str]                — texts for LLM prompt
        - docs_with_scores : list[tuple[str, float]]  — for hallucination guard
    """
    # ── Step 1: Query rewriting ──────────────────────────────────────────────
    rewritten = rewrite_query(query)
    logger.debug("Query rewritten: '%s' → '%s'", query, rewritten)

    # ── Step 2: Single embedding, parallel semantic search ───────────────────────
    query_vector = embeddings.embed_query(rewritten)

    # Pass source_filename to each collection's semantic search
    tasks = [
        _semantic_search_collection(c, query_vector, source_filename=source_filename)
        for c in selected
    ]
    semantic_results_per_col = await asyncio.gather(*tasks)

    # Map collection → semantic hits
    per_collection: dict[str, list[tuple[str, float]]] = {
        col: hits
        for col, hits in zip(selected, semantic_results_per_col)
    }

    total_semantic = sum(len(h) for h in per_collection.values())
    logger.debug(
        "Semantic search: %d docs across %d collections (post-threshold, source_filter=%s)",
        total_semantic, len(selected), source_filename or "none",
    )

    # ── Step 3 + 4: BM25 search + Ensemble + RRF fusion ───────────────────────
    fused: list[tuple[str, float]] = fuse_all_collections(
        per_collection, rewritten, source_filename=source_filename
    )

    if not fused:
        return [], []

    top_docs = [text for text, _ in fused]

    # ── Guard scores: use RAW COSINE similarities, NOT fused RRF scores ────────
    # RRF stability scores are always ~0.015–0.016 regardless of relevance.
    # The guard needs REAL relevance signals (cosine 0-1) to correctly classify
    # HIGH vs LOW vs NONE confidence.
    all_semantic: list[tuple[str, float]] = []
    for hits in per_collection.values():
        all_semantic.extend(hits)
    all_semantic.sort(key=lambda x: x[1], reverse=True)
    guard_scores = all_semantic[:len(top_docs)] or fused  # fallback to fused if empty

    logger.info(
        "Retrieval: %d fused docs | guard scores: top=%.3f avg=%.3f",
        len(top_docs),
        guard_scores[0][1] if guard_scores else 0,
        sum(s for _, s in guard_scores) / len(guard_scores) if guard_scores else 0,
    )
    return top_docs, guard_scores