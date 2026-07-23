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
# Raised from 5 -> 8: queries of 8+ words are almost always already well-formed
# information-seeking questions and rarely benefit from rewriting.
_REWRITE_MIN_WORDS: int = 8

# Vague reference words that ALWAYS trigger a rewrite regardless of length.
# These indicate the user is gesturing at a document rather than asking a specific question.
_VAGUE_TRIGGERS = {"pdf", "document", "file", "this", "that", "it", "about"}

# Patterns that indicate the query is already specific enough to skip rewriting.
# e.g. "What is the BLEU score?" / "How does attention mechanism work?" — already clear.
_SPECIFIC_STARTERS = {
    "what is", "what are", "how does", "how do", "how to",
    "explain", "define", "describe", "list", "compare",
    "why is", "why does", "when was", "where is", "who is",
}


def _is_already_specific(question: str) -> bool:
    """
    Heuristic check to skip the rewrite LLM call for queries that are already
    well-formed. Returns True if the query starts with a specific interrogative
    or action phrase (no need for Gemini to restructure it).

    Saves ~1-2s on ~40-60% of real queries.
    """
    q_lower = question.lower().strip()
    # Starts with a well-known information-seeking opener
    if any(q_lower.startswith(s) for s in _SPECIFIC_STARTERS):
        return True
    # Contains a quoted term (user already knows exactly what they're looking for)
    if '"' in question or "'" in question:
        return True
    # Contains a number/version (highly specific technical query)
    import re
    if re.search(r'\b\d+\b', question):
        return True
    return False


def rewrite_query(question: str) -> str:
    """Use Gemini to rewrite the user question into a better search query.

    Bypass logic (saves one LLM call per request when applicable):
      - Short queries (< 8 words) with no vague references -> skip
      - Already-specific queries (starts with what/how/explain, contains quotes,
        contains numbers) -> skip
      - All other queries go through the Gemini rewrite step
    """
    words = question.lower().split()
    has_vague = bool(set(words) & _VAGUE_TRIGGERS)
    too_short = len(words) < _REWRITE_MIN_WORDS

    # Skip rewrite: short AND specific (no vague references)
    if too_short and not has_vague:
        logger.debug("Query rewrite skipped (short+specific): '%s'", question[:60])
        return question

    # Skip rewrite: query is already well-formed even if longer
    if not has_vague and _is_already_specific(question):
        logger.debug("Query rewrite skipped (already specific): '%s'", question[:60])
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
    question_embedding: list[float] | None = None,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query:              raw user question
        selected:           list of Qdrant collection names to search
        source_filename:    optional PDF basename to restrict retrieval to.
                            When set, both Qdrant vector search and BM25 results
                            are filtered to only chunks from this specific file.
        question_embedding: optional pre-computed embedding of the original query
                            from the orchestrator (O3 optimization).
                            When the query rewriter returns an UNCHANGED query, this
                            embedding is reused directly — saving one embed_query()
                            API call (~300-600ms). When the query IS rewritten, a new
                            embedding is computed for the rewritten form.

    Returns:
        (top_docs, docs_with_scores)
        - top_docs         : list[str]                -- texts for LLM prompt
        - docs_with_scores : list[tuple[str, float]]  -- for hallucination guard
    """
    # ── Step 1: Query rewriting ──────────────────────────────────────────────
    rewritten = rewrite_query(query)
    logger.debug("Query rewritten: '%s' -> '%s'", query, rewritten)

    # ── Step 2: Embedding — reuse if query unchanged (O3) ────────────────────
    query_unchanged = (rewritten.strip().lower() == query.strip().lower())
    if query_unchanged and question_embedding is not None:
        # O3: reuse the pre-computed embedding from the orchestrator.
        # This saves one embed_query() API call when the rewriter makes no change.
        query_vector = question_embedding
        logger.info("O3: Embedding REUSED from orchestrator (query unchanged)")
    else:
        # Query was rewritten — must embed the new form for accurate vector search
        query_vector = embeddings.embed_query(rewritten)
        if question_embedding is not None:
            logger.info("O3: Embedding RE-COMPUTED (query was rewritten: '%s')", rewritten[:60])

    # ── Step 3: Parallel Qdrant semantic search across all collections (O4) ───
    # asyncio.gather fires all collection queries simultaneously.
    # For universal tab with 5 collections this reduces Qdrant latency from
    # 5x sequential RTT to 1x RTT — typically saves 400-800ms.
    tasks = [
        _semantic_search_collection(c, query_vector, source_filename=source_filename)
        for c in selected
    ]
    semantic_results_per_col = await asyncio.gather(*tasks)

    # Map collection -> semantic hits
    per_collection: dict[str, list[tuple[str, float]]] = {
        col: hits
        for col, hits in zip(selected, semantic_results_per_col)
    }

    total_semantic = sum(len(h) for h in per_collection.values())
    logger.debug(
        "Semantic search: %d docs across %d collections (post-threshold, source_filter=%s)",
        total_semantic, len(selected), source_filename or "none",
    )

    # ── Step 4+5: BM25 search + Ensemble + RRF fusion + Deduplication ────────
    fused: list[tuple[str, float]] = fuse_all_collections(
        per_collection, rewritten, source_filename=source_filename
    )

    if not fused:
        return [], []

    top_docs = [text for text, _ in fused]

    # ── Guard scores: use RAW COSINE similarities, NOT fused RRF scores ────────
    # RRF stability scores are always ~0.015-0.016 regardless of relevance.
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