"""
Orchestrator
------------
The master coordinator for the multi-agent RAG pipeline.

Pipeline paths (chosen by the Router Agent):

  Path A — CONVERSATIONAL (fastest)
  ─────────────────────────────────
  Router → Memory Agent (conversational_reply) → stream reply

  Path B — MEMORY HIT (very fast, hallucination-free)
  ────────────────────────────────────────────────────
  Router → Memory Agent (semantic lookup, doc_version-gated) → stream cached answer

  Path C — FULL RETRIEVAL (complete pipeline)
  ────────────────────────────────────────────
  Router
    └─→ Retrieval Agent   (planner + hybrid search: semantic + BM25 + RRF)
           └─→ Reranker Agent  (Gemini pointwise reranking)
                  └─→ Guard    (confidence assessment)
                         └─→ Answer Agent  (grounded response, streamed)

After every Path C response:
  - Conversation memory is updated (with embedding AND doc_version cached on the turn)
  - Semantic cache is updated (TTL = 1 hour, keyed by session + doc_version)

DOC VERSION FLOW:
  - doc_version is the MD5 fingerprint of the currently active PDF in this session.
  - It is passed by the frontend in every ChatRequest (populated after upload).
  - Cache and memory lookups ONLY hit if both session_id AND doc_version match.
  - This means a new PDF upload automatically forces full retrieval for any
    previously cached question — no "New Chat" button needed.

Every agent step is timed and logged so you can trace slow requests in production.
"""

import logging
import time
import asyncio
from typing import AsyncIterator

from agents.router import router_agent, RouteType
from agents.verification_agent import verification_agent
from agents.memory_agent import memory_agent
from agents.retrieval_agent import retrieval_agent
from agents.reranker import reranker_agent
from agents.answer_agent import answer_agent
from retrieval.guard import assess_confidence
from memory.memory import chat_memory
from cache.cache import response_cache
from core.embeddings import embeddings

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates all agents and owns session state management."""

    async def handle(
        self,
        query: str,
        session_id: str,
        collections: list[str] | None = None,
        doc_version: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Main entry point.  Returns an async generator of text chunks
        (compatible with FastAPI StreamingResponse).

        Args:
            query:       raw user question
            session_id:  unique session identifier
            collections: optional list of Qdrant collections to search
            doc_version: MD5 fingerprint of the active PDF for this session.
                         When provided, cache and memory hits are gated on this
                         version so a new PDF upload forces fresh retrieval.
        """
        t_start = time.perf_counter()
        logger.info(
            "=== Orchestrator START | session=%s | doc_version=%s | query='%s'",
            session_id, doc_version, query[:80],
        )

        # ── Step 1: Embed question (single call, reused by cache + memory) ──
        question_embedding: list[float] | None = None
        try:
            question_embedding = embeddings.embed_query(query)
        except Exception as e:
            logger.warning("Embedding failed: %s", e)

        # ── Step 2: Semantic cache check ────────────────────────────────────
        # CRITICAL: session_id AND doc_version are both required for a hit.
        # A new PDF upload changes doc_version → cache miss → forces retrieval.
        if question_embedding:
            cached = response_cache.get(session_id, question_embedding, doc_version)
            if cached:
                logger.info(
                    "Cache HIT (doc_version=%s) — returning in %.0fms",
                    doc_version, (time.perf_counter() - t_start) * 1000,
                )
                async def _cached_stream():
                    yield cached
                return _cached_stream()

        # ── Step 3: Route the query ──────────────────────────────────────────
        route_result = router_agent.run(query)
        route: RouteType = route_result.data
        logger.info("Route: %s", route.value)

        # ── Path A: CONVERSATIONAL ───────────────────────────────────────────
        if route == RouteType.CONVERSATIONAL:
            reply = memory_agent.conversational_reply(query, session_id)
            # Conversational turns are stored WITHOUT doc_version — they are
            # not document-specific and should always be memory-eligible.
            chat_memory.update(session_id, query, reply, embedding=question_embedding, doc_version=None)
            async def _conv_stream():
                yield reply
            return _conv_stream()

        # ── Path B: MEMORY FIRST — check semantic memory ─────────────────────
        # doc_version is passed so only same-document turns are considered.
        if route == RouteType.MEMORY_FIRST and question_embedding:
            mem_result = memory_agent.run(session_id, question_embedding, doc_version)
            if mem_result.success and mem_result.data:
                logger.info(
                    "Memory HIT (sim=%.3f, doc_version=%s) — returning in %.0fms",
                    mem_result.metadata.get("similarity", 0),
                    doc_version,
                    (time.perf_counter() - t_start) * 1000,
                )
                answer = mem_result.data
                # Cache this re-served memory answer in the same session's cache
                if question_embedding:
                    response_cache.set(session_id, question_embedding, answer, doc_version)
                async def _mem_stream():
                    yield answer
                return _mem_stream()
            # Memory miss → fall through to full retrieval

        # ── Path C: FULL RETRIEVAL PIPELINE ─────────────────────────────────

        # Step C1: Hybrid retrieval (semantic + BM25 + RRF)
        retrieval_result = await retrieval_agent.run_async(query, collections=collections)
        top_docs: list[str] = retrieval_result.data.get("docs", [])
        docs_with_scores: list[tuple[str, float]] = retrieval_result.data.get("scored", [])

        # Step C2: Reranking — only if enough docs to meaningfully reorder
        # Reranking adds ~10s (one extra Gemini call). With <=4 docs the ensemble
        # ordering is already high quality, so skip it to save latency.
        RERANK_MIN_DOCS = 5
        if len(top_docs) >= RERANK_MIN_DOCS:
            rerank_result = reranker_agent.run(query, top_docs)
            reranked_docs: list[str] = rerank_result.data if rerank_result.success else top_docs
            logger.info("Reranker: applied (%d docs)", len(reranked_docs))
        else:
            reranked_docs = top_docs
            logger.info("Reranker: skipped (%d docs, threshold=%d) — using ensemble order",
                        len(top_docs), RERANK_MIN_DOCS)

        # Step C3: Hallucination guard (uses original scores for confidence)
        confidence_level, advisory_note = assess_confidence(docs_with_scores)
        logger.info("Guard confidence: %s", confidence_level)

        # Step C4: Build context string from reranked docs
        if reranked_docs:
            context = "\n\n---\n\n".join(reranked_docs)
        else:
            context = "No relevant documents found."

        # Step C5: Conversation memory for context
        memory_str = chat_memory.format_history(session_id)

        # Step C6: Verified Answer Generation & Self-Correction Loop
        full_response = ""

        async def _retrieval_stream() -> AsyncIterator[str]:
            nonlocal full_response
            
            # Generate initial answer candidate in memory
            answer_candidate = await answer_agent.generate(query, context, memory_str, advisory_note)
            
            # Run verification loop
            max_retries = 2
            attempts = 0
            feedback = ""
            verified = False
            
            while attempts < max_retries:
                attempts += 1
                logger.info("VerificationAgent: check attempt %d for query='%s'", attempts, query[:50])
                
                check = await verification_agent.verify_async(answer_candidate, context)
                status = check.get("status")
                
                if status == "verified":
                    verified = True
                    logger.info("VerificationAgent: check PASSED on attempt %d", attempts)
                    break
                elif status == "failed":
                    feedback = check.get("reason", "")
                    logger.warning(
                        "VerificationAgent: check FAILED on attempt %d: %s. Triggering self-correction...",
                        attempts, feedback,
                    )
                    # Regenerate with correction feedback passed to the LLM
                    answer_candidate = await answer_agent.generate(
                        query, context, memory_str, advisory_note, feedback=feedback
                    )
                else:
                    # Error state (e.g. network timeout or API error) — proceed to avoid blocking user flow
                    logger.warning(
                        "VerificationAgent: check errored: %s. Bypassing verification.",
                        check.get("reason"),
                    )
                    verified = True
                    break
            
            if not verified:
                logger.error("VerificationAgent: check failed after max retries. Serving best candidate.")
            
            final_answer = answer_candidate
            full_response = final_answer

            # Stream the final verified answer progressively (typing effect)
            chunk_size = 12
            for i in range(0, len(final_answer), chunk_size):
                chunk = final_answer[i : i + chunk_size]
                yield chunk
                await asyncio.sleep(0.005)  # fast, smooth progressive typing effect
                
            # Post-stream: update memory and cache WITH doc_version
            # This ensures future memory/cache hits are gated on the same document.
            if full_response and "[Error" not in full_response:
                chat_memory.update(
                    session_id, query, full_response,
                    embedding=question_embedding,
                    doc_version=doc_version,   # ← tag this answer to the current PDF
                )
                if question_embedding:
                    response_cache.set(
                        session_id, question_embedding, full_response,
                        doc_version=doc_version,   # ← cache hit requires same PDF
                    )
            
            total_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "=== Orchestrator DONE | session=%s | doc_version=%s | %.0fms | docs=%d | confidence=%s | verified=%s",
                session_id, doc_version, total_ms, len(reranked_docs),
                confidence_level, "yes" if verified else "no",
            )

        return _retrieval_stream()


# Module-level singleton
orchestrator = Orchestrator()
