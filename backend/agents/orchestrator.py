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
  Router → Memory Agent (semantic lookup) → stream cached answer

  Path C — FULL RETRIEVAL (complete pipeline)
  ────────────────────────────────────────────
  Router
    └─→ Retrieval Agent   (planner + hybrid search: semantic + BM25 + RRF)
           └─→ Reranker Agent  (Gemini pointwise reranking)
                  └─→ Guard    (confidence assessment)
                         └─→ Answer Agent  (grounded response, streamed)

After every Path C response:
  - Conversation memory is updated (with embedding cached on the turn)
  - Semantic cache is updated (TTL = 1 hour)

Every agent step is timed and logged so you can trace slow requests in production.
"""

import logging
import time
from typing import AsyncIterator

from agents.router import router_agent, RouteType
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
    ) -> AsyncIterator[str]:
        """
        Main entry point.  Returns an async generator of text chunks
        (compatible with FastAPI StreamingResponse).
        """
        t_start = time.perf_counter()
        logger.info("=== Orchestrator START | session=%s | query='%s'", session_id, query[:80])

        # ── Step 1: Embed question (single call, reused by cache + memory) ──
        question_embedding: list[float] | None = None
        try:
            question_embedding = embeddings.embed_query(query)
        except Exception as e:
            logger.warning("Embedding failed: %s", e)

        # ── Step 2: Semantic cache check ────────────────────────────────────
        if question_embedding:
            cached = response_cache.get(question_embedding)
            if cached:
                logger.info("Cache HIT — returning in %.0fms", (time.perf_counter() - t_start) * 1000)
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
            chat_memory.update(session_id, query, reply, embedding=question_embedding)
            async def _conv_stream():
                yield reply
            return _conv_stream()

        # ── Path B: MEMORY FIRST — check semantic memory ─────────────────────
        if route == RouteType.MEMORY_FIRST and question_embedding:
            mem_result = memory_agent.run(session_id, question_embedding)
            if mem_result.success and mem_result.data:
                logger.info(
                    "Memory HIT (sim=%.3f) — returning in %.0fms",
                    mem_result.metadata.get("similarity", 0),
                    (time.perf_counter() - t_start) * 1000,
                )
                answer = mem_result.data
                # Cache this re-served memory answer too
                if question_embedding:
                    response_cache.set(question_embedding, answer)
                async def _mem_stream():
                    yield answer
                return _mem_stream()
            # Memory miss → fall through to full retrieval

        # ── Path C: FULL RETRIEVAL PIPELINE ─────────────────────────────────

        # Step C1: Hybrid retrieval (semantic + BM25 + RRF)
        retrieval_result = await retrieval_agent.run_async(query)
        top_docs: list[str] = retrieval_result.data.get("docs", [])
        docs_with_scores: list[tuple[str, float]] = retrieval_result.data.get("scored", [])

        # Step C2: Reranking (Gemini pointwise scoring)
        rerank_result = reranker_agent.run(query, top_docs)
        reranked_docs: list[str] = rerank_result.data if rerank_result.success else top_docs

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

        # Step C6: Stream answer
        full_response = ""

        async def _retrieval_stream() -> AsyncIterator[str]:
            nonlocal full_response
            async for chunk in answer_agent.stream(query, context, memory_str, advisory_note):
                full_response += chunk
                yield chunk
            # Post-stream: update memory and cache
            if full_response and "[Error" not in full_response:
                chat_memory.update(
                    session_id, query, full_response,
                    embedding=question_embedding,
                )
                if question_embedding:
                    response_cache.set(question_embedding, full_response)
            total_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "=== Orchestrator DONE | session=%s | %.0fms | docs=%d | confidence=%s",
                session_id, total_ms, len(reranked_docs), confidence_level,
            )

        return _retrieval_stream()


# Module-level singleton
orchestrator = Orchestrator()
