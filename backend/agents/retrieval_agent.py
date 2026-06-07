"""
Retrieval Agent
---------------
Wraps the full hybrid search pipeline into a single agent interface:
  1. Planner     — selects which Qdrant collections to search
  2. hybrid_retrieve — semantic search + BM25 search + RRF fusion
  3. Returns (top_docs, docs_with_scores) for the reranker and guard
"""

import asyncio
import logging
from agents.base import BaseAgent, AgentResult
from retrieval.planner import planner
from retrieval.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    name = "RetrievalAgent"

    def _run(self, query: str) -> AgentResult:  # type: ignore[override]
        # Planner — fast keyword routing (no API call)
        collections = planner(query)
        logger.info("RetrievalAgent routing to: %s", collections)

        # Run async hybrid retrieval synchronously inside the agent wrapper.
        # (The orchestrator is async, so it calls this via await run_async.)
        loop = asyncio.get_event_loop()
        top_docs, docs_with_scores = loop.run_until_complete(
            hybrid_retrieve(query, collections)
        )

        logger.info(
            "RetrievalAgent: %d docs retrieved from %s",
            len(top_docs), collections,
        )
        return AgentResult(
            agent=self.name,
            success=True,
            data={"docs": top_docs, "scored": docs_with_scores},
            latency_ms=0.0,
            metadata={"collections": collections, "doc_count": len(top_docs)},
        )

    async def run_async(self, query: str, collections: list[str] | None = None) -> AgentResult:
        """Async version — called by the orchestrator directly."""
        import time
        t0 = time.perf_counter()
        try:
            if collections is None:
                collections = planner(query)
            logger.info("RetrievalAgent routing to: %s", collections)
            top_docs, docs_with_scores = await hybrid_retrieve(query, collections)
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                "RetrievalAgent: %d docs in %.0fms from %s",
                len(top_docs), latency, collections,
            )
            return AgentResult(
                agent=self.name,
                success=True,
                data={"docs": top_docs, "scored": docs_with_scores},
                latency_ms=latency,
                metadata={"collections": collections, "doc_count": len(top_docs)},
            )
        except Exception as exc:
            import time
            latency = (time.perf_counter() - t0) * 1000
            logger.error("RetrievalAgent failed: %s", exc)
            return AgentResult(
                agent=self.name,
                success=False,
                data={"docs": [], "scored": []},
                latency_ms=latency,
                metadata={"error": str(exc)},
            )


# Module-level singleton
retrieval_agent = RetrievalAgent()
