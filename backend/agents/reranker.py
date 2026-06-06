"""
Reranker Agent
--------------
Cross-encoder style reranking using Gemini as the scoring model.

Why reranking matters:
  After hybrid search (semantic + BM25 + RRF fusion), the top-k documents are
  still ranked by vector/keyword signals — they don't know the EXACT question.
  A reranker sees (query, document) pairs together and scores true relevance,
  catching cases like:
    - A document that is conceptually similar but answers a DIFFERENT aspect
    - A document that matches keywords but is not actually useful for the query
    - The #3 document that actually contains the best answer

Approach — Pointwise LLM reranking:
  We send all retrieved documents to Gemini in a SINGLE prompt, ask for
  relevance scores (0.0 – 1.0) and reorder by those scores.
  This avoids N separate API calls and is fast enough for up to 10 docs.

Fallback:
  If the LLM call fails or returns unparseable output, the original order
  (from RRF fusion) is preserved — no disruption to the pipeline.
"""

import json
import logging
import re
from agents.base import BaseAgent, AgentResult
from core.llm import llm

logger = logging.getLogger(__name__)

# Snippet length sent to Gemini per document (controls prompt size & cost)
SNIPPET_LEN: int = 400


class RerankerAgent(BaseAgent):
    name = "RerankerAgent"

    def _run(  # type: ignore[override]
        self, query: str, docs: list[str]
    ) -> AgentResult:
        if len(docs) <= 1:
            return AgentResult(
                agent=self.name, success=True, data=docs, latency_ms=0.0,
                metadata={"reranked": False, "reason": "single_or_empty"},
            )

        reranked = self._rerank(query, docs)
        return AgentResult(
            agent=self.name,
            success=True,
            data=reranked,
            latency_ms=0.0,
            metadata={"original_count": len(docs), "reranked_count": len(reranked)},
        )

    def _rerank(self, query: str, docs: list[str]) -> list[str]:
        """Score each document against the query and return reordered docs."""
        # Build a numbered list of document snippets
        numbered = "\n\n".join(
            f"[{i + 1}] {doc[:SNIPPET_LEN]}" for i, doc in enumerate(docs)
        )

        prompt = f"""You are a relevance scoring expert for a retrieval-augmented system.

Query: {query}

Score each document chunk below for relevance to the query.
Use a decimal score between 0.0 (completely irrelevant) and 1.0 (perfectly answers the query).

Documents:
{numbered}

Return ONLY a valid JSON array of {len(docs)} decimal scores, one per document.
Example for 3 documents: [0.9, 0.2, 0.7]
Your scores:"""

        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()

            # Extract JSON array from the response (handles extra text / markdown)
            match = re.search(r"\[[\d\s.,]+\]", raw)
            if not match:
                logger.warning("RerankerAgent: could not parse scores from: %s", raw[:200])
                return docs  # fallback: original order

            scores = json.loads(match.group())

            if len(scores) != len(docs):
                logger.warning(
                    "RerankerAgent: score count mismatch (%d scores for %d docs)",
                    len(scores), len(docs),
                )
                return docs

            # Pair, sort by score descending, unpack
            paired = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            reranked = [doc for doc, _ in paired]

            logger.info(
                "RerankerAgent: reranked %d docs | top score=%.2f | bottom=%.2f",
                len(docs), scores[0] if scores else 0, scores[-1] if scores else 0,
            )
            return reranked

        except Exception as e:
            logger.warning("RerankerAgent failed, keeping original order: %s", e)
            return docs  # safe fallback


# Module-level singleton
reranker_agent = RerankerAgent()
