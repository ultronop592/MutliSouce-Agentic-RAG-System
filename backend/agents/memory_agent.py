"""
Memory Agent
------------
Provides two capabilities:

  1. Fast semantic lookup  — Searches the session's recent conversation history
                             using cosine similarity on cached question embeddings.
                             If a past turn answers the current question (similarity
                             ≥ MEMORY_HIT_THRESHOLD), returns the stored answer
                             immediately — no Qdrant call, no LLM call.
                             This is the fastest, most hallucination-free path.

  2. Conversational reply  — Generates a brief, friendly response for
                             CONVERSATIONAL queries (greetings, thanks, etc.)
                             using the LLM with minimal context.

Why this prevents hallucinations:
  Memory answers come from VERIFIED past responses that were already grounded.
  A repeated or follow-up question gets the same grounded answer, not a
  freshly hallucinated one from the LLM's general knowledge.
"""

import math
import logging
from agents.base import BaseAgent, AgentResult
from memory.memory import chat_memory
from core.llm import llm

logger = logging.getLogger(__name__)

# Cosine similarity threshold for a memory cache hit.
# CRITICAL FIX: Raised from 0.88 → 0.94.
# 0.88 was too permissive — generic questions like "Summarize this document"
# and "What are the key findings?" both score ~0.90 similarity even when
# asked about completely different PDFs. At 0.94 only near-identical
# rephrasing of the EXACT same question triggers a memory hit.
MEMORY_HIT_THRESHOLD: float = 0.94

# How many recent turns to scan (scanning all turns is unnecessary and slow)
# Keep at 4: a smaller window means less chance of a stale answer from an
# earlier document bleeding into a newer query in the same session.
RECENT_TURNS_WINDOW: int = 4


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / mag if mag else 0.0


class MemoryAgent(BaseAgent):
    name = "MemoryAgent"

    # ── 1. Semantic memory lookup ────────────────────────────────────────────

    def _run(  # type: ignore[override]
        self,
        session_id: str,
        query_embedding: list[float],
    ) -> AgentResult:
        """
        Search the last RECENT_TURNS_WINDOW turns for a semantically similar
        question. Returns the stored answer if found above threshold, else None.
        """
        history = chat_memory.get_recent(session_id, RECENT_TURNS_WINDOW)
        if not history:
            return AgentResult(agent=self.name, success=False, data=None, latency_ms=0.0)

        best_sim: float = 0.0
        best_answer: str | None = None
        best_question: str = ""

        for turn in history:
            cached_emb = turn.get("embedding")
            if not cached_emb:
                continue  # embedding not cached for this turn — skip
            sim = _cosine(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_answer = turn["assistant"]
                best_question = turn["user"]

        if best_sim >= MEMORY_HIT_THRESHOLD and best_answer:
            logger.info(
                "MemoryAgent HIT (sim=%.3f): '%s' → reusing answer",
                best_sim, best_question[:60],
            )
            return AgentResult(
                agent=self.name,
                success=True,
                data=best_answer,
                latency_ms=0.0,
                metadata={"similarity": best_sim, "matched_question": best_question},
            )

        logger.debug("MemoryAgent MISS (best_sim=%.3f)", best_sim)
        return AgentResult(agent=self.name, success=False, data=None, latency_ms=0.0)

    # ── 2. Conversational reply ──────────────────────────────────────────────

    def conversational_reply(self, query: str, session_id: str) -> str:
        """
        Generate a brief, friendly reply for greetings / thanks / ack queries.
        Uses conversation memory for context so it feels natural.
        """
        memory_str = chat_memory.format_history(session_id)
        prompt = (
            "You are a friendly and helpful AI assistant.\n"
            "Reply naturally and briefly to the user's message.\n"
            "Do NOT use markdown formatting. Keep it to 1-2 sentences.\n\n"
            f"Previous conversation:\n{memory_str}\n"
            f"User: {query}\nAssistant:"
        )
        try:
            resp = llm.invoke(prompt)
            return resp.content.strip()
        except Exception as e:
            logger.warning("Conversational reply failed: %s", e)
            return "Hello! How can I help you today?"


# Module-level singleton
memory_agent = MemoryAgent()
