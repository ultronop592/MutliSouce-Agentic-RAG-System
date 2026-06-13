"""
Memory Agent
------------
Provides two capabilities:

  1. Fast semantic lookup  — Searches the session's recent conversation history
                             using cosine similarity on cached question embeddings.
                             If a past turn answers the current question (similarity
                             ≥ MEMORY_HIT_THRESHOLD) AND the turn was created for
                             the SAME document version, returns the stored answer
                             immediately — no Qdrant call, no LLM call.
                             This is the fastest, most hallucination-free path.

  2. Conversational reply  — Generates a brief, friendly response for
                             CONVERSATIONAL queries (greetings, thanks, etc.)
                             using the LLM with minimal context.

Why doc_version matters:
  Without it, asking "What are the key topics?" with PDF-A cached, then uploading
  PDF-B and asking the same question, would return PDF-A's answer from memory.
  With doc_version filtering, only turns from the SAME document version match.
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
        doc_version: str | None = None,
    ) -> AgentResult:
        """
        Search the last RECENT_TURNS_WINDOW turns for a semantically similar
        question.

        A memory hit requires BOTH:
          - cosine similarity ≥ MEMORY_HIT_THRESHOLD
          - turn.doc_version == current doc_version (same document)

        If doc_version is None (no document uploaded yet), only matches turns
        that also have no doc_version (general conversational turns).

        Returns the stored answer if found, else AgentResult(success=False).
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

            # ── Document version gate ────────────────────────────────────────
            # Only reuse an answer that was generated for the SAME document.
            # If either version is None, require BOTH to be None for a match.
            turn_doc_ver = turn.get("doc_version")
            if doc_version != turn_doc_ver:
                logger.debug(
                    "MemoryAgent: skipping turn (doc_version mismatch: current=%s, turn=%s)",
                    doc_version, turn_doc_ver,
                )
                continue  # different document — NEVER reuse this answer

            sim = _cosine(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_answer = turn["assistant"]
                best_question = turn["user"]

        if best_sim >= MEMORY_HIT_THRESHOLD and best_answer:
            logger.info(
                "MemoryAgent HIT (sim=%.3f, doc_version=%s): '%s' → reusing answer",
                best_sim, doc_version, best_question[:60],
            )
            return AgentResult(
                agent=self.name,
                success=True,
                data=best_answer,
                latency_ms=0.0,
                metadata={"similarity": best_sim, "matched_question": best_question},
            )

        logger.debug("MemoryAgent MISS (best_sim=%.3f, doc_version=%s)", best_sim, doc_version)
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
