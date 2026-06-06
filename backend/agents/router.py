"""
Router Agent
------------
Classifies every incoming query into one of three route types so the
orchestrator can pick the fastest, most appropriate pipeline:

  CONVERSATIONAL  →  Direct LLM reply (no retrieval needed).
                     Greetings, thanks, "who are you", clarifications.

  MEMORY_FIRST    →  Check conversation memory before retrieval.
                     Follow-up questions, pronoun references ("it", "that"),
                     or questions clearly related to the chat so far.

  RETRIEVAL       →  Full retrieval pipeline required.
                     Factual questions, research topics, technical queries.

Uses fast heuristics (no extra API call) so it adds near-zero latency.
Falls back to RETRIEVAL when uncertain — never blocks the pipeline.
"""

from enum import Enum
from agents.base import BaseAgent, AgentResult


class RouteType(str, Enum):
    CONVERSATIONAL = "conversational"
    MEMORY_FIRST   = "memory_first"
    RETRIEVAL      = "retrieval"


# ── Keyword sets ─────────────────────────────────────────────────────────────

_CONVERSATIONAL_TRIGGERS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "who are you", "what are you", "what can you do", "help me",
    "nice", "great", "ok", "okay", "sure", "got it", "understood",
    "good morning", "good evening", "good afternoon",
}

_MEMORY_TRIGGERS = {
    # Pronouns / references that imply a prior context
    "it", "that", "this", "those", "these", "they", "them",
    # Explicit follow-up phrases
    "you said", "you mentioned", "earlier", "previously", "as you said",
    "follow up", "follow-up", "more about", "tell me more",
    "elaborate", "expand on", "what about", "and also", "also tell",
    "continue", "go on", "furthermore", "in addition",
}


class RouterAgent(BaseAgent):
    name = "RouterAgent"

    def _run(self, query: str) -> AgentResult:  # type: ignore[override]
        route = self._classify(query)
        return AgentResult(
            agent=self.name,
            success=True,
            data=route,
            latency_ms=0.0,
            metadata={"query_length": len(query)},
        )

    def _classify(self, query: str) -> RouteType:
        q = query.lower().strip()
        words = set(q.split())

        # Very short queries that are greetings / ack
        if len(q) <= 20 and (words & _CONVERSATIONAL_TRIGGERS or q in _CONVERSATIONAL_TRIGGERS):
            return RouteType.CONVERSATIONAL

        # Follow-up / memory reference signals
        if words & _MEMORY_TRIGGERS:
            return RouteType.MEMORY_FIRST

        # Default: full retrieval
        return RouteType.RETRIEVAL


# Module-level singleton
router_agent = RouterAgent()
