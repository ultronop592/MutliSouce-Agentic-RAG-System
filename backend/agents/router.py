"""
Router Agent
------------
Classifies every incoming query into one of three route types so the
orchestrator can pick the fastest, most appropriate pipeline:

  CONVERSATIONAL  ➔  Direct LLM reply (no retrieval needed).
                     Greetings, thanks, "who are you", clarifications.

  MEMORY_FIRST    ➔  Check conversation memory before retrieval.
                     Follow-up questions, pronoun references ("it", "that"),
                     or questions clearly related to the chat so far.

  RETRIEVAL       ➔  Full retrieval pipeline required.
                     Factual questions, research topics, technical queries.

Uses a fast pre-filter heuristic to instantly classify basic greetings (0ms latency),
and upgrades to a Gemini LLM call for semantic intent classification on complex queries.
"""

import logging
from enum import Enum
from agents.base import BaseAgent, AgentResult
from core.llm import llm

logger = logging.getLogger(__name__)


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
        import time
        t0 = time.perf_counter()
        route = self._classify(query)
        latency = (time.perf_counter() - t0) * 1000
        logger.info("RouterAgent classification: %s in %.1fms", route.value, latency)
        return AgentResult(
            agent=self.name,
            success=True,
            data=route,
            latency_ms=latency,
            metadata={"query_length": len(query), "route": route.value},
        )

    def _classify(self, query: str) -> RouteType:
        q = query.lower().strip()
        words = set(q.split())

        # Fast-path heuristic check for simple greetings to save API cost & latency (0ms)
        if len(q) <= 15 and (words & _CONVERSATIONAL_TRIGGERS or q in _CONVERSATIONAL_TRIGGERS):
            logger.debug("RouterAgent: fast-path greeting check hit")
            return RouteType.CONVERSATIONAL

        # Upgrade to Gemini LLM Classifier for semantic intent routing
        prompt = f"""You are an intelligent query router for a Retrieval-Augmented Generation (RAG) assistant.
Classify the user's input query into one of three routes:

1. "conversational": For greetings, thank yous, system status, general chat, or questions about who/what you are (e.g. "hi there", "who built you?", "thanks for your help", "how is it going?").
2. "memory_first": For follow-up questions referencing something discussed earlier, containing pronouns or implicit references, or asking to explain/elaborate/continue (e.g. "can you explain it further?", "what about that?", "go on", "why?").
3. "retrieval": For technical, factual, domain-specific, or document-related questions that require searching external knowledge bases (e.g. "explain self-attention", "how do RRF algorithms work?", "what are developers onboarding rules?").

User query: {query}

Return ONLY a single word: "conversational", "memory_first", or "retrieval". No formatting, markdown, or explanation.
Your classification:"""

        try:
            response = llm.invoke(prompt)
            raw = response.content.strip().lower().strip('"\'')
            if "conversational" in raw:
                return RouteType.CONVERSATIONAL
            elif "memory" in raw:
                return RouteType.MEMORY_FIRST
            else:
                return RouteType.RETRIEVAL
        except Exception as e:
            logger.warning("RouterAgent: Gemini classification failed, falling back to RETRIEVAL: %s", e)
            return RouteType.RETRIEVAL


# Module-level singleton
router_agent = RouterAgent()
