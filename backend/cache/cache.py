"""
Semantic Cache with TTL — Session-Scoped
-----------------------------------------
CRITICAL FIX: The previous implementation used a single global flat list
shared across ALL sessions and ALL tabs. This meant:
  - User uploads PDF-A, asks "Summarize key findings" → answer cached globally.
  - User uploads PDF-B in a new session, asks "Summarize main topics" →
    embedding similarity > 0.95 → old PDF-A answer returned. HALLUCINATION.

This version keys the cache by session_id so cache hits from one session
NEVER bleed into another session. Each session has its own independent
answer store.

Additional fix: clear(session_id) now also wipes that session's cache,
so "New Chat" truly starts fresh with no stale answers.
"""

import time
import math
from collections import defaultdict

# ── Tunable constants ────────────────────────────────────────────────────────
# Raised threshold: even identical-looking questions about different PDFs must
# not collide. 0.98 = near-identical phrasing only within the same session.
SIMILARITY_THRESHOLD: float = 0.97
TTL_SECONDS: int = 3600             # 1 hour — stale/hallucinated answers auto-expire


def _cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    """
    Session-scoped in-memory semantic cache backed by embedding similarity.

    Each session has its own independent list of (embedding, answer, timestamp)
    entries. A cache hit in session A can NEVER be returned for session B.

    This prevents the primary hallucination vector: serving a cached answer
    from PDF-A when the user is asking about a completely different PDF-B in
    a different session or tab.
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        ttl_seconds: int = TTL_SECONDS,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        # Keyed by session_id → list of (embedding, answer, timestamp)
        self._store: dict[str, list[tuple[list[float], str, float]]] = defaultdict(list)

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl_seconds

    def _evict_expired(self, session_id: str):
        """Remove expired entries for a specific session (lazy eviction)."""
        if session_id in self._store:
            self._store[session_id] = [
                entry for entry in self._store[session_id]
                if not self._is_expired(entry[2])
            ]

    def get(self, session_id: str, query_embedding: list[float]) -> str | None:
        """
        Return a cached answer if a semantically similar question exists
        WITHIN THE SAME SESSION and has not expired.
        Returns None on cache miss.

        The session_id parameter is MANDATORY — cross-session hits are
        intentionally impossible by design.
        """
        self._evict_expired(session_id)
        session_entries = self._store.get(session_id, [])
        best_sim = 0.0
        best_answer = None
        for cached_emb, cached_answer, _ in session_entries:
            sim = _cosine(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_answer = cached_answer
        if best_sim >= self.similarity_threshold:
            return best_answer
        return None

    def set(self, session_id: str, query_embedding: list[float], answer: str):
        """Store a new cache entry scoped to session_id."""
        self._evict_expired(session_id)
        self._store[session_id].append((query_embedding, answer, time.time()))

    def clear(self, session_id: str | None = None):
        """
        Clear cache for a specific session (e.g., when user clicks "New Chat").
        If session_id is None, clears ALL sessions.
        """
        if session_id is not None:
            self._store.pop(session_id, None)
        else:
            self._store.clear()

    def clear_all(self):
        """Clear the entire cache across all sessions."""
        self._store.clear()

    def __len__(self) -> int:
        total = 0
        for session_id in list(self._store.keys()):
            self._evict_expired(session_id)
            total += len(self._store.get(session_id, []))
        return total


# Global singleton — imported everywhere
response_cache = SemanticCache()