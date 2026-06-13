"""
Semantic Cache with TTL — Session-Scoped, Document-Version-Aware
-----------------------------------------------------------------
CRITICAL FIX (v2): The previous implementation was session-scoped but
NOT document-aware. This meant:
  - User uploads PDF-A, asks "Summarize key findings" → answer cached for session.
  - User uploads PDF-B in the SAME session, asks same question →
    embedding similarity > 0.97 → old PDF-A answer returned. HALLUCINATION.

This version additionally keys each cache entry by `doc_version`
(an MD5 fingerprint of the uploaded file). A cache hit requires:
  1. Same session_id   (already enforced in v1)
  2. Same doc_version  (NEW — prevents cross-document stale answers)

When a new PDF is uploaded to a session, the session's doc_version changes
and ALL previous cache entries for that session become invalid automatically.
No "New Chat" click required.

Additional fix: clear(session_id) also wipes that session's doc_version.
"""

import time
import math
from collections import defaultdict

# ── Tunable constants ────────────────────────────────────────────────────────
# Near-identical phrasing within same session AND same document only.
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
    Session-scoped, document-version-aware in-memory semantic cache.

    Each cache entry stores: (embedding, answer, timestamp, doc_version).
    A cache hit requires matching session_id AND doc_version.

    This prevents two hallucination vectors:
      1. Cross-session bleed: answer from one tab/session appearing in another.
      2. Cross-document bleed: answer cached for PDF-A returned when PDF-B is
         the active document in the same session.
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        ttl_seconds: int = TTL_SECONDS,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        # session_id → list of (embedding, answer, timestamp, doc_version)
        self._store: dict[str, list[tuple[list[float], str, float, str | None]]] = (
            defaultdict(list)
        )
        # session_id → current active doc_version
        self._session_doc_version: dict[str, str | None] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl_seconds

    def _evict_expired(self, session_id: str):
        """Remove expired entries for a specific session (lazy eviction)."""
        if session_id in self._store:
            self._store[session_id] = [
                entry for entry in self._store[session_id]
                if not self._is_expired(entry[2])
            ]

    # ── Document version management ───────────────────────────────────────────

    def set_doc_version(self, session_id: str, doc_version: str):
        """
        Register a new document version for a session.
        This is called by the upload endpoint when a new PDF is ingested.
        All old cache entries for this session become stale (won't match).
        """
        old = self._session_doc_version.get(session_id)
        self._session_doc_version[session_id] = doc_version
        if old != doc_version:
            # Eagerly purge entries that won't match the new version anyway
            if session_id in self._store:
                self._store[session_id] = [
                    entry for entry in self._store[session_id]
                    if entry[3] == doc_version
                ]

    def get_doc_version(self, session_id: str) -> str | None:
        """Return the current doc_version for a session (or None if unset)."""
        return self._session_doc_version.get(session_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        session_id: str,
        query_embedding: list[float],
        doc_version: str | None = None,
    ) -> str | None:
        """
        Return a cached answer if a semantically similar question exists
        WITHIN THE SAME SESSION, has not expired, AND was cached for the
        SAME document version.

        Returns None on any mismatch (session, doc_version, similarity, TTL).
        """
        self._evict_expired(session_id)
        session_entries = self._store.get(session_id, [])
        best_sim = 0.0
        best_answer = None

        for cached_emb, cached_answer, _, cached_doc_ver in session_entries:
            # ── Document version gate ─────────────────────────────────────────
            # If a doc_version is provided, only match entries from that version.
            # If both are None, allow the match (pre-upload questions).
            if doc_version is not None and cached_doc_ver != doc_version:
                continue  # different document — skip, never return stale answer

            sim = _cosine(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_answer = cached_answer

        if best_sim >= self.similarity_threshold:
            return best_answer
        return None

    def set(
        self,
        session_id: str,
        query_embedding: list[float],
        answer: str,
        doc_version: str | None = None,
    ):
        """Store a new cache entry scoped to session_id and doc_version."""
        self._evict_expired(session_id)
        self._store[session_id].append(
            (query_embedding, answer, time.time(), doc_version)
        )

    def clear(self, session_id: str | None = None):
        """
        Clear cache for a specific session (e.g., when user clicks "New Chat").
        Also clears the stored doc_version for that session.
        If session_id is None, clears ALL sessions.
        """
        if session_id is not None:
            self._store.pop(session_id, None)
            self._session_doc_version.pop(session_id, None)
        else:
            self._store.clear()
            self._session_doc_version.clear()

    def clear_all(self):
        """Clear the entire cache across all sessions."""
        self._store.clear()
        self._session_doc_version.clear()

    def __len__(self) -> int:
        total = 0
        for session_id in list(self._store.keys()):
            self._evict_expired(session_id)
            total += len(self._store.get(session_id, []))
        return total


# Global singleton — imported everywhere
response_cache = SemanticCache()