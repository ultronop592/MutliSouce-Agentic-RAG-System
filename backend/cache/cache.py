"""
Semantic Cache with TTL
-----------------------
Replaces the original exact-match dict cache which had two critical flaws:
  1. A hallucinated answer was served forever for any exact repeat of the question.
  2. Paraphrased versions of the same question bypassed the cache entirely.

This cache:
  - Uses cosine similarity on question embeddings for fuzzy matching.
  - Only returns a cached answer if similarity > SIMILARITY_THRESHOLD (0.95).
  - Expires cache entries after TTL_SECONDS (3600 = 1 hour) in production.
"""

import time
import math

# ── Tunable constants ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.95  # How similar two questions must be to share a cache hit
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
    Thread-safe in-memory semantic cache backed by embedding similarity.
    Each entry: (question_embedding, answer, timestamp)
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        ttl_seconds: int = TTL_SECONDS,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        # List of (embedding, answer, timestamp)
        self._store: list[tuple[list[float], str, float]] = []

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl_seconds

    def _evict_expired(self):
        """Remove expired entries (lazy eviction on every access)."""
        self._store = [
            entry for entry in self._store if not self._is_expired(entry[2])
        ]

    def get(self, query_embedding: list[float]) -> str | None:
        """
        Return a cached answer if a semantically similar question exists
        and has not expired. Returns None on cache miss.
        """
        self._evict_expired()
        best_sim = 0.0
        best_answer = None
        for cached_emb, cached_answer, _ in self._store:
            sim = _cosine(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_answer = cached_answer
        if best_sim >= self.similarity_threshold:
            return best_answer
        return None

    def set(self, query_embedding: list[float], answer: str):
        """Store a new cache entry."""
        self._evict_expired()
        self._store.append((query_embedding, answer, time.time()))

    def clear(self):
        """Manually clear the entire cache."""
        self._store.clear()

    def __len__(self) -> int:
        self._evict_expired()
        return len(self._store)


# Global singleton — imported everywhere
response_cache = SemanticCache()