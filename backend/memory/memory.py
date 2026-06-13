"""
Chat Memory
-----------
Persistent in-memory conversation history, keyed by session_id.

Enhancements over the original:
  1. Embedding caching — each turn optionally stores the question embedding
     so the MemoryAgent can do semantic lookup without re-computing embeddings.
  2. get_recent() — returns only the last N turns (used by MemoryAgent for
     fast cosine comparison without scanning the full history).
  3. format_history() — unchanged interface for the Answer Agent.
  4. [NEW] doc_version on each turn — MemoryAgent can skip turns that belong
     to a different uploaded document, preventing cross-document answer bleed.
"""

from collections import defaultdict


class ChatMemory:
    """In-memory conversation store, keyed by session_id."""

    def __init__(self, max_turns: int = 15):
        self.max_turns = max_turns
        # Each turn: {
        #   "user": str, "assistant": str,
        #   "embedding": list|None,
        #   "doc_version": str|None   ← NEW: fingerprint of active PDF
        # }
        self._store: dict[str, list[dict]] = defaultdict(list)

    # ── Write ────────────────────────────────────────────────────────────────

    def update(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        embedding: list[float] | None = None,
        doc_version: str | None = None,
    ):
        """
        Store a conversation turn.

        Args:
            session_id:    unique session identifier
            user_msg:      user's question
            assistant_msg: assistant's answer
            embedding:     pre-computed embedding of user_msg (optional).
                           When provided, the MemoryAgent can match this
                           turn in future semantic lookups without re-embedding.
            doc_version:   MD5 fingerprint of the currently active PDF (optional).
                           When provided, the MemoryAgent will only reuse this
                           answer when the SAME document version is active,
                           preventing cross-document answer bleed.
        """
        history = self._store[session_id]
        history.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "embedding": embedding,
            "doc_version": doc_version,  # None if no document uploaded yet
        })
        # Trim to window
        if len(history) > self.max_turns:
            self._store[session_id] = history[-self.max_turns:]

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_recent(self, session_id: str, n: int) -> list[dict]:
        """Return the last n turns for a session (used by MemoryAgent)."""
        return self._store.get(session_id, [])[-n:]

    def format_history(self, session_id: str) -> str:
        """Format the last max_turns turns as a readable dialogue string."""
        history = self._store.get(session_id, [])
        if not history:
            return ""
        lines = []
        for turn in history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}\n")
        return "\n".join(lines)

    # ── Clear ────────────────────────────────────────────────────────────────

    def clear(self, session_id: str):
        self._store.pop(session_id, None)

    def clear_all(self):
        self._store.clear()

    # ── Stats ────────────────────────────────────────────────────────────────

    def turn_count(self, session_id: str) -> int:
        return len(self._store.get(session_id, []))

    def session_ids(self) -> list[str]:
        return list(self._store.keys())


# Global singleton
chat_memory = ChatMemory()
