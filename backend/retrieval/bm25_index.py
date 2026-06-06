"""
BM25 Index Manager
------------------
Builds and maintains an in-memory BM25Okapi index per Qdrant collection.
The index is populated by scrolling all document texts from Qdrant so that
BM25 search runs independently on the FULL corpus — not just the few chunks
that the vector search already returned.

This is what makes the search truly hybrid:
  - Semantic search finds conceptually similar passages.
  - BM25 finds passages with matching keywords / exact terms.
  - Both are fused with Reciprocal Rank Fusion (RRF) in hybrid.py.

Usage:
    bm25_manager.refresh(collection)     # call after ingestion
    results = bm25_manager.search(collection, query, top_k=10)
"""

import logging
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from core.qdrant_client import qdrant

logger = logging.getLogger(__name__)

# How many docs to pull from Qdrant per scroll batch
SCROLL_BATCH = 200


@dataclass
class CollectionIndex:
    texts: list[str] = field(default_factory=list)
    bm25: BM25Okapi | None = None

    def build(self):
        """Rebuild the BM25Okapi index from the stored texts."""
        if not self.texts:
            self.bm25 = None
            return
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)


class BM25IndexManager:
    """
    Manages one BM25 index per Qdrant collection.
    Thread-safe for read; refresh should be called from a single writer.
    """

    def __init__(self):
        self._indexes: dict[str, CollectionIndex] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def refresh(self, collection: str) -> int:
        """
        Scroll ALL documents in `collection` from Qdrant and rebuild the
        BM25 index.  Returns the number of documents indexed.
        """
        texts: list[str] = []
        offset = None

        try:
            while True:
                results, next_offset = qdrant.scroll(
                    collection_name=collection,
                    limit=SCROLL_BATCH,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # we only need text, save bandwidth
                )
                for point in results:
                    txt = (point.payload or {}).get("text", "")
                    if txt:
                        texts.append(txt)
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning("BM25 refresh failed for %s: %s", collection, e)
            return 0

        idx = CollectionIndex(texts=texts)
        idx.build()
        self._indexes[collection] = idx
        logger.info("BM25 index refreshed for '%s': %d docs", collection, len(texts))
        return len(texts)

    def refresh_all(self, collections: list[str]) -> dict[str, int]:
        """Refresh BM25 indexes for multiple collections."""
        return {col: self.refresh(col) for col in collections}

    def search(
        self, collection: str, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Search the BM25 index for `collection`.
        Returns a list of (text, bm25_score) sorted descending.
        Returns [] if the collection has no index yet.
        """
        idx = self._indexes.get(collection)
        if idx is None or idx.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = idx.bm25.get_scores(tokenized_query)

        # Pair each text with its BM25 score and return top_k
        paired = list(zip(idx.texts, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        # Filter zero-score results (no keyword overlap at all)
        return [(t, s) for t, s in paired[:top_k] if s > 0.0]

    def has_index(self, collection: str) -> bool:
        idx = self._indexes.get(collection)
        return idx is not None and idx.bm25 is not None

    def doc_count(self, collection: str) -> int:
        idx = self._indexes.get(collection)
        return len(idx.texts) if idx else 0


# Global singleton
bm25_manager = BM25IndexManager()
