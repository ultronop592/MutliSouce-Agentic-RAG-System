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

[FIX] source_filename-aware BM25 search:
  Each BM25 index entry now stores the bare source filename alongside the
  text. When source_filename is provided to search(), only chunks from that
  specific file are scored and returned. This prevents cross-document keyword
  bleed when multiple PDFs exist in the same Qdrant collection.

Usage:
    bm25_manager.refresh(collection)     # call after ingestion
    results = bm25_manager.search(collection, query, top_k=10)
    results = bm25_manager.search(collection, query, top_k=10, source_filename="report.pdf")
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
    # Parallel lists: texts[i] is from source_filenames[i]
    texts: list[str] = field(default_factory=list)
    source_filenames: list[str] = field(default_factory=list)  # bare filename per chunk
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
        Now stores the bare source filename alongside each text so that
        search() can filter by source_filename.
        """
        texts: list[str] = []
        source_filenames: list[str] = []
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
                    payload = point.payload or {}
                    txt = payload.get("text", "")
                    if txt:
                        texts.append(txt)
                        # "source_filename" is stored by ingestion.py (bare basename)
                        src = payload.get("source_filename", "")
                        source_filenames.append(src)
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning("BM25 refresh failed for %s: %s", collection, e)
            return 0

        idx = CollectionIndex(texts=texts, source_filenames=source_filenames)
        idx.build()
        self._indexes[collection] = idx
        logger.info("BM25 index refreshed for '%s': %d docs", collection, len(texts))
        return len(texts)

    def refresh_all(self, collections: list[str]) -> dict[str, int]:
        """Refresh BM25 indexes for multiple collections."""
        return {col: self.refresh(col) for col in collections}

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        source_filename: str | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search the BM25 index for `collection`.
        Returns a list of (text, bm25_score) sorted descending.
        Returns [] if the collection has no index yet.

        source_filename (optional):
            When provided, only chunks whose stored source_filename matches
            are scored and returned. This is the primary fix for cross-document
            keyword bleed: uploading a new PDF no longer causes BM25 to return
            chunks from older PDFs in the same collection.
        """
        idx = self._indexes.get(collection)
        if idx is None or idx.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = idx.bm25.get_scores(tokenized_query)

        # Pair each text with its BM25 score and source filename
        paired: list[tuple[str, float, str]] = list(
            zip(idx.texts, scores, idx.source_filenames)
        )

        # ── Source file filter ───────────────────────────────────────────────
        # When a specific PDF is active, restrict results to only that file's
        # chunks. This mirrors the Qdrant payload filter on the semantic leg.
        if source_filename:
            paired = [
                (text, score, src)
                for text, score, src in paired
                if src == source_filename
            ]
            if not paired:
                logger.debug(
                    "BM25 '%s': no chunks found for source_filename='%s'",
                    collection, source_filename,
                )

        # Sort descending by score, take top_k, filter zero-score results
        paired.sort(key=lambda x: x[1], reverse=True)
        return [(text, score) for text, score, _ in paired[:top_k] if score > 0.0]

    def has_index(self, collection: str) -> bool:
        idx = self._indexes.get(collection)
        return idx is not None and idx.bm25 is not None

    def doc_count(self, collection: str) -> int:
        idx = self._indexes.get(collection)
        return len(idx.texts) if idx else 0


# Global singleton
bm25_manager = BM25IndexManager()
