import asyncio
from rank_bm25 import BM25Okapi
from core.qdrant_client import qdrant
from core.embeddings import embeddings
from core.llm import llm

COLLECTION_CONFIDENCE = {
    "research_papers": 1.0,
    "knowledge_base": 0.8,
    "code_docs": 1.2,
    "faq_data": 0.7,
}


def dynamic_k(name: str) -> int:
    return 5 if name == "code_docs" else 3


def rewrite_query(question: str) -> str:
    """Use Gemini to rewrite the user question into a better search query."""
    try:
        prompt = f"""Rewrite the following question into ONE clear standalone search query.

Return ONLY the rewritten query.
Do NOT explain.
Do NOT give options.

Question: {question}
"""
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        if rewritten and len(rewritten) > 3:
            return rewritten
    except Exception:
        pass
    return question


async def retrieve(collection: str, query: str) -> list[tuple[str, float]]:
    """Retrieve documents from a single Qdrant collection."""
    try:
        vector = embeddings.embed_query(query)

        results = qdrant.query_points(
            collection_name=collection,
            query=vector,
            limit=dynamic_k(collection),
        )

        confidence = COLLECTION_CONFIDENCE.get(collection, 1.0)
        return [
            (point.payload["text"], point.score * confidence)
            for point in results.points
            if "text" in point.payload
        ]
    except Exception:
        # Collection might be empty or not exist
        return []


async def hybrid_retrieve(query: str, selected: list[str]) -> list[str]:
    """
    Hybrid retrieval: vector search across multiple collections,
    then BM25 reranking on the merged results.
    """
    # Step 1: Rewrite query for better retrieval
    rewritten = rewrite_query(query)

    # Step 2: Parallel retrieval from selected collections
    tasks = [retrieve(c, rewritten) for c in selected]
    results = await asyncio.gather(*tasks)

    # Step 3: Merge and sort by vector score
    merged = []
    for r in results:
        merged.extend(r)

    if not merged:
        return []

    merged.sort(key=lambda x: x[1], reverse=True)
    top_vector_docs = [doc for doc, score in merged[:8]]

    if len(top_vector_docs) < 2:
        return top_vector_docs

    # Step 4: BM25 reranking
    try:
        tokenized_docs = [doc.split() for doc in top_vector_docs]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(rewritten.split())
        combined = list(zip(top_vector_docs, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in combined[:5]]
    except Exception:
        return top_vector_docs[:5]