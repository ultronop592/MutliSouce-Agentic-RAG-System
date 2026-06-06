"""
Hallucination Guard Module
--------------------------
Analyses the quality of retrieved context BEFORE it reaches the LLM.
Provides two things:
  1. A confidence level ("high" / "low" / "none") based on average retrieval scores.
  2. A system advisory note that is injected into the prompt so the LLM
     knows to be more conservative when the context is weak.
"""

# Minimum average score for "high" confidence retrieval.
# Below this threshold the LLM is explicitly warned to be conservative.
CONFIDENCE_THRESHOLD = 0.45


def assess_confidence(docs_with_scores: list[tuple[str, float]]) -> tuple[str, str]:
    """
    Assess the quality of retrieved documents.

    Args:
        docs_with_scores: list of (text, score) tuples from the retriever.

    Returns:
        (confidence_level, advisory_note) where:
        - confidence_level: "high", "low", or "none"
        - advisory_note: string to inject into the LLM prompt
    """
    if not docs_with_scores:
        return (
            "none",
            "[RETRIEVAL CONFIDENCE: NONE — No relevant documents were found in the "
            "knowledge base for this question. You MUST respond with exactly: "
            "'I don't have enough information in my knowledge base to answer this "
            "question accurately.' Do not attempt to answer from general knowledge.]",
        )

    avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
    max_score = max(score for _, score in docs_with_scores)

    if avg_score >= CONFIDENCE_THRESHOLD or max_score >= 0.60:
        return (
            "high",
            "[RETRIEVAL CONFIDENCE: HIGH — The retrieved context is directly relevant. "
            "Answer strictly from this context only.]",
        )
    else:
        return (
            "low",
            f"[RETRIEVAL CONFIDENCE: LOW — Average relevance score is {avg_score:.2f}, "
            "which is below the confidence threshold. The retrieved context may not be "
            "directly relevant to this question. You MUST be extremely conservative: "
            "only state what is explicitly present in the context, and clearly "
            "acknowledge any uncertainty. Do NOT infer, extrapolate, or add information "
            "from your general training knowledge.]",
        )
