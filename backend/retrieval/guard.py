"""
Hallucination Guard Module
--------------------------
Analyses the quality of retrieved context BEFORE it reaches the LLM.
Provides two things:
  1. A confidence level ("high" / "low" / "none") based on retrieval scores.
  2. A system advisory note injected into the prompt so the LLM knows how
     confident to be.

IMPORTANT — Score scale awareness:
  Raw cosine scores: 0.0 – 1.0
  RRF fused scores:  0.01 – 0.033  (much smaller, purely rank-based)
  The guard auto-detects which scale is being used and applies the
  correct threshold so confidence is never mis-classified.
"""

# Thresholds for raw cosine scores (0-1 range)
COSINE_CONFIDENCE_THRESHOLD: float = 0.40
COSINE_HIGH_MAX: float = 0.50

# Thresholds for RRF fused scores (0.01-0.033 range)
RRF_CONFIDENCE_THRESHOLD: float = 0.018
RRF_HIGH_MAX: float = 0.025

# If max score is below this value, treat as RRF scores
RRF_SCALE_CUTOFF: float = 0.10


def assess_confidence(docs_with_scores: list[tuple[str, float]]) -> tuple[str, str]:
    """
    Assess the quality of retrieved documents.

    Args:
        docs_with_scores: list of (text, score) tuples from the retriever.
                          Scores can be cosine similarities OR RRF fused scores.

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

    scores = [s for _, s in docs_with_scores]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    # Auto-detect score scale: RRF scores are always < 0.10
    is_rrf = max_score < RRF_SCALE_CUTOFF
    conf_threshold = RRF_CONFIDENCE_THRESHOLD if is_rrf else COSINE_CONFIDENCE_THRESHOLD
    high_max = RRF_HIGH_MAX if is_rrf else COSINE_HIGH_MAX

    if avg_score >= conf_threshold or max_score >= high_max:
        return (
            "high",
            "[RETRIEVAL CONFIDENCE: HIGH — The retrieved context is directly relevant. "
            "Answer strictly from this context only.]",
        )
    else:
        return (
            "low",
            f"[RETRIEVAL CONFIDENCE: LOW — Average relevance score is {avg_score:.4f}. "
            "The retrieved context may not be directly relevant. Be extremely conservative: "
            "only state what is explicitly present in the context, and clearly "
            "acknowledge any uncertainty. Do NOT infer or add information from general "
            "training knowledge.]",
        )

