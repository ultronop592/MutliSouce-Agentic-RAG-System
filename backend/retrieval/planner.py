ALL_COLLECTIONS = ["research_papers", "knowledge_base", "code_docs", "faq_data"]


def planner(question: str) -> list[str]:
    """
    Smarter keyword-based collection routing.
    Uses broader matching and always includes a core fallback set
    so the retriever has the best chance of finding relevant context
    (wrong collection → irrelevant chunks → model fills gaps by hallucinating).
    """
    q = question.lower()
    selected = set()

    # Code / API / technical queries
    if any(kw in q for kw in ["api", "function", "code", "class", "method",
                                "implement", "syntax", "example", "snippet",
                                "library", "module", "import", "error", "bug",
                                "exception", "debug", "trace", "stack"]):
        selected.update(["code_docs", "research_papers"])

    # How-to / FAQ / help queries
    if any(kw in q for kw in ["how", "faq", "help", "explain", "what is",
                                "what are", "why", "guide", "tutorial",
                                "step", "process", "procedure"]):
        selected.update(["faq_data", "knowledge_base"])

    # Research / academic / concept queries
    if any(kw in q for kw in ["paper", "research", "study", "model",
                                "algorithm", "architecture", "theory",
                                "transformer", "attention", "neural",
                                "training", "dataset", "evaluation",
                                "benchmark", "performance", "result",
                                "conclusion", "abstract", "experiment"]):
        selected.update(["research_papers", "knowledge_base"])

    # General knowledge / definition queries
    if any(kw in q for kw in ["define", "definition", "concept", "overview",
                                "introduction", "summary", "describe",
                                "difference", "compare", "vs", "versus"]):
        selected.update(["knowledge_base", "research_papers"])

    # If nothing matched, search ALL collections (better than wrong subset)
    if not selected:
        return ALL_COLLECTIONS

    return list(selected)