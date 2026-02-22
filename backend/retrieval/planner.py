import json
from core.llm import llm


AVAILABLE_COLLECTIONS = ["research_papers", "knowledge_base", "code_docs", "faq_data"]


def planner(question: str) -> list[str]:
    """
    Use Gemini to intelligently select which Qdrant collections
    to search based on the user's question.
    Falls back to keyword-based selection on error.
    """
    try:
        prompt = f"""You are a query routing agent. Given a user question, decide which data sources to search.

Available collections:
- research_papers: Academic papers, scientific research, technical publications
- knowledge_base: General knowledge, documentation, guides, tutorials
- code_docs: Code documentation, API references, programming guides
- faq_data: Frequently asked questions and answers

Return ONLY a JSON array of collection names to search (1-3 collections).
Example: ["research_papers", "knowledge_base"]

Question: {question}
"""
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON array from response
        if "[" in content and "]" in content:
            start = content.index("[")
            end = content.rindex("]") + 1
            selected = json.loads(content[start:end])
            # Validate collection names
            valid = [c for c in selected if c in AVAILABLE_COLLECTIONS]
            if valid:
                return valid
    except Exception:
        pass

    # Fallback: keyword-based selection
    q = question.lower()
    if "api" in q or "function" in q or "code" in q:
        return ["code_docs", "research_papers"]
    if "how" in q or "faq" in q or "help" in q:
        return ["faq_data", "knowledge_base"]
    return ["research_papers", "knowledge_base"]