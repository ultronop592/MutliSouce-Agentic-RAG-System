"""
Answer Agent
------------
Generates the final, grounded answer from retrieved + reranked context.

Responsibilities:
  - Builds the anti-hallucination prompt with the guard advisory injected.
  - Streams the LLM response token-by-token.
  - The prompt enforces strict grounding: every claim must be traceable to the
    retrieved context. No inference, no training-knowledge leakage.

The Answer Agent is always the LAST agent in the pipeline — it never calls
other agents and never modifies memory or cache (that's the orchestrator's job).
"""

import logging
from typing import AsyncIterator
from core.llm import llm

logger = logging.getLogger(__name__)


class AnswerAgent:
    name = "AnswerAgent"

    def build_prompt(
        self,
        query: str,
        context: str,
        memory_str: str,
        advisory_note: str,
    ) -> str:
        """Assemble the strict anti-hallucination prompt."""
        return f"""You are a precise, factual assistant. Your ONLY job is to answer \
the user's question using the retrieved context below.

{advisory_note}

STRICT GROUNDING RULES:
1. ONLY use facts, names, numbers, dates, and claims that appear EXPLICITLY in \
the Retrieved Context below.
2. Do NOT add information from your general training knowledge — even if you are \
confident it is correct.
3. Do NOT infer, extrapolate, or fill in anything absent from the context.
4. If the context is insufficient, reply with exactly: \
"I don't have enough information in my knowledge base to answer this accurately."
5. No markdown headers (##, ###), bullet points, bold (**), or italic (*).
6. No citation numbers like [1] or figure references like [Figure 1].
7. Write 2–4 plain sentences only. Be direct.
8. Mentally verify each sentence before writing: \
"Is this explicitly stated in the Retrieved Context?" — if not, omit it.

Previous Conversation:
{memory_str if memory_str else "(none)"}

Retrieved Context:
{context}

User Question:
{query}

Answer (plain prose, grounded strictly in the context above):"""

    async def stream(
        self,
        query: str,
        context: str,
        memory_str: str,
        advisory_note: str,
    ) -> AsyncIterator[str]:
        """Stream the answer token by token."""
        prompt = self.build_prompt(query, context, memory_str, advisory_note)
        try:
            response = llm.stream(prompt)
            for chunk in response:
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("AnswerAgent stream error: %s", e)
            yield f"\n\n[Error generating response: {e}]"


# Module-level singleton
answer_agent = AnswerAgent()
