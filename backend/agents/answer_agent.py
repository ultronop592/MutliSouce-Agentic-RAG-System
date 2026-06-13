"""
Answer Agent
------------
Generates the final, grounded answer from retrieved + reranked context.

Produces Claude/GPT-style responses:
  - Adaptive length: short for factual, detailed for complex questions
  - Markdown formatting: **bold**, bullet points, numbered lists, headers
  - Streams token-by-token so the frontend displays words progressively
  - Strict grounding: every claim must be traceable to the retrieved context

HALLUCINATION FIX: Explicit cross-document isolation instruction added.
The LLM is now explicitly told to ignore any facts from Previous Conversation
that are not also confirmed by the current Retrieved Context. This prevents
bleed from earlier PDF discussions into the current answer.
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
        """Assemble the Claude/GPT-style grounded prompt."""
        return f"""You are a precise, knowledgeable assistant. Answer the user's question using ONLY the retrieved context below.

{advisory_note}

══ RESPONSE FORMAT — adapt based on the question type ══
• Simple fact / yes-no question   → 1–2 direct sentences, no headers
• Explanation / how it works      → Short paragraphs, **bold** key terms
• List questions ("what are...")  → Bullet points  (- item)
• Step-by-step / process          → Numbered list  (1. step)
• Overview / summary              → Paragraphs + **bold** section labels
• Comparison / pros-cons          → Short structured layout with bullets
• Complex multi-part question     → Use ## headers to organise sections

══ WRITING STYLE ══
• Be direct and confident — write like Claude or ChatGPT
• Use **bold** to highlight the most important terms and concepts
• Use bullet points for 3 or more parallel items
• Use numbered lists only for sequential steps or ranked items
• Match response length to question complexity:
    – Simple question → a few sentences is enough
    – Deep question   → multiple paragraphs with structure
• Do NOT open with "Based on the context..." or "According to the document..."
• Do NOT say "the retrieved context says..." — just answer naturally
• Do NOT add citation numbers like [1] or [2]

══ GROUNDING RULES (these override everything — non-negotiable) ══
1. Use ONLY facts, names, numbers, and claims that appear EXPLICITLY in the Retrieved Context.
2. Do NOT add anything from your general training knowledge, even if you are certain it is correct.
3. Do NOT infer, extrapolate, or fill gaps — only state what the context says.
4. If the context does not contain enough information, respond with exactly:
   "I don't have enough information in my knowledge base to answer this accurately."
5. Before writing each sentence, mentally ask: "Is this explicitly in the Retrieved Context?" — if not, omit it.

══ CROSS-DOCUMENT ISOLATION (critical anti-hallucination rule) ══
6. The "Previous Conversation" section below contains answers from earlier questions.
   Those answers may have been about a DIFFERENT document than the current one.
7. Do NOT use any fact or claim from Previous Conversation UNLESS that exact fact
   also appears in the current Retrieved Context.
8. The Retrieved Context is the ONLY trusted source for this response.
   Previous Conversation is provided for conversational tone only — NOT as a fact source.

Previous Conversation:
{memory_str if memory_str else "(none)"}

Retrieved Context:
{context}

User Question: {query}

Answer:"""

    async def generate(
        self,
        query: str,
        context: str,
        memory_str: str,
        advisory_note: str,
        feedback: str = "",
    ) -> str:
        """Generate the full answer in memory (non-streaming). Supports correction feedback."""
        prompt = self.build_prompt(query, context, memory_str, advisory_note)
        if feedback:
            prompt += f"\n\n[Verification Correction Feedback: Your previous generation was flagged for containing ungrounded facts or discrepancies: '{feedback}'. Please rewrite the response, ensuring that it is 100% faithful to the retrieved context and resolves this issue.]"
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("AnswerAgent generation error: %s", e)
            return f"[Error generating response: {e}]"

    async def stream(
        self,
        query: str,
        context: str,
        memory_str: str,
        advisory_note: str,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token for word-by-word display."""
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
