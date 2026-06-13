"""
Verification Agent
------------------
A fact-checking agent that verifies the generated RAG response against the
original retrieved documents to detect and reject hallucinations.

HALLUCINATION FIX:
  - Removed the overly broad short-circuit that treated ANY answer containing
    "enough information" as verified. Now only the exact sentinel message is
    allowed through without verification.
  - Tightened the verification prompt to explicitly catch generic-knowledge
    hallucinations (facts that feel correct but aren't in the context).
"""

import json
import logging
import re
from agents.base import BaseAgent, AgentResult
from core.llm import llm

logger = logging.getLogger(__name__)

# The exact message the AnswerAgent emits when context is insufficient.
# Only this exact message is allowed through without full verification.
_NO_INFO_SENTINEL = "I don't have enough information in my knowledge base to answer this accurately."


class VerificationAgent(BaseAgent):
    name = "VerificationAgent"

    def _run(self, query: str) -> AgentResult:
        # Standard synchronous signature wrapper
        return AgentResult(
            agent=self.name,
            success=False,
            data={"status": "error", "reason": "Use verify_async"},
            latency_ms=0.0,
        )

    async def verify_async(self, answer: str, context: str) -> dict:
        """
        Verify the generated answer against the retrieved context.
        Returns a dict: {"status": "verified" | "failed" | "error", "reason": str}
        """
        # ── Exact sentinel check (not a broad substring match) ────────────────
        # Only the precise "I don't have enough information..." message is
        # pre-approved. Any other answer that merely contains words like
        # "enough information" must go through full verification.
        answer_stripped = answer.strip()
        if answer_stripped == _NO_INFO_SENTINEL:
            return {"status": "verified", "reason": "No-info default reply"}

        prompt = f"""You are an elite RAG Fact-Verification Agent. Your task is to evaluate a generated answer against the retrieved context to verify that every single statement in the answer is 100% grounded in the context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Instructions:
1. Carefully compare every claim in the Generated Answer against the Retrieved Context.
2. Flag any statement that meets ANY of these criteria:
   - Not explicitly stated in the retrieved context.
   - An inference or extrapolation beyond what the context says.
   - A generic fact from common knowledge that does NOT appear in the context.
   - A mix of facts from different documents that creates a misleading composite.
3. If the answer is 100% grounded (every claim has a direct basis in the context), return:
{{"status": "verified", "reason": "all claims are grounded"}}
4. If ANY hallucination, extrapolation, generic-knowledge fact, or ungrounded claim exists, return:
{{"status": "failed", "reason": "<specific detail: which exact statement is ungrounded and why>"}}

IMPORTANT: Be strict. An answer that is mostly correct but contains even one sentence
drawn from general knowledge (not the context) should be marked "failed".

Return ONLY a valid JSON object. No explanation outside the JSON.
Verification Result:"""

        import time
        t0 = time.perf_counter()

        try:
            # Use async invoke to avoid blocking other concurrent connections
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()

            # Parse JSON
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                logger.warning("VerificationAgent: could not parse JSON from: %s", raw[:200])
                return {"status": "error", "reason": "Non-JSON response"}

            parsed = json.loads(match.group())
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                "VerificationAgent: status=%s | latency=%.0fms | reason='%s'",
                parsed.get("status"), latency, parsed.get("reason")
            )
            return parsed

        except Exception as e:
            logger.error("VerificationAgent failed: %s", e)
            return {"status": "error", "reason": str(e)}


# Module-level singleton
verification_agent = VerificationAgent()
