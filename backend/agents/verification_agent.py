"""
Verification Agent
------------------
A fact-checking agent that verifies the generated RAG response against the
original retrieved documents to detect and reject hallucinations.
"""

import json
import logging
import re
from agents.base import BaseAgent, AgentResult
from core.llm import llm

logger = logging.getLogger(__name__)


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
        # If the answer is the empty-context message, it's always verified
        NO_INFO_MSG = "I don't have enough information in my knowledge base to answer this accurately."
        if NO_INFO_MSG in answer or "enough information" in answer.lower():
            return {"status": "verified", "reason": "No-info default reply"}

        prompt = f"""You are an elite RAG Fact-Verification Agent. Your task is to evaluate a generated answer against the retrieved context to verify that every single statement in the answer is 100% grounded in the context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Instructions:
1. Carefully compare the Generated Answer against the Retrieved Context.
2. Check if the Generated Answer contains any statements, assumptions, or facts that do NOT appear explicitly in the retrieved context.
3. If the answer is 100% grounded and contains no ungrounded claims or extrapolations, return:
{{"status": "verified", "reason": "all claims are grounded"}}
4. If there is a hallucination, extrapolation, or ungrounded claim, return:
{{"status": "failed", "reason": "<specific detail describing which statement in the answer is ungrounded or hallucinated>"}}

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
