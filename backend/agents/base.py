"""
Base Agent
----------
Abstract foundation for every agent in the pipeline.
Each agent has a name, a structured run() method, and timing instrumentation
so the orchestrator can log exactly how long each step took in production.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Structured output from any agent — passed between pipeline stages."""
    agent: str          # agent name for logging
    success: bool       # False → downstream agents use fallback
    data: Any           # the agent's primary output
    latency_ms: float   # wall-clock execution time
    metadata: dict = field(default_factory=dict)  # optional debug info

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"[{self.agent}] {status} ({self.latency_ms:.0f}ms)"


class BaseAgent(ABC):
    """Every agent inherits from this class."""

    name: str = "base"

    def run(self, *args, **kwargs) -> AgentResult:
        """Run the agent, measuring latency and catching unexpected errors."""
        t0 = time.perf_counter()
        try:
            result = self._run(*args, **kwargs)
            latency = (time.perf_counter() - t0) * 1000
            logger.info("%s completed in %.0fms", self.name, latency)
            result.latency_ms = latency
            return result
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("%s failed after %.0fms: %s", self.name, latency, exc)
            return AgentResult(
                agent=self.name,
                success=False,
                data=None,
                latency_ms=latency,
                metadata={"error": str(exc)},
            )

    @abstractmethod
    def _run(self, *args, **kwargs) -> AgentResult:
        """Override this in each concrete agent."""
        ...
