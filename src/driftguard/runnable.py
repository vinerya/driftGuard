from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from driftguard.alerts import AlertManager
from driftguard.detector import DriftDetector
from driftguard.schema import DriftError


class DriftRunnable:
    """Factory for LangChain Runnables that check for drift.

    Two modes:

    - ``as_passthrough()``: passes input through, annotates with drift info.
      Returns ``{"output": <original>, "drift": DriftResult}``.

    - ``as_guard()``: raises ``DriftError`` if drift is detected,
      otherwise passes text through unchanged.

    Usage::

        drift = DriftRunnable(detector=detector)
        chain = llm | StrOutputParser() | drift.as_guard()
    """

    def __init__(
        self,
        detector: DriftDetector,
        alerts: AlertManager | None = None,
    ) -> None:
        self.detector = detector
        self.alerts = alerts or AlertManager()

    def as_passthrough(self) -> Runnable:
        """Runnable that annotates output with drift result."""

        def _check(text: Any) -> dict[str, Any]:
            text = str(text) if not isinstance(text, str) else text
            result = self.detector.check(text)
            if result.is_drift:
                self.alerts.alert(result)
            return {"output": text, "drift": result}

        async def _acheck(text: Any) -> dict[str, Any]:
            text = str(text) if not isinstance(text, str) else text
            result = await self.detector.acheck(text)
            if result.is_drift:
                await self.alerts.aalert(result)
            return {"output": text, "drift": result}

        return RunnableLambda(_check, afunc=_acheck)

    def as_guard(self) -> Runnable:
        """Runnable that raises DriftError on drift, passes through otherwise."""

        def _guard(text: Any) -> str:
            text = str(text) if not isinstance(text, str) else text
            result = self.detector.check(text)
            if result.is_drift:
                self.alerts.alert(result)
                raise DriftError(result)
            return text

        async def _aguard(text: Any) -> str:
            text = str(text) if not isinstance(text, str) else text
            result = await self.detector.acheck(text)
            if result.is_drift:
                await self.alerts.aalert(result)
                raise DriftError(result)
            return text

        return RunnableLambda(_guard, afunc=_aguard)
