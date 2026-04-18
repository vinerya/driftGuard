from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import LLMResult

from driftguard.alerts import AlertManager
from driftguard.detector import DriftDetector
from driftguard.schema import DriftResult


def _extract_text(response: LLMResult) -> str | None:
    """Extract generated text from an LLMResult."""
    for gen_list in response.generations:
        for gen in gen_list:
            if gen.text:
                return gen.text
            # Chat models: check message content
            if hasattr(gen, "message") and hasattr(gen.message, "content"):
                content = gen.message.content
                if isinstance(content, str) and content:
                    return content
    return None


class DriftCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that monitors responses for drift.

    Non-blocking: observes responses, fires alerts on drift,
    but does not halt the pipeline.

    Usage::

        handler = DriftCallbackHandler(detector=detector)
        llm = ChatOpenAI(callbacks=[handler])
    """

    def __init__(
        self,
        detector: DriftDetector,
        alerts: AlertManager | None = None,
        on_drift: Any | None = None,
    ) -> None:
        self.detector = detector
        self.alerts = alerts or AlertManager()
        self.on_drift = on_drift
        self.history: list[DriftResult] = []

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        text = _extract_text(response)
        if not text:
            return
        result = self.detector.check(text, run_id=str(run_id))
        self.history.append(result)
        if result.is_drift:
            self.alerts.alert(result)
            if self.on_drift:
                self.on_drift(result)


class AsyncDriftCallbackHandler(AsyncCallbackHandler):
    """Async version of DriftCallbackHandler."""

    def __init__(
        self,
        detector: DriftDetector,
        alerts: AlertManager | None = None,
        on_drift: Any | None = None,
    ) -> None:
        self.detector = detector
        self.alerts = alerts or AlertManager()
        self.on_drift = on_drift
        self.history: list[DriftResult] = []

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        text = _extract_text(response)
        if not text:
            return
        result = await self.detector.acheck(text, run_id=str(run_id))
        self.history.append(result)
        if result.is_drift:
            await self.alerts.aalert(result)
            if self.on_drift:
                ret = self.on_drift(result)
                if hasattr(ret, "__await__"):
                    await ret
