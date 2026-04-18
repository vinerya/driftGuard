from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from dataclasses import asdict
from typing import Awaitable, Callable, Sequence, Union

from driftguard.schema import DriftResult

logger = logging.getLogger("driftguard")

Sink = Union[str, Callable[[DriftResult], None], Callable[[DriftResult], Awaitable[None]]]


class AlertManager:
    """Dispatches drift alerts to configured sinks.

    Sink types:
    - "log": logs at WARNING level
    - HTTP URL string: POST JSON payload
    - callable: invoked with the DriftResult
    """

    def __init__(self, sinks: Sequence[Sink] | None = None) -> None:
        self._sinks: list[Sink] = list(sinks) if sinks else ["log"]

    def alert(self, result: DriftResult) -> None:
        """Send alert to all sinks (sync)."""
        for sink in self._sinks:
            if sink == "log":
                self._log(result)
            elif isinstance(sink, str) and sink.startswith("http"):
                self._webhook(sink, result)
            elif callable(sink):
                sink(result)

    async def aalert(self, result: DriftResult) -> None:
        """Send alert to all sinks (async)."""
        for sink in self._sinks:
            if sink == "log":
                self._log(result)
            elif isinstance(sink, str) and sink.startswith("http"):
                await asyncio.to_thread(self._webhook, sink, result)
            elif callable(sink):
                ret = sink(result)
                if asyncio.iscoroutine(ret):
                    await ret

    def _log(self, result: DriftResult) -> None:
        logger.warning(
            "DRIFT DETECTED | centroid_sim=%.4f threshold=%.4f | %s",
            result.centroid_similarity,
            result.threshold,
            result.text[:120],
        )

    def _webhook(self, url: str, result: DriftResult) -> None:
        payload = json.dumps(asdict(result), default=str).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            logger.exception("Failed to send drift webhook to %s", url)
