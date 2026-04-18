from __future__ import annotations

from typing import Any, Callable

from driftguard.detector import DriftDetector
from driftguard.schema import DriftResult


def drift_node(
    detector: DriftDetector,
    text_key: str = "response",
    result_key: str = "drift",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a synchronous LangGraph node that checks the state for drift.

    Reads ``state[text_key]``, runs drift detection, and returns
    ``{result_key: DriftResult}`` as a state update.

    Usage::

        graph.add_node("drift_check", drift_node(detector, text_key="response"))
        graph.add_conditional_edges("drift_check", route_on_drift,
                                    {"drift": "fallback", "ok": "respond"})
    """
    def node(state: dict[str, Any]) -> dict[str, Any]:
        return {result_key: detector.check(str(state[text_key]))}

    return node


def adrift_node(
    detector: DriftDetector,
    text_key: str = "response",
    result_key: str = "drift",
) -> Callable[[dict[str, Any]], Any]:
    """Return an async LangGraph node that checks the state for drift."""

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        return {result_key: await detector.acheck(str(state[text_key]))}

    return node


def make_route_on_drift(
    result_key: str = "drift",
    on_drift: str = "drift",
    on_ok: str = "ok",
) -> Callable[[dict[str, Any]], str]:
    """Return a conditional-edge function that routes based on drift detection.

    The returned function reads ``state[result_key]`` (a ``DriftResult``) and
    returns ``on_drift`` or ``on_ok``.  A missing key is treated as no drift.

    Args:
        result_key: State key written by ``drift_node`` (default ``"drift"``).
        on_drift: Route label to return when drift is detected (default ``"drift"``).
        on_ok: Route label to return when no drift (default ``"ok"``).
    """
    def router(state: dict[str, Any]) -> str:
        result: DriftResult | None = state.get(result_key)
        return on_drift if (result is not None and result.is_drift) else on_ok

    return router


# Pre-built router for the common case — use directly as the conditional edge function.
route_on_drift: Callable[[dict[str, Any]], str] = make_route_on_drift()
