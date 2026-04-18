from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional

import numpy as np
from langchain_core.embeddings import Embeddings

from driftguard._math import compute_centroid, cosine_similarity
from driftguard.corpus import ReferenceCorpus
from driftguard.detector import DriftDetector
from driftguard.schema import DriftResult, WindowDriftResult


class WindowedDriftDetector:
    """Detects drift at the distribution level using a sliding window of responses.

    Rather than checking each response independently, accumulates the last
    ``window_size`` responses and checks whether their collective embedding
    distribution has shifted from the reference corpus.  More robust to
    one-off anomalies than per-response detection.

    Two complementary signals trigger drift:

    - ``window_centroid_similarity < threshold``: the window's centroid has moved
      away from the reference domain (gradual topic drift).
    - ``drift_fraction > drift_fraction_threshold``: too many individual responses
      in the window are off-topic (burst drift).

    Args:
        corpus: Reference corpus to compare against.
        window_size: Number of responses in the sliding window (default 20).
        drift_fraction_threshold: Fraction of individually-drifted responses that
            triggers a window-level alert (default 0.3).
        embeddings_model: Override the corpus's embedding model.
        on_drift: Optional callback invoked with a ``WindowDriftResult`` whenever
            drift is detected.
    """

    def __init__(
        self,
        corpus: ReferenceCorpus,
        window_size: int = 20,
        drift_fraction_threshold: float = 0.3,
        embeddings_model: Optional[Embeddings] = None,
        on_drift: Optional[Callable[[WindowDriftResult], Any]] = None,
    ) -> None:
        self._corpus = corpus
        self._window_size = window_size
        self._drift_fraction_threshold = drift_fraction_threshold
        self._model = embeddings_model or corpus._model
        self._on_drift = on_drift
        self._detector = DriftDetector(corpus=corpus, embeddings_model=self._model)
        self._window_vecs: deque[np.ndarray] = deque(maxlen=window_size)
        self._window_results: deque[DriftResult] = deque(maxlen=window_size)

    @property
    def window_full(self) -> bool:
        return len(self._window_vecs) == self._window_size

    @property
    def history(self) -> list[DriftResult]:
        """Individual DriftResults currently in the window."""
        return list(self._window_results)

    def update(self, text: str, **metadata: Any) -> WindowDriftResult | None:
        """Add a response to the window.

        Returns a ``WindowDriftResult`` on every call once the window is full,
        ``None`` while the window is still filling up.
        """
        vec = np.array(self._model.embed_query(text))
        result = self._detector._evaluate(vec, text, metadata)
        self._window_vecs.append(vec)
        self._window_results.append(result)
        if not self.window_full:
            return None
        return self._fire()

    async def aupdate(self, text: str, **metadata: Any) -> WindowDriftResult | None:
        """Async variant of update."""
        vec = np.array(await self._model.aembed_query(text))
        result = self._detector._evaluate(vec, text, metadata)
        self._window_vecs.append(vec)
        self._window_results.append(result)
        if not self.window_full:
            return None
        return await self._afire()

    def _evaluate(self) -> WindowDriftResult:
        window_vecs = np.array(list(self._window_vecs))
        window_centroid = compute_centroid(window_vecs)
        ref_centroid, threshold = self._corpus.cluster_for(window_centroid)
        window_centroid_sim = cosine_similarity(window_centroid, ref_centroid)
        drift_fraction = sum(r.is_drift for r in self._window_results) / len(self._window_results)
        is_drift = (
            window_centroid_sim < threshold
            or drift_fraction > self._drift_fraction_threshold
        )
        return WindowDriftResult(
            is_drift=is_drift,
            window_centroid_similarity=window_centroid_sim,
            drift_fraction=drift_fraction,
            window_size=len(self._window_vecs),
            threshold=threshold,
            drift_fraction_threshold=self._drift_fraction_threshold,
        )

    def _fire(self) -> WindowDriftResult:
        result = self._evaluate()
        if result.is_drift and self._on_drift:
            self._on_drift(result)
        return result

    async def _afire(self) -> WindowDriftResult:
        result = self._evaluate()
        if result.is_drift and self._on_drift:
            ret = self._on_drift(result)
            if hasattr(ret, "__await__"):
                await ret
        return result

    def reset(self) -> None:
        """Clear the window."""
        self._window_vecs.clear()
        self._window_results.clear()
