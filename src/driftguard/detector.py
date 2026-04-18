from __future__ import annotations

from typing import Any, Optional

import numpy as np
from langchain_core.embeddings import Embeddings

from driftguard._math import cosine_similarity, max_similarity_to_set
from driftguard.corpus import ReferenceCorpus
from driftguard.schema import DriftResult


class DriftDetector:
    """Core drift detection engine.

    Embeds a text, compares to the reference corpus, returns a DriftResult.

    Drift is flagged when BOTH conditions hold:
    1. centroid_similarity < threshold  (cluster-aware if n_clusters is set)
    2. max_reference_similarity < nn_threshold  (nearest-neighbour check)

    Condition 2 acts as a rescue: if a response is close to any reference text
    it is not drift, even if the global centroid distance is high (e.g. paraphrases
    of a reference that happen to sit far from the centroid).
    """

    def __init__(
        self,
        corpus: ReferenceCorpus,
        embeddings_model: Optional[Embeddings] = None,
    ) -> None:
        self._corpus = corpus
        self._model = embeddings_model or corpus._model

    def check(self, text: str, **metadata: Any) -> DriftResult:
        """Check a text for drift against the reference corpus."""
        query_vec = np.array(self._model.embed_query(text))
        return self._evaluate(query_vec, text, metadata)

    async def acheck(self, text: str, **metadata: Any) -> DriftResult:
        """Async variant of check."""
        query_vec = np.array(await self._model.aembed_query(text))
        return self._evaluate(query_vec, text, metadata)

    def _evaluate(
        self, query_vec: np.ndarray, text: str, metadata: dict
    ) -> DriftResult:
        # Option 1: cluster-aware centroid and threshold
        centroid, threshold = self._corpus.cluster_for(query_vec)
        centroid_sim = cosine_similarity(query_vec, centroid)

        max_ref_sim = max_similarity_to_set(query_vec, self._corpus.embeddings)

        nn_thr = self._corpus._nn_threshold
        if nn_thr is not None:
            # Option 2: both signals must agree — reduces paraphrase false positives
            is_drift = centroid_sim < threshold and max_ref_sim < nn_thr
        else:
            is_drift = centroid_sim < threshold

        return DriftResult(
            is_drift=is_drift,
            centroid_similarity=centroid_sim,
            max_reference_similarity=max_ref_sim,
            threshold=threshold,
            nn_threshold=nn_thr if nn_thr is not None else 0.0,
            text=text,
            metadata=metadata,
        )
