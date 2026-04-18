from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from langchain_core.embeddings import Embeddings

from driftguard._math import (
    compute_adaptive_threshold,
    compute_centroid,
    compute_nn_threshold,
    cosine_similarity,
    farthest_point_sampling,
    kmeans,
)


class ReferenceCorpus:
    """Manages reference embeddings for drift detection.

    Stores embedding vectors, computes centroid and adaptive threshold.
    Can be seeded from text strings and persisted to disk.

    Args:
        embeddings_model: LangChain-compatible embeddings.
        threshold_percentile: Percentile of within-corpus similarities used as
            the drift threshold (default 5 → ~95% of reference texts pass).
        n_clusters: When set, k-means clusters the corpus into this many groups.
            Detection uses the nearest cluster's centroid and threshold rather
            than the global ones, reducing false positives on multi-topic corpora.
    """

    def __init__(
        self,
        embeddings_model: Embeddings,
        threshold_percentile: float = 5.0,
        n_clusters: int | None = None,
    ) -> None:
        self._model = embeddings_model
        self._percentile = threshold_percentile
        self._n_clusters = n_clusters
        self._embeddings: Optional[np.ndarray] = None
        self._texts: list[str] = []
        self._centroid: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
        self._nn_threshold: Optional[float] = None
        self._cluster_centroids: Optional[np.ndarray] = None
        self._cluster_thresholds: Optional[list[float]] = None

    def add_texts(self, texts: Sequence[str]) -> None:
        """Embed texts and add to the reference corpus."""
        new_vecs = np.array(self._model.embed_documents(list(texts)))
        self._texts.extend(texts)
        if self._embeddings is None:
            self._embeddings = new_vecs
        else:
            self._embeddings = np.vstack([self._embeddings, new_vecs])
        self._recompute()

    async def aadd_texts(self, texts: Sequence[str]) -> None:
        """Async variant of add_texts."""
        new_vecs = np.array(await self._model.aembed_documents(list(texts)))
        self._texts.extend(texts)
        if self._embeddings is None:
            self._embeddings = new_vecs
        else:
            self._embeddings = np.vstack([self._embeddings, new_vecs])
        self._recompute()

    def _recompute(self) -> None:
        if self._embeddings is None or len(self._embeddings) < 1:
            self._centroid = None
            self._threshold = None
            self._nn_threshold = None
            self._cluster_centroids = None
            self._cluster_thresholds = None
            return

        self._centroid = compute_centroid(self._embeddings)

        if len(self._embeddings) >= 2:
            self._threshold = compute_adaptive_threshold(
                self._embeddings, self._centroid, self._percentile
            )
            self._nn_threshold = compute_nn_threshold(self._embeddings, self._percentile)
            self._recompute_clusters()

    def _recompute_clusters(self) -> None:
        if self._n_clusters is None or self._embeddings is None:
            self._cluster_centroids = None
            self._cluster_thresholds = None
            return

        k = min(self._n_clusters, len(self._embeddings))
        centroids, labels = kmeans(self._embeddings, k)
        self._cluster_centroids = centroids
        self._cluster_thresholds = []
        for j in range(k):
            cluster_vecs = self._embeddings[labels == j]
            if len(cluster_vecs) >= 2:
                self._cluster_thresholds.append(
                    compute_adaptive_threshold(cluster_vecs, centroids[j], self._percentile)
                )
            else:
                # Single-member cluster: fall back to global threshold
                self._cluster_thresholds.append(self._threshold or 0.0)

    def cluster_for(self, query_vec: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the (centroid, threshold) of the cluster nearest to query_vec.

        Falls back to (global centroid, global threshold) when clustering is off.
        """
        if self._cluster_centroids is None:
            return self.centroid, self.threshold
        sims = [cosine_similarity(query_vec, c) for c in self._cluster_centroids]
        best = int(np.argmax(sims))
        return self._cluster_centroids[best], self._cluster_thresholds[best]  # type: ignore[index]

    @property
    def centroid(self) -> np.ndarray:
        if self._centroid is None:
            raise ValueError("Corpus is empty. Call add_texts() first.")
        return self._centroid

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise ValueError("Need at least 2 reference texts to compute threshold.")
        return self._threshold

    @property
    def nn_threshold(self) -> float:
        if self._nn_threshold is None:
            raise ValueError("Need at least 2 reference texts to compute nn_threshold.")
        return self._nn_threshold

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            raise ValueError("Corpus is empty.")
        return self._embeddings

    @property
    def size(self) -> int:
        return 0 if self._embeddings is None else len(self._embeddings)

    def save(self, path: str | Path) -> None:
        """Save corpus to a .npz file with JSON sidecar for texts."""
        path = Path(path)
        arrays: dict = dict(
            embeddings=self._embeddings,
            centroid=self._centroid,
            threshold=np.array([self._threshold or 0.0]),
            percentile=np.array([self._percentile]),
            nn_threshold=np.array([self._nn_threshold or 0.0]),
            n_clusters=np.array([self._n_clusters or 0]),
        )
        if self._cluster_centroids is not None:
            arrays["cluster_centroids"] = self._cluster_centroids
            arrays["cluster_thresholds"] = np.array(self._cluster_thresholds)
        np.savez_compressed(path, **arrays)
        texts_path = path.with_suffix(".texts.json")
        texts_path.write_text(json.dumps(self._texts))

    @classmethod
    def from_texts(
        cls,
        candidates: Sequence[str],
        embeddings_model: Embeddings,
        n: int = 20,
        threshold_percentile: float = 5.0,
        n_clusters: int | None = None,
        seed: int = 42,
    ) -> "ReferenceCorpus":
        """Build a corpus by selecting n maximally diverse texts from a larger pool.

        Uses Farthest Point Sampling on cosine distance to maximise coverage of the
        target domain without manual curation.
        """
        candidates = list(candidates)
        all_vecs = np.array(embeddings_model.embed_documents(candidates))
        indices = farthest_point_sampling(all_vecs, n=min(n, len(candidates)), seed=seed)
        corpus = cls(
            embeddings_model=embeddings_model,
            threshold_percentile=threshold_percentile,
            n_clusters=n_clusters,
        )
        corpus._texts = [candidates[i] for i in indices]
        corpus._embeddings = all_vecs[indices]
        corpus._recompute()
        return corpus

    @classmethod
    async def afrom_texts(
        cls,
        candidates: Sequence[str],
        embeddings_model: Embeddings,
        n: int = 20,
        threshold_percentile: float = 5.0,
        n_clusters: int | None = None,
        seed: int = 42,
    ) -> "ReferenceCorpus":
        """Async variant of from_texts."""
        candidates = list(candidates)
        all_vecs = np.array(await embeddings_model.aembed_documents(candidates))
        indices = farthest_point_sampling(all_vecs, n=min(n, len(candidates)), seed=seed)
        corpus = cls(
            embeddings_model=embeddings_model,
            threshold_percentile=threshold_percentile,
            n_clusters=n_clusters,
        )
        corpus._texts = [candidates[i] for i in indices]
        corpus._embeddings = all_vecs[indices]
        corpus._recompute()
        return corpus

    def compare(
        self,
        other: "ReferenceCorpus",
        significant_shift_threshold: float = 0.05,
    ) -> "CorpusComparison":
        """Compare this corpus to another and return a CorpusComparison.

        Useful for detecting domain shift between prompt versions, model
        upgrades, or dataset changes.

        Args:
            other: The corpus to compare against (e.g. a newer version).
            significant_shift_threshold: Cosine distance above which the shift
                is considered significant (default 0.05).
        """
        from driftguard._math import cosine_similarity
        from driftguard.auditor import CorpusComparison

        centroid_shift = 1.0 - cosine_similarity(self.centroid, other.centroid)
        threshold_delta = other.threshold - self.threshold
        nn_delta = (
            (other._nn_threshold or 0.0) - (self._nn_threshold or 0.0)
        )
        return CorpusComparison(
            centroid_shift=float(centroid_shift),
            threshold_delta=float(threshold_delta),
            nn_threshold_delta=float(nn_delta),
            size_delta=other.size - self.size,
            is_significant=centroid_shift > significant_shift_threshold,
        )

    def plot(
        self,
        check_texts: list[str] | None = None,
        title: str | None = None,
        ax=None,
    ):
        """Plot the corpus in 2D using t-SNE. Requires ``langchain-drift[viz]``."""
        from driftguard.viz import plot_corpus
        return plot_corpus(self, check_texts=check_texts, title=title, ax=ax)

    def load(self, path: str | Path) -> None:
        """Load corpus from a .npz file."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")
        data = np.load(path)
        self._embeddings = data["embeddings"]
        self._centroid = data["centroid"]
        self._threshold = float(data["threshold"][0])
        self._percentile = float(data["percentile"][0])
        self._nn_threshold = float(data["nn_threshold"][0]) if "nn_threshold" in data else None
        nc = int(data["n_clusters"][0]) if "n_clusters" in data else 0
        self._n_clusters = nc if nc > 0 else None
        if "cluster_centroids" in data:
            self._cluster_centroids = data["cluster_centroids"]
            self._cluster_thresholds = list(data["cluster_thresholds"])
        else:
            self._cluster_centroids = None
            self._cluster_thresholds = None
        texts_path = path.with_suffix(".texts.json")
        if texts_path.exists():
            self._texts = json.loads(texts_path.read_text())
