from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def compute_centroid(embeddings: NDArray) -> NDArray:
    """Mean centroid of embedding matrix (rows = vectors)."""
    return embeddings.mean(axis=0)


def compute_adaptive_threshold(
    embeddings: NDArray,
    centroid: NDArray,
    percentile: float = 5.0,
) -> float:
    """Adaptive threshold from within-corpus similarity distribution.

    Computes cosine similarity of every reference embedding to the centroid,
    then returns the given percentile. A percentile of 5.0 means ~95% of
    reference embeddings score above the threshold.
    """
    sims = np.array([cosine_similarity(e, centroid) for e in embeddings])
    return float(np.percentile(sims, percentile))


def max_similarity_to_set(query: NDArray, corpus: NDArray) -> float:
    """Max cosine similarity between query vector and any vector in corpus."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    corpus_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    sims = corpus_norms @ query_norm
    return float(np.max(sims))


def compute_nn_threshold(embeddings: NDArray, percentile: float = 5.0) -> float:
    """Percentile of leave-one-out nearest-neighbour cosine similarity within corpus.

    For each reference text, finds its closest other reference; returns the given
    percentile of that distribution. Used as the NN drift threshold: a query whose
    max-similarity to the corpus falls below this is far from every reference.
    """
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    sim_matrix = normed @ normed.T  # (n, n)
    np.fill_diagonal(sim_matrix, -np.inf)  # exclude self-similarity
    max_sims = sim_matrix.max(axis=1)
    return float(np.percentile(max_sims, percentile))


def farthest_point_sampling(
    embeddings: NDArray, n: int, seed: int = 42
) -> NDArray:
    """Select n maximally diverse indices from embeddings using Farthest Point Sampling.

    Iteratively picks the point with the greatest cosine distance from all
    already-selected points.  Returns an integer index array of length min(n, len).
    """
    n = min(n, len(embeddings))
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(len(embeddings)))]
    min_dists = np.full(len(embeddings), np.inf)

    for _ in range(n - 1):
        last_norm = normed[selected[-1]]
        dists = 1.0 - (normed @ last_norm)          # cosine distance to last selected
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected[-1]] = -np.inf            # don't re-select
        selected.append(int(np.argmax(min_dists)))

    return np.array(selected, dtype=int)


def kmeans(
    embeddings: NDArray,
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """k-means clustering using cosine similarity.

    Returns (centroids [k × d], labels [n]).  k is clamped to len(embeddings).
    Empty clusters keep their previous centroid rather than collapsing.
    """
    k = min(k, len(embeddings))
    rng = np.random.RandomState(seed)
    centroids = embeddings[rng.choice(len(embeddings), k, replace=False)].copy().astype(float)
    labels = np.zeros(len(embeddings), dtype=int)

    for _ in range(max_iter):
        emb_n = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        cen_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        new_labels = np.argmax(emb_n @ cen_n.T, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = embeddings[mask].mean(axis=0)

    return centroids, labels
