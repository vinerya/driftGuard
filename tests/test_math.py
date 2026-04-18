import numpy as np

from driftguard._math import (
    compute_adaptive_threshold,
    compute_centroid,
    compute_nn_threshold,
    cosine_similarity,
    kmeans,
    max_similarity_to_set,
)


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9


def test_cosine_similarity_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert cosine_similarity(a, b) == 0.0


def test_compute_centroid():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    centroid = compute_centroid(embs)
    np.testing.assert_array_almost_equal(centroid, [0.5, 0.5])


def test_compute_adaptive_threshold():
    # Create a tight cluster — threshold should be high
    rng = np.random.RandomState(42)
    base = np.array([1.0, 0.0, 0.0, 0.0])
    embs = np.array([base + rng.randn(4) * 0.01 for _ in range(50)])
    centroid = compute_centroid(embs)
    threshold = compute_adaptive_threshold(embs, centroid, percentile=5.0)
    # Tight cluster → threshold close to 1.0
    assert threshold > 0.9


def test_max_similarity_to_set():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    query = np.array([1.0, 0.0])
    sim = max_similarity_to_set(query, corpus)
    assert abs(sim - 1.0) < 1e-6  # exact match with first vector


def test_max_similarity_no_match():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    query = np.array([-1.0, -1.0])
    sim = max_similarity_to_set(query, corpus)
    assert sim < 0.0  # opposite direction


def test_compute_nn_threshold_tight_cluster():
    # All vectors nearly identical → each has a near-1.0 neighbour → high threshold
    rng = np.random.RandomState(0)
    base = np.array([1.0, 0.0, 0.0, 0.0])
    embs = np.array([base + rng.randn(4) * 0.01 for _ in range(20)])
    thr = compute_nn_threshold(embs, percentile=5.0)
    assert thr > 0.95


def test_compute_nn_threshold_orthogonal():
    # Two perfectly orthogonal groups → NN sim within groups is 0 → low threshold
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    embs = np.array([a, a * 0.99, b, b * 0.99])
    thr = compute_nn_threshold(embs, percentile=5.0)
    # Each point's nearest neighbour is in the same axis cluster (sim ≈ 1.0), so
    # threshold should still be high — the two clusters are internally tight.
    assert thr > 0.98


def test_kmeans_separates_two_clusters():
    # Points clearly on x-axis vs y-axis
    cluster_a = np.array([[1.0, 0.0], [0.95, 0.05], [0.98, 0.02]])
    cluster_b = np.array([[0.0, 1.0], [0.05, 0.95], [0.02, 0.98]])
    embs = np.vstack([cluster_a, cluster_b])
    centroids, labels = kmeans(embs, k=2, seed=0)
    assert centroids.shape == (2, 2)
    # Each cluster should contain exactly 3 points
    counts = [int((labels == j).sum()) for j in range(2)]
    assert sorted(counts) == [3, 3]


def test_kmeans_k1_returns_all_in_one_cluster():
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    centroids, labels = kmeans(embs, k=1)
    assert centroids.shape == (1, 2)
    assert np.all(labels == 0)


def test_kmeans_clamps_k_to_n():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    centroids, labels = kmeans(embs, k=10)
    # k is clamped to len(embs) = 2
    assert centroids.shape[0] <= 2
