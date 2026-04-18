import numpy as np
import pytest

from driftguard._math import farthest_point_sampling
from driftguard.corpus import ReferenceCorpus
from tests.conftest import FakeEmbeddings


def _orthogonal_pool(n: int) -> np.ndarray:
    """n orthogonal unit vectors (n <= n)."""
    return np.eye(n)


def test_fps_returns_n_indices():
    embs = np.random.RandomState(0).randn(20, 8)
    indices = farthest_point_sampling(embs, n=5)
    assert len(indices) == 5


def test_fps_clamps_to_corpus_size():
    embs = np.eye(3)
    indices = farthest_point_sampling(embs, n=10)
    assert len(indices) == 3


def test_fps_no_duplicates():
    embs = np.random.RandomState(1).randn(30, 8)
    indices = farthest_point_sampling(embs, n=10)
    assert len(set(indices.tolist())) == 10


def test_fps_maximises_diversity():
    # 4 orthogonal axes + many noisy copies of axis 0.
    # FPS should pick one from each axis cluster, not 5 from axis 0.
    rng = np.random.RandomState(42)
    cluster_a = np.array([[1, 0, 0, 0]] * 10, dtype=float) + rng.randn(10, 4) * 0.01
    cluster_b = np.array([[0, 1, 0, 0]] * 2, dtype=float) + rng.randn(2, 4) * 0.01
    cluster_c = np.array([[0, 0, 1, 0]] * 2, dtype=float) + rng.randn(2, 4) * 0.01
    cluster_d = np.array([[0, 0, 0, 1]] * 2, dtype=float) + rng.randn(2, 4) * 0.01
    pool = np.vstack([cluster_a, cluster_b, cluster_c, cluster_d])

    indices = farthest_point_sampling(pool, n=4, seed=0)
    selected = pool[indices]
    # Selected vectors should span all 4 axes — verify by checking that the
    # maximum entry of each selected vector covers all 4 dimensions.
    dominant_dims = set(int(np.argmax(np.abs(v))) for v in selected)
    assert dominant_dims == {0, 1, 2, 3}


def test_from_texts_selects_n():
    mapping = {f"text_{i}": np.eye(8)[i % 8].tolist() for i in range(20)}
    emb = FakeEmbeddings(mapping=mapping, dim=8)
    candidates = list(mapping.keys())

    corpus = ReferenceCorpus.from_texts(candidates, embeddings_model=emb, n=5)
    assert corpus.size == 5
    assert len(corpus._texts) == 5
    assert corpus.threshold is not None


def test_from_texts_subset_of_candidates():
    mapping = {f"text_{i}": np.eye(8)[i % 8].tolist() for i in range(20)}
    emb = FakeEmbeddings(mapping=mapping, dim=8)
    candidates = list(mapping.keys())

    corpus = ReferenceCorpus.from_texts(candidates, embeddings_model=emb, n=6)
    for t in corpus._texts:
        assert t in candidates


async def test_afrom_texts():
    mapping = {f"text_{i}": np.eye(8)[i % 8].tolist() for i in range(16)}
    emb = FakeEmbeddings(mapping=mapping, dim=8)
    candidates = list(mapping.keys())

    corpus = await ReferenceCorpus.afrom_texts(candidates, embeddings_model=emb, n=4)
    assert corpus.size == 4


def test_from_texts_with_clustering():
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    medical_base = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
    rng = np.random.RandomState(7)
    mapping = {}
    for i in range(10):
        v = (legal_base + rng.randn(8) * 0.05).tolist()
        mapping[f"legal_{i}"] = v
    for i in range(10):
        v = (medical_base + rng.randn(8) * 0.05).tolist()
        mapping[f"medical_{i}"] = v

    emb = FakeEmbeddings(mapping=mapping, dim=8)
    candidates = list(mapping.keys())

    corpus = ReferenceCorpus.from_texts(
        candidates, embeddings_model=emb, n=6, n_clusters=2
    )
    assert corpus.size == 6
    assert corpus._cluster_centroids is not None
