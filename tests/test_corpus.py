import tempfile
from pathlib import Path

import numpy as np
import pytest

from driftguard.corpus import ReferenceCorpus
from tests.conftest import FakeEmbeddings


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


def test_empty_corpus(embeddings):
    corpus = ReferenceCorpus(embeddings_model=embeddings)
    assert corpus.size == 0
    with pytest.raises(ValueError, match="empty"):
        _ = corpus.centroid
    with pytest.raises(ValueError, match="empty"):
        _ = corpus.embeddings


def test_add_texts(embeddings):
    corpus = ReferenceCorpus(embeddings_model=embeddings)
    corpus.add_texts(["hello", "world"])
    assert corpus.size == 2
    assert corpus.centroid is not None
    assert corpus.threshold is not None


def test_single_text_no_threshold(embeddings):
    corpus = ReferenceCorpus(embeddings_model=embeddings)
    corpus.add_texts(["only one"])
    assert corpus.size == 1
    assert corpus.centroid is not None
    with pytest.raises(ValueError, match="at least 2"):
        _ = corpus.threshold


def test_incremental_add(embeddings):
    corpus = ReferenceCorpus(embeddings_model=embeddings)
    corpus.add_texts(["a", "b"])
    assert corpus.size == 2
    corpus.add_texts(["c", "d"])
    assert corpus.size == 4


def test_save_load(embeddings):
    corpus = ReferenceCorpus(embeddings_model=embeddings)
    corpus.add_texts(["legal case one", "legal case two", "legal case three"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "corpus.npz"
        corpus.save(path)

        loaded = ReferenceCorpus(embeddings_model=embeddings)
        loaded.load(path)

        assert loaded.size == corpus.size
        assert abs(loaded.threshold - corpus.threshold) < 1e-9
        np.testing.assert_array_almost_equal(loaded.centroid, corpus.centroid)
        np.testing.assert_array_almost_equal(loaded.embeddings, corpus.embeddings)


def test_nn_threshold_computed_after_two_texts(embeddings):
    corpus = ReferenceCorpus(embeddings_model=embeddings)
    corpus.add_texts(["hello", "world"])
    assert corpus.nn_threshold is not None


def test_nn_threshold_not_available_with_one_text(embeddings):
    corpus = ReferenceCorpus(embeddings_model=embeddings)
    corpus.add_texts(["only one"])
    with pytest.raises(ValueError, match="at least 2"):
        _ = corpus.nn_threshold


def test_n_clusters_roundtrip():
    """Clustering corpus saves and loads cluster data correctly."""
    legal_base = np.array([1.0, 0, 0, 0])
    medical_base = np.array([0, 1.0, 0, 0])
    mapping = {
        "tort law": (legal_base + np.array([0.01, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.02, 0, 0, 0])).tolist(),
        "malpractice": (medical_base + np.array([0, 0.01, 0, 0])).tolist(),
        "diagnosis": (medical_base + np.array([0, 0.02, 0, 0])).tolist(),
    }
    from tests.conftest import FakeEmbeddings

    emb = FakeEmbeddings(mapping=mapping, dim=4)
    corpus = ReferenceCorpus(embeddings_model=emb, n_clusters=2)
    corpus.add_texts(["tort law", "contract law", "malpractice", "diagnosis"])

    assert corpus._cluster_centroids is not None
    assert len(corpus._cluster_thresholds) == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "clustered.npz"
        corpus.save(path)

        loaded = ReferenceCorpus(embeddings_model=emb, n_clusters=2)
        loaded.load(path)

        assert loaded._cluster_centroids is not None
        np.testing.assert_array_almost_equal(
            loaded._cluster_centroids, corpus._cluster_centroids
        )


def test_cluster_for_routes_to_nearest_cluster():
    """cluster_for returns the nearest cluster's centroid, not the global one."""
    legal_base = np.array([1.0, 0, 0, 0])
    medical_base = np.array([0, 1.0, 0, 0])
    mapping = {
        "tort law": (legal_base + np.array([0.01, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.02, 0, 0, 0])).tolist(),
        "malpractice": (medical_base + np.array([0, 0.01, 0, 0])).tolist(),
        "diagnosis": (medical_base + np.array([0, 0.02, 0, 0])).tolist(),
        "legal query": (legal_base + np.array([0.015, 0, 0, 0])).tolist(),
    }
    from tests.conftest import FakeEmbeddings

    emb = FakeEmbeddings(mapping=mapping, dim=4)
    corpus = ReferenceCorpus(embeddings_model=emb, n_clusters=2)
    corpus.add_texts(["tort law", "contract law", "malpractice", "diagnosis"])

    legal_vec = np.array(emb.embed_query("legal query"))
    centroid, _ = corpus.cluster_for(legal_vec)

    # The returned centroid should be closer to legal_base than to medical_base
    sim_legal = float(np.dot(centroid / np.linalg.norm(centroid), legal_base))
    sim_medical = float(np.dot(centroid / np.linalg.norm(centroid), medical_base))
    assert sim_legal > sim_medical
