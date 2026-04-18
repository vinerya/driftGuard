import numpy as np

from driftguard.corpus import ReferenceCorpus
from driftguard.detector import DriftDetector
from tests.conftest import FakeEmbeddings


def _make_legal_embeddings():
    """Create embeddings where legal texts cluster together
    and off-topic texts are far away."""
    legal_base = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cooking_base = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    mapping = {
        "tort law": (legal_base + np.array([0.01, 0.02, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.02, 0.01, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence": (legal_base + np.array([0.0, 0.03, 0, 0, 0, 0, 0, 0])).tolist(),
        "criminal law": (legal_base + np.array([0.03, 0.0, 0, 0, 0, 0, 0, 0])).tolist(),
        "statute of limitations": (legal_base + np.array([0.01, 0.01, 0.01, 0, 0, 0, 0, 0])).tolist(),
        # On-topic query
        "habeas corpus": (legal_base + np.array([0.02, 0.02, 0, 0, 0, 0, 0, 0])).tolist(),
        # Off-topic
        "chocolate cake recipe": cooking_base.tolist(),
        "weather forecast": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).tolist(),
    }
    return FakeEmbeddings(mapping=mapping)


def test_on_topic_no_drift():
    emb = _make_legal_embeddings()
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence", "criminal law", "statute of limitations"])

    detector = DriftDetector(corpus=corpus)
    result = detector.check("habeas corpus")
    assert not result.is_drift
    assert result.centroid_similarity > result.threshold


def test_off_topic_drift():
    emb = _make_legal_embeddings()
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence", "criminal law", "statute of limitations"])

    detector = DriftDetector(corpus=corpus)
    result = detector.check("chocolate cake recipe")
    assert result.is_drift
    assert result.centroid_similarity < result.threshold


def test_off_topic_drift_weather():
    emb = _make_legal_embeddings()
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence", "criminal law", "statute of limitations"])

    detector = DriftDetector(corpus=corpus)
    result = detector.check("weather forecast")
    assert result.is_drift


def test_metadata_passthrough():
    emb = _make_legal_embeddings()
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law"])

    detector = DriftDetector(corpus=corpus)
    result = detector.check("tort law", run_id="abc-123")
    assert result.metadata == {"run_id": "abc-123"}


def test_nn_threshold_in_result():
    emb = _make_legal_embeddings()
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence"])
    detector = DriftDetector(corpus=corpus)
    result = detector.check("habeas corpus")
    assert result.nn_threshold > 0


def test_clustering_multimodal_corpus():
    """With a two-topic corpus, clustering avoids false positives near either topic."""
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    medical_base = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
    food_base = np.array([0, 0, 1.0, 0, 0, 0, 0, 0])

    mapping = {
        # Legal cluster
        "tort law": (legal_base + np.array([0.05, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.10, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence": (legal_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        # Medical cluster
        "malpractice": (medical_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        "diagnosis": (medical_base + np.array([0, 0.10, 0, 0, 0, 0, 0, 0])).tolist(),
        "prognosis": (medical_base + np.array([0, 0, 0.05, 0, 0, 0, 0, 0])).tolist(),
        # On-topic queries
        "habeas corpus": (legal_base + np.array([0.08, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "clinical trial": (medical_base + np.array([0, 0.08, 0, 0, 0, 0, 0, 0])).tolist(),
        # Off-topic
        "pasta recipe": food_base.tolist(),
    }

    emb = _make_legal_embeddings().__class__(mapping=mapping)
    from tests.conftest import FakeEmbeddings
    emb = FakeEmbeddings(mapping=mapping)

    corpus = ReferenceCorpus(embeddings_model=emb, n_clusters=2)
    corpus.add_texts([
        "tort law", "contract law", "negligence",
        "malpractice", "diagnosis", "prognosis",
    ])
    detector = DriftDetector(corpus=corpus)

    # Legal query should not be flagged as drift
    assert not detector.check("habeas corpus").is_drift

    # Medical query should not be flagged as drift
    assert not detector.check("clinical trial").is_drift

    # Off-topic should be flagged
    assert detector.check("pasta recipe").is_drift
