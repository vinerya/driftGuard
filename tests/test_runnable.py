import numpy as np
import pytest

from driftguard.corpus import ReferenceCorpus
from driftguard.detector import DriftDetector
from driftguard.runnable import DriftRunnable
from driftguard.schema import DriftError
from tests.conftest import FakeEmbeddings


def _make_drift_runnable():
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    mapping = {
        "legal one": (legal_base + np.array([0.01, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "legal two": (legal_base + np.array([0.02, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "on topic law": (legal_base + np.array([0.03, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "off topic cooking": np.array([0, 0, 0, 0, 1.0, 0, 0, 0]).tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["legal one", "legal two"])
    detector = DriftDetector(corpus=corpus)
    return DriftRunnable(detector=detector)


def test_passthrough_on_topic():
    drift = _make_drift_runnable()
    runnable = drift.as_passthrough()
    result = runnable.invoke("on topic law")
    assert result["output"] == "on topic law"
    assert not result["drift"].is_drift


def test_passthrough_drift():
    drift = _make_drift_runnable()
    runnable = drift.as_passthrough()
    result = runnable.invoke("off topic cooking")
    assert result["output"] == "off topic cooking"
    assert result["drift"].is_drift


def test_guard_on_topic():
    drift = _make_drift_runnable()
    guard = drift.as_guard()
    result = guard.invoke("on topic law")
    assert result == "on topic law"


def test_guard_drift_raises():
    drift = _make_drift_runnable()
    guard = drift.as_guard()
    with pytest.raises(DriftError) as exc_info:
        guard.invoke("off topic cooking")
    assert exc_info.value.result.is_drift
    assert exc_info.value.result.centroid_similarity < exc_info.value.result.threshold
