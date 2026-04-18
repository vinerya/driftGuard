"""End-to-end integration test: full pipeline with FakeEmbeddings."""

import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from driftguard import (
    AlertManager,
    DriftCallbackHandler,
    DriftDetector,
    DriftError,
    DriftRunnable,
    ReferenceCorpus,
)
from tests.conftest import FakeEmbeddings


def _build_legal_system():
    """Build a complete drift detection system for legal domain."""
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    mapping = {
        # Reference corpus — wider spread so threshold is reasonable
        "tort law elements": (legal_base + np.array([0.1, 0.1, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract formation": (legal_base + np.array([0.2, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence standard": (legal_base + np.array([0, 0.2, 0, 0, 0, 0, 0, 0])).tolist(),
        "criminal intent": (legal_base + np.array([0.1, 0.2, 0, 0, 0, 0, 0, 0])).tolist(),
        "due process rights": (legal_base + np.array([0.2, 0.1, 0, 0, 0, 0, 0, 0])).tolist(),
        # On-topic queries — within the same cluster
        "habeas corpus": (legal_base + np.array([0.15, 0.1, 0, 0, 0, 0, 0, 0])).tolist(),
        "summary judgment": (legal_base + np.array([0.1, 0.15, 0, 0, 0, 0, 0, 0])).tolist(),
        # Off-topic queries
        "best pasta recipe": np.array([0, 0, 0, 0, 1.0, 0, 0, 0]).tolist(),
        "weather in paris": np.array([0, 0, 0, 0, 0, 1.0, 0, 0]).tolist(),
        "python programming": np.array([0, 0, 0, 0, 0, 0, 1.0, 0]).tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts([
        "tort law elements",
        "contract formation",
        "negligence standard",
        "criminal intent",
        "due process rights",
    ])
    return emb, corpus


def test_full_pipeline_callback():
    """Test callback handler detects drift correctly across multiple responses."""
    _, corpus = _build_legal_system()
    detector = DriftDetector(corpus=corpus)

    drift_alerts = []
    alerts = AlertManager(sinks=[lambda r: drift_alerts.append(r)])
    handler = DriftCallbackHandler(detector=detector, alerts=alerts)

    # Simulate LLM responses
    for text, should_drift in [
        ("habeas corpus", False),
        ("summary judgment", False),
        ("best pasta recipe", True),
        ("weather in paris", True),
        ("python programming", True),
    ]:
        msg = AIMessage(content=text)
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        handler.on_llm_end(result, run_id=uuid4())

    assert len(handler.history) == 5
    assert not handler.history[0].is_drift  # habeas corpus
    assert not handler.history[1].is_drift  # summary judgment
    assert handler.history[2].is_drift  # pasta recipe
    assert handler.history[3].is_drift  # weather
    assert handler.history[4].is_drift  # programming
    assert len(drift_alerts) == 3


def test_full_pipeline_runnable_guard():
    """Test runnable guard blocks off-topic and passes on-topic."""
    _, corpus = _build_legal_system()
    detector = DriftDetector(corpus=corpus)
    drift = DriftRunnable(detector=detector)
    guard = drift.as_guard()

    # On-topic passes through
    result = guard.invoke("habeas corpus")
    assert result == "habeas corpus"

    # Off-topic raises
    import pytest
    with pytest.raises(DriftError):
        guard.invoke("best pasta recipe")


def test_corpus_save_load_round_trip():
    """Test that a saved corpus produces identical detection results."""
    _, corpus = _build_legal_system()
    detector1 = DriftDetector(corpus=corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legal.npz"
        corpus.save(path)

        # Use same embedding mapping so query vectors match
        emb, _ = _build_legal_system()
        loaded_corpus = ReferenceCorpus(embeddings_model=emb)
        loaded_corpus.load(path)

        detector2 = DriftDetector(corpus=loaded_corpus)

        r1 = detector1.check("habeas corpus")
        r2 = detector2.check("habeas corpus")

        assert r1.is_drift == r2.is_drift
        assert abs(r1.centroid_similarity - r2.centroid_similarity) < 1e-9
        assert abs(r1.threshold - r2.threshold) < 1e-9
