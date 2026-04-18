"""Tests for the LangGraph integration module.

These tests exercise the returned callables directly — no LangGraph installation
required.  The node functions are plain callables; LangGraph just calls them.
"""
import numpy as np
import pytest

from driftguard import DriftDetector, DriftResult, ReferenceCorpus
from driftguard.langgraph import (
    adrift_node,
    drift_node,
    make_route_on_drift,
    route_on_drift,
)
from tests.conftest import FakeEmbeddings


def _make_detector() -> DriftDetector:
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    food_base = np.array([0, 0, 0, 0, 1.0, 0, 0, 0])
    mapping = {
        "tort law": (legal_base + np.array([0.05, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.10, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence": (legal_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        "habeas corpus": (legal_base + np.array([0.08, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "pasta recipe": food_base.tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence"])
    return DriftDetector(corpus=corpus)


# --- drift_node (sync) ---

def test_drift_node_on_topic():
    node = drift_node(_make_detector())
    update = node({"response": "habeas corpus"})
    assert "drift" in update
    assert not update["drift"].is_drift


def test_drift_node_off_topic():
    node = drift_node(_make_detector())
    update = node({"response": "pasta recipe"})
    assert update["drift"].is_drift


def test_drift_node_custom_keys():
    node = drift_node(_make_detector(), text_key="output", result_key="drift_result")
    update = node({"output": "pasta recipe"})
    assert "drift_result" in update
    assert update["drift_result"].is_drift


def test_drift_node_returns_drift_result_type():
    node = drift_node(_make_detector())
    update = node({"response": "habeas corpus"})
    assert isinstance(update["drift"], DriftResult)


# --- adrift_node (async) ---

async def test_adrift_node_on_topic():
    node = adrift_node(_make_detector())
    update = await node({"response": "habeas corpus"})
    assert not update["drift"].is_drift


async def test_adrift_node_off_topic():
    node = adrift_node(_make_detector())
    update = await node({"response": "pasta recipe"})
    assert update["drift"].is_drift


# --- route_on_drift ---

def _make_result(is_drift: bool) -> DriftResult:
    return DriftResult(
        is_drift=is_drift,
        centroid_similarity=0.3 if is_drift else 0.9,
        max_reference_similarity=0.4 if is_drift else 0.95,
        threshold=0.8,
        nn_threshold=0.75,
        text="test",
    )


def test_route_on_drift_returns_ok_when_clean():
    assert route_on_drift({"drift": _make_result(False)}) == "ok"


def test_route_on_drift_returns_drift_when_drifting():
    assert route_on_drift({"drift": _make_result(True)}) == "drift"


def test_route_on_drift_missing_key_is_ok():
    assert route_on_drift({}) == "ok"


# --- make_route_on_drift ---

def test_make_route_on_drift_custom_labels():
    router = make_route_on_drift(on_drift="blocked", on_ok="pass")
    assert router({"drift": _make_result(True)}) == "blocked"
    assert router({"drift": _make_result(False)}) == "pass"


def test_make_route_on_drift_custom_result_key():
    router = make_route_on_drift(result_key="guard")
    assert router({"guard": _make_result(True)}) == "drift"
    assert router({"guard": _make_result(False)}) == "ok"
    assert router({}) == "ok"  # missing key → ok
