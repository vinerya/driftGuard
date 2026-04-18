from uuid import uuid4

import numpy as np
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage

from driftguard.alerts import AlertManager
from driftguard.callback import DriftCallbackHandler, _extract_text
from driftguard.corpus import ReferenceCorpus
from driftguard.detector import DriftDetector
from tests.conftest import FakeEmbeddings


def _make_llm_result(text: str) -> LLMResult:
    msg = AIMessage(content=text)
    gen = ChatGeneration(message=msg)
    return LLMResult(generations=[[gen]])


def _make_detector():
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    mapping = {
        "legal response": (legal_base + np.array([0.01, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "another legal": (legal_base + np.array([0.02, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "off topic cooking": np.array([0, 0, 0, 0, 1.0, 0, 0, 0]).tolist(),
        "on topic law": (legal_base + np.array([0.03, 0, 0, 0, 0, 0, 0, 0])).tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["legal response", "another legal"])
    return DriftDetector(corpus=corpus)


def test_extract_text_from_chat_generation():
    result = _make_llm_result("hello world")
    assert _extract_text(result) == "hello world"


def test_extract_text_empty():
    result = LLMResult(generations=[[]])
    assert _extract_text(result) is None


def test_callback_on_topic():
    detector = _make_detector()
    handler = DriftCallbackHandler(detector=detector)
    llm_result = _make_llm_result("on topic law")
    handler.on_llm_end(llm_result, run_id=uuid4())
    assert len(handler.history) == 1
    assert not handler.history[0].is_drift


def test_callback_drift_fires_alert():
    detector = _make_detector()
    drift_alerts = []
    alerts = AlertManager(sinks=[lambda r: drift_alerts.append(r)])
    handler = DriftCallbackHandler(detector=detector, alerts=alerts)
    llm_result = _make_llm_result("off topic cooking")
    handler.on_llm_end(llm_result, run_id=uuid4())
    assert len(handler.history) == 1
    assert handler.history[0].is_drift
    assert len(drift_alerts) == 1


def test_callback_on_drift_callback():
    detector = _make_detector()
    custom_calls = []
    handler = DriftCallbackHandler(
        detector=detector,
        on_drift=lambda r: custom_calls.append(r),
    )
    llm_result = _make_llm_result("off topic cooking")
    handler.on_llm_end(llm_result, run_id=uuid4())
    assert len(custom_calls) == 1
