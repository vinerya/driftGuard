import numpy as np
import pytest

from driftguard import DriftDetector, ReferenceCorpus, WindowDriftResult, WindowedDriftDetector
from tests.conftest import FakeEmbeddings


def _make_system():
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    food_base = np.array([0, 0, 0, 0, 1.0, 0, 0, 0])
    mapping = {
        "tort law": (legal_base + np.array([0.05, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.10, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence": (legal_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        "due process": (legal_base + np.array([0.08, 0.03, 0, 0, 0, 0, 0, 0])).tolist(),
        "habeas corpus": (legal_base + np.array([0.07, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "pasta recipe": food_base.tolist(),
        "weather": np.array([0, 0, 0, 0, 0, 1.0, 0, 0]).tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence", "due process"])
    return emb, corpus


def test_windowed_returns_none_while_filling():
    _, corpus = _make_system()
    wd = WindowedDriftDetector(corpus=corpus, window_size=5)
    for i in range(4):
        result = wd.update("habeas corpus")
        assert result is None


def test_windowed_returns_result_when_full():
    _, corpus = _make_system()
    wd = WindowedDriftDetector(corpus=corpus, window_size=3)
    wd.update("habeas corpus")
    wd.update("habeas corpus")
    result = wd.update("habeas corpus")
    assert isinstance(result, WindowDriftResult)


def test_windowed_on_topic_no_drift():
    _, corpus = _make_system()
    wd = WindowedDriftDetector(corpus=corpus, window_size=3)
    for _ in range(3):
        result = wd.update("habeas corpus")
    assert result is not None
    assert not result.is_drift
    assert result.drift_fraction == 0.0


def test_windowed_off_topic_drift_fraction():
    _, corpus = _make_system()
    # drift_fraction_threshold=0.3 → 2/3 drifted responses should trigger
    wd = WindowedDriftDetector(corpus=corpus, window_size=3, drift_fraction_threshold=0.3)
    wd.update("habeas corpus")   # on-topic
    wd.update("pasta recipe")    # drift
    result = wd.update("weather")  # drift  → 2/3 = 0.67 > 0.3
    assert result is not None
    assert result.is_drift
    assert result.drift_fraction > 0.3


def test_windowed_centroid_shift():
    _, corpus = _make_system()
    # All off-topic → window centroid far from reference → centroid signal fires
    wd = WindowedDriftDetector(corpus=corpus, window_size=3, drift_fraction_threshold=0.99)
    wd.update("pasta recipe")
    wd.update("pasta recipe")
    result = wd.update("pasta recipe")
    assert result is not None
    assert result.is_drift
    assert result.window_centroid_similarity < result.threshold


def test_windowed_on_drift_callback():
    _, corpus = _make_system()
    fired = []
    wd = WindowedDriftDetector(
        corpus=corpus, window_size=3,
        drift_fraction_threshold=0.3,
        on_drift=lambda r: fired.append(r),
    )
    wd.update("pasta recipe")
    wd.update("pasta recipe")
    wd.update("pasta recipe")
    assert len(fired) == 1
    assert fired[0].is_drift


def test_windowed_reset():
    _, corpus = _make_system()
    wd = WindowedDriftDetector(corpus=corpus, window_size=3)
    for _ in range(3):
        wd.update("habeas corpus")
    wd.reset()
    assert not wd.window_full
    result = wd.update("habeas corpus")
    assert result is None  # window re-filling after reset


def test_windowed_history():
    _, corpus = _make_system()
    wd = WindowedDriftDetector(corpus=corpus, window_size=3)
    for _ in range(3):
        wd.update("habeas corpus")
    assert len(wd.history) == 3


async def test_windowed_async():
    _, corpus = _make_system()
    wd = WindowedDriftDetector(corpus=corpus, window_size=3)
    await wd.aupdate("habeas corpus")
    await wd.aupdate("habeas corpus")
    result = await wd.aupdate("habeas corpus")
    assert isinstance(result, WindowDriftResult)
    assert not result.is_drift


def test_windowed_result_fields():
    _, corpus = _make_system()
    wd = WindowedDriftDetector(corpus=corpus, window_size=3, drift_fraction_threshold=0.25)
    for _ in range(3):
        result = wd.update("habeas corpus")
    assert result.window_size == 3
    assert result.drift_fraction_threshold == 0.25
    assert isinstance(result.threshold, float)
    assert isinstance(result.window_centroid_similarity, float)
